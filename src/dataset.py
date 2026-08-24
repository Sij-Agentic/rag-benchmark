"""Build the layout-aware RAG benchmark corpus.

Two sources, each contributing roughly half of a 50-100 page corpus:

FinanceBench (real human-authored Q&A over SEC filings)
    Samples table-heavy Q&A pairs, downloads the corresponding 10-K/10-Q PDFs,
    verifies which PDF page actually holds the cited evidence, and trims each
    PDF down to just the evidence pages (plus a little context).

DocLayNet (layout-complex page images: multi-column papers + manuals)
    Pulls pages from the `scientific_articles` and `manuals` categories.
    DocLayNet ships page *images* with token-level text and bounding boxes, so
    each page is rebuilt as a single-page PDF: the rendered image plus an
    invisible text layer positioned from the DocLayNet boxes. Without that text
    layer a text-only loader (Pipeline A) would extract nothing at all and the
    comparison would measure "has OCR" rather than "handles layout".

    DocLayNet is a layout *detection* dataset and contains no questions, so
    Q&A for these pages must be authored. `--generate-qa` drafts them with
    Gemini; every generated pair is tagged `qa_source="gemini-generated"` and
    needs a human pass before the numbers are trustworthy.

Usage
-----
    python src/dataset.py                 # build everything (no Q&A generation)
    python src/dataset.py --generate-qa   # also draft DocLayNet Q&A via Gemini
    python src/dataset.py --skip-doclaynet
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "raw_pdfs"
FB_PDF_DIR = PDF_DIR / "financebench"
DLN_PDF_DIR = PDF_DIR / "doclaynet"
PAGE_IMG_DIR = DATA_DIR / "pages"
GROUND_TRUTH = DATA_DIR / "ground_truth.json"

FINANCEBENCH_HF = "PatronusAI/financebench"
FINANCEBENCH_PDF_BASE = (
    "https://github.com/patronus-ai/financebench/raw/main/pdfs/{doc_name}.pdf"
)

DOCLAYNET_HF = "pierreguillou/DocLayNet-base"
DOCLAYNET_CONFIG = "DocLayNet_2022.08_processed_on_2023.01"
DOCLAYNET_SPLIT = "test"
DOCLAYNET_CATEGORIES = ("scientific_articles", "manuals")
# DocLayNet-base still ships a legacy loading script, which `datasets` >=4
# refuses to execute, so read the Hub's auto-converted parquet instead.
DOCLAYNET_PARQUET_API = (
    "https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}"
)

SEED = 20250824

# Pages kept on each side of a cited evidence page. Retrieval has to actually
# discriminate, so the evidence page is never the only page in the document.
CONTEXT_PAGES = 1


# ---------------------------------------------------------------------------
# text helpers (used to locate evidence pages inside a PDF)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.,%$()-]*", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    """Distinctive tokens: >=4 chars, lowercased, punctuation-tolerant."""
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 4}


def _overlap(evidence: str, page_text: str) -> float:
    """Fraction of the evidence's distinctive tokens present on the page."""
    ev = _tokens(evidence)
    if not ev:
        return 0.0
    return len(ev & _tokens(page_text)) / len(ev)


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------


@dataclass
class GTItem:
    """One benchmark question, resolved against the PDF the pipelines ingest."""

    id: str
    source_dataset: str
    doc_id: str
    pdf_path: str
    question: str | None
    answer: str | None
    # 1-based page numbers *within pdf_path* -- this is what retrieval is scored on.
    target_pages: list[int]
    layout_challenge: str
    qa_source: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "source_dataset": self.source_dataset,
            "doc_id": self.doc_id,
            "pdf_path": self.pdf_path,
            "question": self.question,
            "answer": self.answer,
            "target_pages": self.target_pages,
            "layout_challenge": self.layout_challenge,
            "qa_source": self.qa_source,
        }
        d.update(self.extra)
        return d


# ---------------------------------------------------------------------------
# FinanceBench
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path, *, timeout: int = 120) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    fh.write(chunk)
            tmp.rename(dest)
        return True
    except Exception as exc:  # noqa: BLE001 - report and continue to next doc
        print(f"    ! download failed ({type(exc).__name__}: {exc})")
        return False


def _select_financebench_rows(rows: list[dict], n: int) -> list[dict]:
    """Pick n table-heavy questions, spread across companies and documents.

    Prefers `metrics-generated` questions (values read straight out of the
    financial statements) because those are the ones a naive text extractor
    mangles when a table's column alignment is lost.
    """
    candidates = [
        r
        for r in rows
        if r.get("evidence")
        and any(e.get("evidence_page_num") is not None for e in r["evidence"])
        and r.get("answer")
    ]
    candidates.sort(
        key=lambda r: (
            0 if "metrics-generated" in (r.get("question_type") or "") else 1,
            r["financebench_id"],
        )
    )

    picked: list[dict] = []
    per_company: defaultdict[str, int] = defaultdict(int)
    per_doc: defaultdict[str, int] = defaultdict(int)
    # Two passes: first spread wide (<=1 per company), then backfill.
    for company_cap, doc_cap in ((1, 1), (2, 2), (99, 99)):
        for r in candidates:
            if len(picked) >= n:
                break
            if r in picked:
                continue
            company, doc = r.get("company") or "?", r["doc_name"]
            if per_company[company] >= company_cap or per_doc[doc] >= doc_cap:
                continue
            picked.append(r)
            per_company[company] += 1
            per_doc[doc] += 1
        if len(picked) >= n:
            break
    return picked[:n]


def build_financebench(n: int) -> list[GTItem]:
    from datasets import load_dataset
    import pypdf
    import pymupdf

    print(f"\n[FinanceBench] loading {FINANCEBENCH_HF} ...")
    ds = load_dataset(FINANCEBENCH_HF, split="train")
    rows = [dict(r) for r in ds]
    print(f"[FinanceBench] {len(rows)} Q&A pairs available")

    selected = _select_financebench_rows(rows, n)
    by_doc: defaultdict[str, list[dict]] = defaultdict(list)
    for r in selected:
        by_doc[r["doc_name"]].append(r)
    print(
        f"[FinanceBench] selected {len(selected)} questions "
        f"across {len(by_doc)} documents"
    )

    items: list[GTItem] = []
    for doc_name, doc_rows in by_doc.items():
        print(f"  - {doc_name} ({len(doc_rows)} question(s))")
        src_pdf = DATA_DIR / "cache" / f"{doc_name}.pdf"
        url = FINANCEBENCH_PDF_BASE.format(doc_name=doc_name)
        if not _download(url, src_pdf):
            fallback = doc_rows[0].get("doc_link")
            if not (fallback and _download(fallback, src_pdf)):
                print("    ! skipping document -- no PDF obtainable")
                continue

        try:
            doc = pymupdf.open(src_pdf)
        except Exception as exc:  # noqa: BLE001
            print(f"    ! unreadable PDF ({exc}); skipping")
            continue
        page_texts = [doc[i].get_text() for i in range(doc.page_count)]
        doc.close()

        # Resolve each cited evidence page to a real 0-based PDF index. The
        # dataset's `evidence_page_num` is a hint only -- it is inconsistently
        # offset (printed page label vs. PDF index), so verify by text overlap
        # and fall back to a whole-document search.
        resolved: dict[str, list[int]] = {}
        confidences: dict[str, float] = {}
        for r in doc_rows:
            hits, scores = [], []
            for ev in r["evidence"]:
                ev_text = ev.get("evidence_text") or ""
                hint = ev.get("evidence_page_num")
                best_idx, best_score = None, 0.0
                # Search near the hint first, then everywhere.
                order: list[int] = []
                if hint is not None:
                    order += [
                        i
                        for i in range(int(hint) - 3, int(hint) + 3)
                        if 0 <= i < len(page_texts)
                    ]
                order += [i for i in range(len(page_texts)) if i not in order]
                for i in order:
                    score = _overlap(ev_text, page_texts[i])
                    if score > best_score:
                        best_idx, best_score = i, score
                    if best_score > 0.95:
                        break
                if best_idx is not None and best_score >= 0.35:
                    hits.append(best_idx)
                    scores.append(best_score)
                elif hint is not None and 0 <= int(hint) < len(page_texts):
                    hits.append(int(hint))
                    scores.append(best_score)
                    print(
                        f"    ~ {r['financebench_id']}: weak evidence match "
                        f"({best_score:.2f}); falling back to hint page {hint}"
                    )
            resolved[r["financebench_id"]] = sorted(set(hits))
            confidences[r["financebench_id"]] = min(scores) if scores else 0.0

        # Trim the document to the union of evidence pages + context.
        keep: set[int] = set()
        for hits in resolved.values():
            for p in hits:
                keep.update(
                    range(
                        max(0, p - CONTEXT_PAGES),
                        min(len(page_texts), p + CONTEXT_PAGES + 1),
                    )
                )
        if not keep:
            print("    ! no evidence pages resolved; skipping document")
            continue
        keep_sorted = sorted(keep)
        old_to_new = {old: i + 1 for i, old in enumerate(keep_sorted)}  # 1-based

        out_pdf = FB_PDF_DIR / f"{doc_name}.pdf"
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        reader = pypdf.PdfReader(str(src_pdf))
        writer = pypdf.PdfWriter()
        for old in keep_sorted:
            writer.add_page(reader.pages[old])
        with open(out_pdf, "wb") as fh:
            writer.write(fh)
        print(f"    kept {len(keep_sorted)} of {len(page_texts)} pages -> {out_pdf.name}")

        for r in doc_rows:
            hits = resolved[r["financebench_id"]]
            if not hits:
                continue
            items.append(
                GTItem(
                    id=r["financebench_id"],
                    source_dataset="financebench",
                    doc_id=doc_name,
                    pdf_path=str(out_pdf.relative_to(PROJECT_ROOT)),
                    question=r["question"],
                    answer=r["answer"],
                    target_pages=[old_to_new[p] for p in hits],
                    layout_challenge="dense nested financial tables",
                    qa_source="financebench-human",
                    extra={
                        "company": r.get("company"),
                        "doc_type": r.get("doc_type"),
                        "doc_period": r.get("doc_period"),
                        "question_type": r.get("question_type"),
                        "justification": r.get("justification"),
                        "evidence_page_num_original": [
                            e.get("evidence_page_num") for e in r["evidence"]
                        ],
                        "source_pdf_pages_original": hits,
                        "page_match_confidence": round(
                            confidences[r["financebench_id"]], 3
                        ),
                    },
                )
            )
    return items


# ---------------------------------------------------------------------------
# DocLayNet
# ---------------------------------------------------------------------------


def _doclaynet_page_to_pdf(row: dict, out_pdf: Path, img_path: Path) -> None:
    """Write a single-page PDF: page image + invisible text layer from bboxes."""
    import pymupdf

    img = row["image"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(img_path, "JPEG", quality=92)

    pw, ph = float(row["original_width"]), float(row["original_height"])
    cw, ch = float(row["coco_width"]), float(row["coco_height"])
    sx, sy = pw / cw, ph / ch

    doc = pymupdf.open()
    page = doc.new_page(width=pw, height=ph)
    page.insert_image(pymupdf.Rect(0, 0, pw, ph), filename=str(img_path))

    # render_mode=3 -> invisible glyphs, so the visual page is untouched while
    # pypdf/LlamaParse still see real, positioned text.
    #
    # insert_text (one show-text op per line) rather than insert_textbox: the
    # textbox variant re-wraps and re-positions each word, and extractors then
    # cannot infer the word gaps -- text comes back as "RecordSelectionfields".
    # A single run per line keeps the spaces intact.
    for text, bbox in zip(row["texts"], row["bboxes_line"]):
        line = " ".join((text or "").split())
        if not line:
            continue
        x, y, w, h = (float(v) for v in bbox)
        x0, y0, x1, y1 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
        bw, bh = x1 - x0, y1 - y0
        if bw <= 1 or bh <= 1:
            continue
        # Fit the font to the box width so glyph positions track the real layout.
        fontsize = max(3.0, min(bh * 0.9, 20.0))
        natural = pymupdf.get_text_length(line, fontname="helv", fontsize=fontsize)
        if natural > bw and natural > 0:
            fontsize = max(1.0, fontsize * bw / natural)
        page.insert_text(
            (x0, y1 - bh * 0.15),  # baseline just above the box bottom
            line,
            fontsize=fontsize,
            fontname="helv",
            render_mode=3,
        )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_pdf))
    doc.close()


def _load_doclaynet_split():
    """Load the DocLayNet split from the Hub's auto-converted parquet files."""
    from datasets import Image as HFImage, load_dataset

    api = DOCLAYNET_PARQUET_API.format(
        repo=DOCLAYNET_HF, config=DOCLAYNET_CONFIG, split=DOCLAYNET_SPLIT
    )
    urls = requests.get(api, timeout=60).json()
    if not isinstance(urls, list) or not urls:
        raise RuntimeError(f"unexpected parquet listing from {api}: {urls!r}")
    print(f"[DocLayNet] {len(urls)} parquet shard(s)")

    # Fetch to disk first: passing huggingface.co URLs straight to load_dataset
    # makes it resolve them as Hub repo paths and fail with "repository not found".
    local: list[str] = []
    cache = DATA_DIR / "cache" / "doclaynet"
    for i, url in enumerate(urls):
        dest = cache / f"{DOCLAYNET_SPLIT}_{i}.parquet"
        if not _download(url, dest):
            raise RuntimeError(f"could not download parquet shard {url}")
        local.append(str(dest))
    ds = load_dataset("parquet", data_files=local, split="train")
    # Parquet gives image as {bytes, path}; cast so rows yield PIL images.
    return ds.cast_column("image", HFImage())


def build_doclaynet(n_per_category: int) -> list[GTItem]:
    print(f"\n[DocLayNet] loading {DOCLAYNET_HF} ({DOCLAYNET_SPLIT}) ...")
    ds = _load_doclaynet_split()

    # Rank by text richness (a page with 3 tokens gives nothing to retrieve) but
    # rank over the WHOLE category, so document diversity is not sacrificed to a
    # single verbose manual dominating the text-rich head of the list.
    categories = ds["doc_category"]
    filenames = ds["original_filename"]
    n_texts = [len(t) for t in ds["texts"]]
    wanted: dict[str, list[int]] = {}
    for cat in DOCLAYNET_CATEGORIES:
        idxs = [i for i, c in enumerate(categories) if c == cat and n_texts[i] >= 20]
        idxs.sort(key=lambda i: -n_texts[i])
        wanted[cat] = idxs
        print(f"[DocLayNet] {cat}: {len(idxs)} usable pages available")

    items: list[GTItem] = []
    for cat, idxs in wanted.items():
        chosen: list[int] = []
        # Round-robin over source documents: take each document's richest page
        # first, only taking a second page from a document once every document
        # has contributed one.
        by_doc: defaultdict[str, list[int]] = defaultdict(list)
        for i in idxs:
            by_doc[filenames[i]].append(i)
        docs = sorted(by_doc, key=lambda d: -n_texts[by_doc[d][0]])
        print(f"[DocLayNet] {cat}: {len(docs)} distinct source document(s)")
        rank = 0
        while len(chosen) < n_per_category and any(
            len(by_doc[d]) > rank for d in docs
        ):
            for d in docs:
                if len(chosen) >= n_per_category:
                    break
                if len(by_doc[d]) > rank:
                    chosen.append(by_doc[d][rank])
            rank += 1

        for i in tqdm(chosen, desc=f"  {cat}", unit="page"):
            row = ds[i]
            stem = f"{cat}_{Path(row['original_filename']).stem}_p{row['page_no']}"
            stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)[:120]
            out_pdf = DLN_PDF_DIR / f"{stem}.pdf"
            img_path = PAGE_IMG_DIR / f"{stem}.jpg"
            _doclaynet_page_to_pdf(row, out_pdf, img_path)

            items.append(
                GTItem(
                    id=f"doclaynet_{row['page_hash'][:12]}",
                    source_dataset="doclaynet",
                    doc_id=stem,
                    pdf_path=str(out_pdf.relative_to(PROJECT_ROOT)),
                    question=None,
                    answer=None,
                    target_pages=[1],  # single-page documents
                    layout_challenge=(
                        "multi-column scientific layout with figures"
                        if cat == "scientific_articles"
                        else "technical manual with callouts and diagrams"
                    ),
                    qa_source="pending",
                    extra={
                        "doc_category": cat,
                        "collection": row["collection"],
                        "original_filename": row["original_filename"],
                        "original_page_no": row["page_no"],
                        "page_image": str(img_path.relative_to(PROJECT_ROOT)),
                        "page_text": "\n".join(row["texts"]),
                        "needs_human_review": True,
                    },
                )
            )
    return items


# ---------------------------------------------------------------------------
# optional: draft Q&A for DocLayNet pages with Gemini
# ---------------------------------------------------------------------------

QA_PROMPT = """You are building a retrieval benchmark over layout-complex documents.

Below is the text of a single page from a {category} document, extracted with
its original reading order. Write ONE question that:
  - is answerable *only* from this page,
  - targets information whose meaning depends on the page's LAYOUT (a table
    cell, a multi-column passage, a figure caption, or a callout box),
  - has a short, unambiguous, verifiable answer (a number, name, or phrase).

Return strict JSON only: {{"question": "...", "answer": "...", "rationale": "..."}}

PAGE TEXT:
{page_text}
"""


def generate_doclaynet_qa(items: list[GTItem], model: str) -> None:
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    targets = [it for it in items if it.source_dataset == "doclaynet"]
    print(f"\n[Q&A] drafting {len(targets)} questions with {model} ...")

    for it in tqdm(targets, desc="  gemini", unit="q"):
        page_text = (it.extra.get("page_text") or "").strip()
        if len(page_text) < 200:
            continue
        prompt = QA_PROMPT.format(
            category=it.extra.get("doc_category", "document"),
            page_text=page_text[:12000],
        )
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            raw = (resp.text or "").strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            qa = json.loads(raw)
            it.question = qa["question"]
            it.answer = qa["answer"]
            it.qa_source = "gemini-generated"
            it.extra["qa_rationale"] = qa.get("rationale")
            it.extra["qa_model"] = model
        except Exception as exc:  # noqa: BLE001 - keep going, leave item pending
            it.extra["qa_error"] = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _page_count(pdf: Path) -> int:
    import pymupdf

    try:
        with pymupdf.open(pdf) as d:
            return d.page_count
    except Exception:  # noqa: BLE001
        return 0


def write_ground_truth(items: list[GTItem]) -> None:
    pdfs = sorted({it.pdf_path for it in items})
    total_pages = sum(_page_count(PROJECT_ROOT / p) for p in pdfs)
    by_source: defaultdict[str, int] = defaultdict(int)
    for it in items:
        by_source[it.source_dataset] += 1

    payload = {
        "metadata": {
            "seed": SEED,
            "context_pages": CONTEXT_PAGES,
            "num_questions": len(items),
            "num_documents": len(pdfs),
            "total_pages": total_pages,
            "questions_by_source": dict(by_source),
            "sources": {
                "financebench": FINANCEBENCH_HF,
                "doclaynet": f"{DOCLAYNET_HF}:{DOCLAYNET_CONFIG}:{DOCLAYNET_SPLIT}",
            },
            "notes": [
                "target_pages are 1-based page numbers within pdf_path, which is "
                "the exact file all three pipelines ingest.",
                "FinanceBench PDFs are trimmed to evidence pages +/- context_pages; "
                "source_pdf_pages_original records the pages in the full filing.",
                "DocLayNet pages are rebuilt as single-page PDFs from the page image "
                "plus an invisible text layer derived from DocLayNet bounding boxes.",
                "Any item with qa_source='gemini-generated' is synthetic and needs "
                "human review before results are reported.",
            ],
        },
        "items": [it.to_json() for it in items],
    }
    GROUND_TRUTH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH.write_text(json.dumps(payload, indent=2))

    print(f"\nWrote {GROUND_TRUTH.relative_to(PROJECT_ROOT)}")
    print(f"  questions : {len(items)} {dict(by_source)}")
    print(f"  documents : {len(pdfs)}")
    print(f"  pages     : {total_pages}")
    pending = sum(1 for it in items if not it.question)
    if pending:
        print(f"  !! {pending} item(s) still have no question (qa_source='pending')")


def main(argv: Iterable[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--financebench-n", type=int, default=10)
    ap.add_argument("--doclaynet-n", type=int, default=10,
                    help="pages per DocLayNet category (default 10 each)")
    ap.add_argument("--skip-financebench", action="store_true")
    ap.add_argument("--skip-doclaynet", action="store_true")
    ap.add_argument("--generate-qa", action="store_true",
                    help="draft DocLayNet Q&A with Gemini (needs GEMINI_API_KEY)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    for d in (FB_PDF_DIR, DLN_PDF_DIR, PAGE_IMG_DIR, DATA_DIR / "cache"):
        d.mkdir(parents=True, exist_ok=True)

    items: list[GTItem] = []
    if not args.skip_financebench:
        items += build_financebench(args.financebench_n)
    if not args.skip_doclaynet:
        items += build_doclaynet(args.doclaynet_n)

    if args.generate_qa:
        if not os.getenv("GEMINI_API_KEY"):
            print("\n! --generate-qa requested but GEMINI_API_KEY is not set; "
                  "leaving DocLayNet questions pending.", file=sys.stderr)
        else:
            generate_doclaynet_qa(items, os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash"))

    if not items:
        print("No items built.", file=sys.stderr)
        return 1
    write_ground_truth(items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
