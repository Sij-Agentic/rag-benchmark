"""Verify the benchmark environment: imports, GPU, Gemini, LlamaParse, pixelshot.

Run after `conda activate rag-benchmark`:

    python scripts/dry_run.py            # skip checks whose API key is missing
    python scripts/dry_run.py --strict   # missing key == failure

Exit code is 0 only if every executed check passed.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_ICON = {PASS: "\033[32m✓\033[0m", FAIL: "\033[31m✗\033[0m", SKIP: "\033[33m-\033[0m"}
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    print(f"  {_ICON[status]} {name:34s} {detail}")


# ---------------------------------------------------------------------------


def check_imports() -> None:
    print("\n[1/6] Python packages")
    mods = {
        "pypdf": "pypdf",
        "langchain": "langchain",
        "langchain_community": "langchain_community",
        "langchain_google_genai": "langchain_google_genai",
        "llama_index.core": "llama-index-core",
        "llama_parse": "llama-parse",
        "faiss": "faiss",
        "google.genai": "google-genai",
        "playwright": "playwright",
        "pixelrag": "pixelrag",
        "pymupdf": "pymupdf",
        "pdf2image": "pdf2image",
        "pandas": "pandas",
        "datasets": "datasets",
    }
    for mod, label in mods.items():
        try:
            m = importlib.import_module(mod)
            record(label, PASS, getattr(m, "__version__", "") or "")
        except Exception as exc:  # noqa: BLE001
            record(label, FAIL, f"{type(exc).__name__}: {exc}")
    # pdf2image is a thin wrapper -- pixelshot's PDF path needs the binaries.
    poppler = shutil.which("pdftoppm")
    record("poppler-utils (pdftoppm)", PASS if poppler else FAIL,
           poppler or "missing: sudo apt-get install -y poppler-utils")


def check_faiss_gpu() -> None:
    print("\n[2/6] FAISS / GPU")
    try:
        import faiss
        import numpy as np

        n_gpu = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
        d, n = 64, 256
        rng = np.random.default_rng(0)
        xb = rng.random((n, d), dtype="float32")
        index = faiss.IndexFlatIP(d)
        if n_gpu > 0:
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        index.add(xb)
        dist, idx = index.search(xb[:4], 3)
        ok = bool((idx[:, 0] == np.arange(4)).all())
        record(
            "faiss round-trip",
            PASS if ok else FAIL,
            f"{'GPU' if n_gpu else 'CPU'} ({n_gpu} gpu) self-retrieval "
            f"{'ok' if ok else 'MISMATCH'}",
        )
    except Exception as exc:  # noqa: BLE001
        record("faiss round-trip", FAIL, f"{type(exc).__name__}: {exc}")


def check_gemini(strict: bool) -> None:
    print("\n[3/6] Gemini API")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        record("GEMINI_API_KEY", FAIL if strict else SKIP, "not set in .env")
        return
    embed_model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    llm_model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")

    try:
        from google import genai

        client = genai.Client(api_key=key)
    except Exception as exc:  # noqa: BLE001
        record("genai client", FAIL, f"{type(exc).__name__}: {exc}")
        return

    try:
        r = client.models.embed_content(model=embed_model, contents="layout-aware RAG")
        dim = len(r.embeddings[0].values)
        record(f"embed ({embed_model})", PASS, f"dim={dim}")
    except Exception as exc:  # noqa: BLE001
        record(f"embed ({embed_model})", FAIL, f"{type(exc).__name__}: {exc}")

    try:
        r = client.models.generate_content(
            model=llm_model, contents="Reply with exactly: OK"
        )
        txt = (r.text or "").strip()
        record(f"generate ({llm_model})", PASS if "OK" in txt else FAIL, f"-> {txt[:40]!r}")
    except Exception as exc:  # noqa: BLE001
        record(f"generate ({llm_model})", FAIL, f"{type(exc).__name__}: {exc}")

    # Pipeline C needs the generator to accept an image alongside the prompt.
    try:
        import pymupdf
        from google.genai import types

        doc = pymupdf.open()
        page = doc.new_page(width=200, height=80)
        page.insert_text((20, 45), "BENCH 42", fontsize=24, fontname="helv")
        png = page.get_pixmap(dpi=120).tobytes("png")
        doc.close()
        r = client.models.generate_content(
            model=llm_model,
            contents=[
                types.Part.from_bytes(data=png, mime_type="image/png"),
                "What number appears in this image? Reply with digits only.",
            ],
        )
        txt = (r.text or "").strip()
        record("vision (image+text)", PASS if "42" in txt else FAIL, f"-> {txt[:40]!r}")
    except Exception as exc:  # noqa: BLE001
        record("vision (image+text)", FAIL, f"{type(exc).__name__}: {exc}")


def check_llamaparse(strict: bool) -> None:
    print("\n[4/6] LlamaParse API")
    key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not key:
        record("LLAMA_CLOUD_API_KEY", FAIL if strict else SKIP, "not set in .env")
        return
    try:
        import pymupdf
        from llama_parse import LlamaParse

        # A tiny table is the cheapest way to prove markdown extraction works.
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 90), "Quarter   Revenue", fontsize=12, fontname="helv")
        page.insert_text((72, 110), "Q1        1,234", fontsize=12, fontname="helv")
        page.insert_text((72, 130), "Q2        5,678", fontsize=12, fontname="helv")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "probe.pdf"
            doc.save(str(p))
            doc.close()
            parser = LlamaParse(api_key=key, result_type="markdown", verbose=False)
            docs = parser.load_data(str(p))
        text = "\n".join(d.text for d in docs)
        ok = bool(text.strip()) and "5,678" in text.replace(" ", "")
        record(
            "parse -> markdown",
            PASS if ok else FAIL,
            f"{len(docs)} doc(s), {len(text)} chars"
            + ("" if ok else " (expected value missing)"),
        )
    except Exception as exc:  # noqa: BLE001
        record("parse -> markdown", FAIL, f"{type(exc).__name__}: {exc}")


def check_pixelshot() -> None:
    print("\n[5/6] pixelshot CLI (PixelRAG rendering)")
    exe = shutil.which("pixelshot")
    if not exe:
        record("pixelshot on PATH", FAIL, "not found")
        return
    record("pixelshot on PATH", PASS, exe)
    try:
        import pymupdf

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 100), "PixelRAG render probe", fontsize=18, fontname="helv")
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "probe.pdf"
            out = Path(td) / "tiles"
            doc.save(str(pdf))
            doc.close()
            proc = subprocess.run(
                [exe, str(pdf), "--output", str(out), "--dpi", "150"],
                capture_output=True, text=True, timeout=300,
            )
            tiles = sorted(out.rglob("*.jpg")) + sorted(out.rglob("*.jpeg"))
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
                record("render pdf -> tiles", FAIL,
                       f"exit {proc.returncode}: {tail[0] if tail else '?'}")
            elif not tiles:
                record("render pdf -> tiles", FAIL, "no tiles produced")
            else:
                record("render pdf -> tiles", PASS,
                       f"{len(tiles)} tile(s), {tiles[0].stat().st_size // 1024} KB")
    except Exception as exc:  # noqa: BLE001
        record("render pdf -> tiles", FAIL, f"{type(exc).__name__}: {exc}")


def check_playwright() -> None:
    print("\n[6/6] Playwright / headless Chromium")
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox"])
            pg = browser.new_page(viewport={"width": 875, "height": 600})
            pg.set_content("<h1 style='font:48px sans-serif'>headless ok</h1>")
            shot = pg.screenshot()
            browser.close()
        record("chromium screenshot", PASS if len(shot) > 1000 else FAIL,
               f"{len(shot) // 1024} KB png")
    except Exception as exc:  # noqa: BLE001
        record("chromium screenshot", FAIL, f"{type(exc).__name__}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strict", action="store_true",
                    help="treat missing API keys as failures")
    args = ap.parse_args()

    env_file = PROJECT_ROOT / ".env"
    load_dotenv(env_file)
    print("=" * 68)
    print("Layout-Aware RAG Benchmark -- environment dry run")
    print(f"python  : {sys.version.split()[0]}  ({sys.executable})")
    print(f".env    : {'loaded' if env_file.exists() else 'MISSING (cp .env.example .env)'}")
    print("=" * 68)

    check_imports()
    check_faiss_gpu()
    check_gemini(args.strict)
    check_llamaparse(args.strict)
    check_pixelshot()
    check_playwright()

    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print("\n" + "=" * 68)
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_fail:
        print("\nFailed checks:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  - {name}: {detail}")
    if n_skip:
        print("\nSkipped (add the key to .env, then re-run):")
        for name, status, detail in results:
            if status == SKIP:
                print(f"  - {name}: {detail}")
    print("=" * 68)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
