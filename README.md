# Layout-Aware RAG Benchmark

Comparing three document-retrieval paradigms on layout-complex PDFs (dense
tables, multi-column text, embedded diagrams):

| Pipeline | Ingestion | Retrieval | Generation | Status |
|---|---|---|---|---|
| **A** — naive baseline | `PyPDFLoader` | Gemini text embeddings + FAISS | Gemini | scaffold |
| **B** — extraction | LlamaParse → markdown → `MarkdownNodeParser` | Gemini text embeddings + FAISS | Gemini | scaffold |
| **C** — vision | `pixelshot` tile rendering | multimodal embeddings + FAISS | Gemini (VLM) | scaffold |

Environment and corpus are built and verified; the three pipelines and the
evaluation harness are not implemented yet.

## Setup

```bash
conda create -n rag-benchmark -c conda-forge --override-channels python=3.12 -y
conda activate rag-benchmark
pip install -r requirements.txt
playwright install chromium

# Headless Chromium + PDF rasterisation system libraries (Ubuntu 24.04)
sudo apt-get install -y libgbm1 libnss3 libasound2t64 libxss1 \
    libappindicator3-1 libindicator7 fonts-liberation \
    libatk-bridge2.0-0t64 libgtk-3-0t64 poppler-utils

cp .env.example .env   # then fill in LLAMA_CLOUD_API_KEY and GEMINI_API_KEY
```

On Ubuntu 24.04 the `t64` package names above replace the pre-24.04
`libasound2` / `libatk-bridge2.0-0` / `libgtk-3-0`. `poppler-utils` is required
because `pixelshot` rasterises PDFs through `pdf2image`, not PyMuPDF.

## Verify the environment

```bash
python scripts/dry_run.py          # 23 checks: imports, FAISS/GPU, Gemini, LlamaParse, pixelshot, Playwright
python scripts/dry_run.py --strict # treat missing API keys as failures
```

## Build the corpus

```bash
python src/dataset.py                # FinanceBench + DocLayNet -> data/ground_truth.json
python src/dataset.py --generate-qa  # additionally draft DocLayNet Q&A with Gemini
```

Produces 56 pages across 30 single-purpose documents:

- **FinanceBench** (10 questions, 36 pages) — SEC 10-K filings trimmed to the
  cited evidence pages ±1 page of context. Questions and answers are the
  dataset's own human-authored pairs.
- **DocLayNet** (20 pages, 10 scientific articles + 10 manual pages) — each page
  rebuilt as a single-page PDF from the page image plus an *invisible* text layer
  positioned from DocLayNet's bounding boxes.

`data/` is git-ignored; re-run `src/dataset.py` to regenerate it.

### Two caveats on the DocLayNet half

1. **DocLayNet has no questions.** It is a layout *detection* dataset, so Q&A for
   these 20 pages must be authored. `--generate-qa` drafts them with Gemini and
   tags them `qa_source="gemini-generated"`; they need a human pass before any
   result is trustworthy. A dataset with real Q&A over document images
   (DocVQA, TAT-DQA) would remove this step entirely.
2. **The text layer is synthesised.** DocLayNet ships page images, so a
   text-only loader would otherwise extract *nothing* and Pipeline A would score
   0 — measuring "has OCR" rather than "handles layout". The reconstructed layer
   makes the comparison meaningful, but reading order is DocLayNet's, not the
   original PDF's.

## Ground truth format

`data/ground_truth.json` — `target_pages` are 1-based page numbers **within
`pdf_path`**, the exact file all three pipelines ingest, so retrieval hit-rate is
directly scoreable.

```json
{
  "id": "financebench_id_03029",
  "source_dataset": "financebench",
  "pdf_path": "data/raw_pdfs/financebench/3M_2018_10K.pdf",
  "question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
  "answer": "$1577.00",
  "target_pages": [2],
  "layout_challenge": "dense nested financial tables",
  "qa_source": "financebench-human",
  "source_pdf_pages_original": [59],
  "page_match_confidence": 1.0
}
```

`page_match_confidence` is the token overlap between the dataset's cited
evidence text and the page the builder resolved it to — the filing's page
numbering does not always line up with the PDF index, so every citation is
verified by text search rather than trusted.

## Layout

```
rag-benchmark/
├── .env.example          # template; copy to .env (git-ignored)
├── requirements.txt
├── data/
│   ├── raw_pdfs/         # financebench/ + doclaynet/  (generated)
│   ├── pages/            # DocLayNet page images       (generated)
│   └── ground_truth.json
├── src/
│   ├── dataset.py        # corpus builder
│   ├── pipeline_a.py     # naive text            (not implemented)
│   ├── pipeline_b.py     # LlamaParse            (not implemented)
│   ├── pipeline_c.py     # PixelRAG              (not implemented)
│   └── evaluate.py       # harness               (not implemented)
├── scripts/
│   └── dry_run.py        # environment verification
└── results/
    └── benchmark.csv     # generated
```
