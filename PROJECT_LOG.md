# Layout-Aware RAG Benchmark — Project Log

## Project Goal

Compare three document ingestion methods for layout-complex PDFs:
- **Pipeline A (Naive)**: PyPDFLoader → text embeddings → FAISS
- **Pipeline B (LlamaParse)**: LlamaParse markdown → text embeddings → FAISS  
- **Pipeline C (PixelRAG)**: Visual tile rendering → multimodal embeddings → FAISS

All three use Gemini for embeddings and generation.

---

## Session 1: Environment Setup & Corpus Building (2026-08-24)

### What Was Accomplished

#### 1. Environment Setup ✓
- **Conda environment**: `rag-benchmark` with Python 3.12.14
- **Installation source**: conda-forge (not Anaconda defaults — see Decision #1 below)
- **All dependencies installed and verified**:
  - Core: pypdf, langchain, langchain-community, langchain-google-genai
  - LlamaParse: llama-index, llama-parse, llama-cloud-services
  - PixelRAG: pixelrag[playwright], pdf2image, cef-capi-py (124MB CEF bundle)
  - Vector store: faiss-gpu 1.15.0 (verified on Tesla T4, 1 GPU detected)
  - Dataset tools: datasets 5.0.1, huggingface-hub
  - Rendering: playwright, chromium headless-shell installed

#### 2. System Dependencies ✓
Installed on Ubuntu 24.04:
```bash
libgbm1 libnss3 libasound2t64 libxss1 libappindicator3-1 
libindicator7 fonts-liberation libatk-bridge2.0-0t64 libgtk-3-0t64
poppler-utils  # required for pixelshot PDF rendering
```

**Note**: Ubuntu 24.04 uses `t64` suffixed package names (time64 transition). The brief's original names don't exist on this OS version.

#### 3. API Verification ✓
All endpoints tested successfully (23/23 checks passed):

| Service | Model/Endpoint | Status | Notes |
|---------|---------------|--------|-------|
| Gemini Embeddings | `gemini-embedding-2-preview` | ✓ | 3072-dim vectors |
| Gemini Generation | `gemini-3.7-flash` | ✓ | Text generation OK |
| Gemini Vision | `gemini-3.7-flash` | ✓ | Image+text multimodal verified |
| LlamaParse | markdown extraction | ✓ | Probe table → 92 chars markdown |
| pixelshot | PDF → tiles | ✓ | 1-page PDF → 1 tile (1240×1755 JPEG) |
| Playwright | headless Chromium | ✓ | Screenshot rendered |
| FAISS GPU | index + search | ✓ | Round-trip retrieval working |

#### 4. Corpus Built ✓

**Final corpus: 56 pages across 30 documents**

##### FinanceBench Subset (10 questions, 36 pages)
- **Source**: PatronusAI/financebench Hugging Face dataset (150 Q&A pairs total)
- **Selection criteria**: 
  - Table-heavy "metrics-generated" questions (values extracted from financial statements)
  - Spread across 10 companies, 10 distinct 10-K/10-Q filings
  - Evidence pages resolved by **text overlap verification** (not blind trust of `evidence_page_num`)
- **Quality metrics**:
  - 8/10 questions: page match confidence ≥0.90
  - 9/10 questions: gold answer appears verbatim on target page
  - 1/10 question: derived answer (requires arithmetic on table values)
- **Processing**: Each filing trimmed to evidence pages ±1 context page

**Documents included**:
```
BESTBUY_2017_10K           3 pages
CORNING_2021_10K           3 pages
ACTIVISIONBLIZZARD_2019_10K 4 pages
3M_2018_10K                3 pages
LOCKHEEDMARTIN_2021_10K    3 pages
AMD_2015_10K               6 pages
NETFLIX_2017_10K           3 pages
GENERALMILLS_2020_10K      3 pages
COCACOLA_2017_10K          5 pages
NIKE_2019_10K              3 pages
```

##### DocLayNet Subset (20 pages, no questions yet)
- **Source**: pierreguillou/DocLayNet-base test split (499 pages available)
- **Categories**: 
  - 10 pages: scientific articles (multi-column papers with figures) from 10 distinct arXiv sources
  - 10 pages: technical manuals (callouts, diagrams) from 5 distinct IBM/other manuals
- **Document diversity**: Round-robin selection ensures 10 scientific pages come from 10 different papers
- **Text layer reconstruction**:
  - Each page rebuilt as single-page PDF: page image + invisible text layer
  - Text positioned from DocLayNet's bounding boxes using PyMuPDF render_mode=3
  - **Validation**: avg 2,499 chars/page extractable by pypdf, 0 empty pages
  - Word spacing preserved (switched from `insert_textbox` to `insert_text` to prevent "RecordSelectionfields" artifacts)

**Documents included**:
```
Scientific articles (arXiv):
  1001.0266, 1001.0764, 1001.0788, 1001.1732, 1001.1785
  1001.2120, 1002.1045, 1110.4445, 1410.5885, 1611.06217

Manuals (2 pages each):
  IBM-i-s5445349.pdf, sg247938.pdf, sg248459.pdf, 
  sg246915.pdf, basic-english-language-skills.PDF
```

#### 5. Ground Truth Format

`data/ground_truth.json` structure:
```json
{
  "metadata": {
    "num_questions": 30,
    "num_documents": 30,
    "total_pages": 56,
    "questions_by_source": {"financebench": 10, "doclaynet": 20}
  },
  "items": [
    {
      "id": "financebench_id_03029",
      "source_dataset": "financebench",
      "pdf_path": "data/raw_pdfs/financebench/3M_2018_10K.pdf",
      "question": "What is the FY2018 capital expenditure...",
      "answer": "$1577.00",
      "target_pages": [2],  // 1-based within pdf_path
      "layout_challenge": "dense nested financial tables",
      "qa_source": "financebench-human",
      "page_match_confidence": 1.0
    }
  ]
}
```

**Key design choice**: `target_pages` are 1-based page numbers **within the trimmed `pdf_path`** that all pipelines ingest, so retrieval hit-rate is directly comparable. FinanceBench items also record `source_pdf_pages_original` (the page number in the full 160-page filing) for reference.

#### 6. Project Structure ✓
```
rag-benchmark/
├── .env.example          # Template (committed)
├── .env                  # Real keys (git-ignored)
├── README.md             # Setup & usage docs
├── requirements.txt      # Pinned dependencies
├── data/
│   ├── raw_pdfs/         # 30 PDFs (git-ignored, 7MB)
│   │   ├── financebench/ # 10 trimmed 10-Ks
│   │   └── doclaynet/    # 20 reconstructed pages
│   ├── pages/            # DocLayNet page images (git-ignored, 3.7MB)
│   ├── cache/            # Parquet shards, downloaded PDFs (192MB)
│   └── ground_truth.json # 92KB, committed
├── src/
│   ├── dataset.py        # Corpus builder (implemented ✓)
│   ├── pipeline_a.py     # Naive baseline (scaffold only)
│   ├── pipeline_b.py     # LlamaParse (scaffold only)
│   ├── pipeline_c.py     # PixelRAG (scaffold only)
│   └── evaluate.py       # Harness (scaffold only)
├── scripts/
│   └── dry_run.py        # Environment verification (implemented ✓)
└── results/
    └── benchmark.csv     # Output (not generated yet)
```

#### 7. Git Status
- **Commit**: `23c3674` "Scaffold layout-aware RAG benchmark"
- **Branch**: `main`
- **Remote**: `https://github.com/Sij-Agentic/rag-benchmark.git` (public repo)
- **Status**: Committed locally, **not pushed yet** (waiting for git credentials)

---

## Key Decisions & Deviations from Original Brief

### Decision #1: Conda-Forge vs Anaconda Defaults
**Issue**: `conda create` failed with "Terms of Service have not been accepted for https://repo.anaconda.com"  
**Decision**: Built from `conda-forge` channel instead (`--override-channels`)  
**Impact**: None on functionality; all packages resolved correctly  
**Rationale**: Avoided accepting legal agreement on user's behalf

### Decision #2: DocLayNet Parquet Workaround
**Issue**: `datasets` ≥4.0 dropped support for legacy dataset loading scripts  
**Error**: `RuntimeError: Dataset scripts are no longer supported, but found DocLayNet-base.py`  
**Decision**: Download Hub's auto-converted parquet shards to local cache, load from disk  
**Impact**: 2-second download + local cache storage; dataset fully usable  
**Alternative considered**: Pin `datasets<4.0` (rejected: too restrictive)

### Decision #3: FinanceBench PDF Source
**Original brief**: Download from `doc_link` (investor relations sites)  
**Decision**: Pull from `https://github.com/patronus-ai/financebench/raw/main/pdfs/`  
**Rationale**: All 368 filings available, reliable, no rate limits  
**Impact**: 10/10 PDFs downloaded successfully (1 would have failed from IR site)

### Decision #4: Evidence Page Resolution Method
**Original brief**: Use `evidence_page_num` from dataset  
**Decision**: Verify via text overlap, fall back to hint only if confidence ≥0.35  
**Rationale**: Page numbering inconsistencies (printed labels vs PDF indices)  
**Results**: 8/10 matches at ≥0.90 confidence, 2 at 0.59-0.72 (still validated by gold answer presence)

### Decision #5: DocLayNet Text Layer Method
**Original approach**: `insert_textbox()` per line  
**Issue**: Word spacing lost in extraction (`"Table34.SpecifyRecordSelection"`)  
**Decision**: `insert_text()` with width-fitted font sizing  
**Impact**: Clean word-separated text (avg 2,499 chars/page), preserves layout fidelity for Pipeline A

### Decision #6: DocLayNet Document Diversity
**Original approach**: Sort by text richness, take top N per category  
**Issue**: All 4 manuals sampled came from `IBM-i-s5445349.pdf`  
**Decision**: Round-robin across source documents (take richest page from each doc before taking a 2nd page from any doc)  
**Impact**: 10 scientific pages from 10 papers; 10 manual pages from 5 sources (max 2 per source)

---

## Technical Findings & Notes

### pixelshot Rendering Backend
**Clarification needed**: The brief stated "Playwright visual tile rendering" for Pipeline C.  

**Reality**: 
- `pixelshot` CLI has two backends: `cdp` (default) and `playwright`
- These backends apply **only to URL/HTML inputs**
- For PDF inputs, `pixelshot` always uses **PyMuPDF rasterization** via `pdf2image` + `poppler-utils`
- Playwright/CDP are never invoked for the PDF corpus

**Command**: `pixelshot <pdf> --dpi 200 --output tiles/` produces:
```
tiles/<stem>.tiles/
  ├── tile_0000.jpg     # Tall JPEG strips (default: 8192px height)
  ├── tile_0001.jpg     # (if page was tall enough to need splitting)
  ├── tiles.json        # Tile metadata + page attribution
  └── chunks.json       # Chunk boundaries (for text-aware splitting)
```

### Gemini Embedding Model Clarification
**User's .env configuration**:
```bash
GEMINI_EMBED_MODEL=gemini-embedding-2-preview  # 3072-dim multimodal
GEMINI_LLM_MODEL=gemini-3.7-flash
```

**Verified working** in dry run (see API Verification table above).

**Previous confusion in my summary**: I generically referenced "gemini-embedding-001" when discussing the lack of image embedding in the *Developer API*. This was imprecise — the user correctly configured `gemini-embedding-2-preview`, which supports both text and image inputs and is the right choice for all three pipelines.

### Pipeline C Embedding Strategy
The user has already selected **`gemini-embedding-2-preview`**, which **does support multimodal (image) embeddings**. This resolves the "decision needed" I mentioned.

**Pipeline C flow**:
1. `pixelshot` renders PDF pages → JPEG tiles
2. Tiles embedded via `gemini-embedding-2-preview` (image mode)
3. FAISS index stores tile embeddings
4. Query: embed query text → retrieve top-k tiles → pass tile images to `gemini-3.7-flash` (vision)

**No local embedding model needed** — Gemini handles both text and image embeddings.

### FAISS GPU Performance
- **Hardware**: Tesla T4 (15GB)
- **Library**: faiss-gpu 1.15.0 (manylinux cp310-abi3 wheel, imports on py3.12)
- **Test**: 256 vectors (dim=64), IndexFlatIP, round-trip search
- **Result**: GPU detected (1), self-retrieval 4/4 correct

For 56 pages → ~100-500 chunks, CPU FAISS would be sufficient, but GPU is working and available.

---

## Critical Issues Requiring Decision

### Issue #1: DocLayNet Has No Questions ⚠️
**Status**: 20 DocLayNet items have `qa_source: "pending"`, `question: null`, `answer: null`

**Options**:
1. **Manual authoring**: Write 20 questions by inspecting the page images/text  
   - Time: ~2-3 hours  
   - Quality: High  

2. **`--generate-qa` with Gemini**: Draft questions via VLM  
   - Time: ~2 minutes  
   - Quality: Needs human review; synthetic Q&A + LLM judge = soft benchmark  

3. **Replace with DocVQA or TAT-DQA**: Use a dataset that ships with real human Q&A over document images  
   - Time: ~1 hour to integrate  
   - Quality: High, removes synthetic artifact  

**Recommendation**: If this benchmark's goal is publication or internal decision-making, **Option 3 (DocVQA)** removes a major validity concern. If it's a proof-of-concept for the comparison method, Option 2 is acceptable with caveats documented.

**Current blocker**: Cannot run evaluation harness until DocLayNet questions exist.

### Issue #2: HuggingFace Token Exposure 🔒
**What happened**: During output, I printed the user's `HF_TOKEN` value in full (`hf_...REDACTED`). My redaction regex only matched lines ending in `…KEY=`, so `HF_TOKEN=` slipped through.

**Impact**: Token is now in this transcript.

**Required action**: Rotate the token at https://huggingface.co/settings/tokens

**Status of other secrets**: 
- `GEMINI_API_KEY`: Redacted in all outputs ✓
- `LLAMA_CLOUD_API_KEY`: Redacted in all outputs ✓
- `.env` file: Git-ignored, not staged, not in commit ✓

---

## What's NOT Done Yet

### Pipeline Implementations (All 3)
Each needs:
```python
def ingest(pdf_paths: list[Path], config: dict) -> Index:
    """Build searchable index from PDFs."""
    
def query(index: Index, question: str, k: int = 5) -> Answer:
    """Retrieve + generate answer."""
    # Returns: Answer(text: str, retrieved_pages: list[int], context: str)
```

**Shared interface** allows `evaluate.py` to treat all three identically.

### Evaluation Harness (`src/evaluate.py`)
Needs to:
1. Load `ground_truth.json`
2. For each (pipeline, item): ingest PDF → query → score
3. Compute metrics:
   - **Retrieval hit@k**: Did any retrieved chunk come from `target_pages`?
   - **Answer correctness**: LLM-as-judge (question, gold, prediction) + numeric exact-match for FinanceBench
   - **Latency**: Ingest time (amortized), query time
   - **Cost**: Token usage × pricing + LlamaParse page credits
4. Save per-item results to `results/benchmark.csv`
5. Report aggregates by `source_dataset` and `layout_challenge`

---

## Next Steps (Recommended Order)

### Immediate (Unblock Progress)
1. **Push to GitHub**: User has configured git, can now run `git push -u origin main`
2. **Rotate HF_TOKEN**: https://huggingface.co/settings/tokens

### Phase 1: Baseline Pipeline A
1. Implement `pipeline_a.py`:
   ```python
   PyPDFLoader → RecursiveCharacterTextSplitter(chunk_size=1000) 
   → batch embed via gemini-embedding-2-preview 
   → FAISS IndexFlatIP 
   → retrieve top-5 → concat context → Gemini generate
   ```
2. Implement basic `evaluate.py`:
   - Just FinanceBench 10 questions (skip DocLayNet for now)
   - Retrieval hit@5 + manual answer review (no LLM judge yet)
3. **Outcome**: Baseline numbers on table-heavy 10-Ks with naive text extraction

### Phase 2: LlamaParse Pipeline B
1. Implement `pipeline_b.py`:
   ```python
   LlamaParse(result_type="markdown", cache=True) 
   → MarkdownNodeParser 
   → embed → FAISS → retrieve → generate
   ```
2. Add LlamaParse caching to `data/cache/llamaparse/` (avoid re-billing)
3. Run against same 10 FinanceBench questions
4. **Outcome**: Measure improvement from markdown table preservation

### Phase 3: Vision Pipeline C
1. Implement `pipeline_c.py`:
   ```python
   pixelshot(dpi=200) → tile JPEGs 
   → embed tiles via gemini-embedding-2-preview (image mode)
   → FAISS → retrieve tiles → Gemini vision (image+question)
   ```
2. Handle tile-to-page attribution (from `tiles.json`)
3. **Outcome**: Full three-way comparison on FinanceBench

### Phase 4: DocLayNet Integration (If Needed)
1. **Decide** on Issue #1 (manual Q&A, Gemini draft, or DocVQA replacement)
2. Extend harness to full 30-question corpus
3. Add LLM-as-judge for answer scoring
4. **Outcome**: Complete benchmark with layout variety (tables + multi-column + diagrams)

### Phase 5: Analysis & Write-Up
1. Cost analysis (token pricing, LlamaParse credits)
2. Per-challenge breakdown (tables vs multi-column vs diagrams)
3. Error analysis (what layout patterns does each pipeline miss?)
4. Article draft using this log as narrative backbone

---

## Resource Tracking

### Disk Usage
```
data/raw_pdfs/      7.0 MB   (30 PDFs, git-ignored)
data/pages/         3.7 MB   (20 page images, git-ignored)
data/cache/       192.0 MB   (parquet, source PDFs, git-ignored)
data/ground_truth.json  92 KB   (committed)
```

### API Costs Incurred So Far
- **Gemini**: Dry run test calls only (~5 requests, negligible)
- **LlamaParse**: 1 probe table parse (~1 page credit)
- **Hugging Face**: Dataset downloads (free)

### API Costs Projected for Full Run
Assuming 30 questions × 3 pipelines = 90 query cycles:

**Pipeline A (per question)**:
- Embed: ~10 chunks × 1K tokens → 10K input tokens
- Generate: ~5K context + 1K output → 6K tokens
- **Per question**: ~16K tokens

**Pipeline B (per question)**:
- LlamaParse: 3-6 pages (one-time per doc, amortized) → ~$0.02/doc
- Embed: ~8 markdown chunks → 8K tokens  
- Generate: ~4K context + 1K output → 5K tokens
- **Per question**: ~13K tokens + $0.02 parse

**Pipeline C (per question)**:
- Embed: ~6 tiles (image embeddings, unclear token equiv)
- Generate: 5 tile images + question (multimodal input) → ?K tokens
- **Per question**: TBD (need to test image token pricing)

**Rough total**: ~1M tokens + ~$0.60 LlamaParse credits (10 docs × $0.02 + 20 DocLayNet pages × $0.02)

At Gemini Experimental API pricing (free tier generous), cost should be negligible.

---

## Questions for User

1. **DocLayNet Q&A** (Issue #1): Manual author, Gemini draft, or replace with DocVQA?
2. **Scope for first implementation**: Should I start with Pipeline A + just the FinanceBench 10 questions to get baseline numbers fast, or implement all three pipelines first?
3. **LLM judge model**: Use `gemini-3.7-flash` or a different model for answer correctness grading?
4. **Git push**: You mentioned configuring git — do you want me to try `git push -u origin main` now, or will you handle the push?

---

## Article Writing Notes

### Narrative Arc (Suggested)
1. **Motivation**: Why layout matters (table mangling example from FinanceBench)
2. **Three approaches**: Text extraction, structured extraction (markdown), vision-first
3. **Corpus design**: Small (56 pages) but high-complexity (nested tables, multi-column)
4. **Implementation challenges**: 
   - DocLayNet text layer reconstruction
   - FinanceBench evidence page resolution
   - pixelshot rendering quirks
5. **Results**: [TBD after evaluation runs]
6. **Cost/latency tradeoffs**: [TBD]
7. **When to use each**: Decision tree based on document characteristics

### Key Figures to Generate
1. Example of Pipeline A's table mangling (before/after text extraction)
2. LlamaParse markdown output vs raw PyPDF text (side-by-side)
3. Visual tile + retrieval result for Pipeline C
4. Accuracy by layout type (table vs multi-column vs diagram)
5. Cost vs accuracy scatterplot (3 pipelines)

---

## Session End Status

**Ready to proceed with**:
- ✓ Environment fully working (23/23 checks)
- ✓ Corpus built (56 pages, 30 questions — 20 pending Q&A)
- ✓ Git committed locally
- ⏸ Awaiting user direction on next phase

**Blockers**:
- DocLayNet questions (Issue #1) — blocks full 30-question eval
- Git push credentials — user configured, may be unblocked

**Recommended immediate next step**: Implement Pipeline A + basic harness on FinanceBench-only to get first results.
