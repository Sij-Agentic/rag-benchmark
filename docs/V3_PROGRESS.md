# V3 Evaluation - Progress Update

**Date:** 2026-09-01  
**Status:** In Progress - Pipeline B Running

---

## ✅ Completed

### 1. Ground Truth Fixes
- **Issue 1:** Adobe 2022 10-K PDF missing (only 2017 available)
  - **Fixed:** Removed from corpus
- **Issue 2:** Some `target_pages` were int instead of list
  - **Fixed:** Converted all to list format
- **Final Corpus:** 70 questions (down from 71)

### 2. Pipeline A Complete ✓
**Results:** 9/70 successful (12.9%)

| Dataset | Successful | Retrieval Hits | Notes |
|---------|-----------|----------------|-------|
| FinanceBench | 8/19 | 8 | Text PDFs processed |
| DocLayNet | 1/1 | 1 | Text PDF processed |
| DocVQA | 0/30 | 0 | Image-only (skipped) |
| ChartQA | 0/20 | 0 | Image-only (skipped) |

**Timing:**
- Avg ingest: 1.28s
- Avg query: 1.60s
- Total: 202s (~3 minutes)

**As Expected:** Pipeline A can only process text-extractable PDFs. All 61 image-only PDFs (DocVQA + ChartQA) correctly skipped with error messages.

---

## 🔄 In Progress

### Pipeline B (LlamaParse) - Running
**Started:** 2026-09-01 13:24 UTC  
**Status:** Processing questions (CPU: 58.9%, Mem: 14.9%)  
**Expected:** 70/70 questions (OCR handles all)  
**Est. Time:** 2-3 hours  
**Completion:** ~15:30-16:30 UTC

**What's happening:**
- LlamaParse parsing each PDF with OCR
- Converting to markdown (preserves structure)
- Jina embeddings on GPU
- FAISS indexing on CPU
- Gemini generation

---

## ⏳ Pending

### Pipeline C (VLM)
**Start:** After Pipeline B completes + GPU clear  
**Expected:** 70/70 questions  
**Est. Time:** 2-3 hours  
**Completion:** ~18:00-19:00 UTC

### LLM-as-Judge Evaluation
**Start:** After all pipelines complete  
**Expected:** 70 × 3 = 210 judgments  
**Est. Time:** 30-60 minutes  
**Script:** `src/llm_judge_v3.py` (ready)

---

## 📊 Expected V3 Results

Based on V2 findings (24 questions):

| Pipeline | V2 Result | V3 Expected | Notes |
|----------|-----------|-------------|-------|
| **A (Naive)** | 8.3% (2/24) | ~13% (9/70) | Only text PDFs |
| **B (LlamaParse)** | 66.7% (16/24) | ~65-70% | Markdown structure helps |
| **C (VLM)** | 79.2% (19/24) | ~75-80% | Best overall |

**By Dataset (Expected):**
- **FinanceBench:** B ≈ C > A (both layout-aware ~60-70%)
- **DocVQA:** C > B > A (VLM wins ~85%, OCR ~65%)
- **ChartQA:** C >> B >> A (VLM dominates ~80%+, visual reading)

---

## 🎯 Today's Goals

### Must Complete ✅
- [x] Fix ground truth issues
- [x] Pipeline A execution
- [ ] Pipeline B execution (in progress)
- [ ] Pipeline C execution
- [ ] LLM-as-judge evaluation
- [ ] Results analysis

### Should Complete 🎯
- [ ] V2 vs V3 comparison
- [ ] Per-dataset breakdown
- [ ] Statistical significance (if time)

---

## 🔧 Technical Notes

### Ground Truth V3
- **File:** `data/ground_truth_v3.json`
- **Questions:** 70 (19 FinanceBench, 30 DocVQA, 20 ChartQA, 1 DocLayNet)
- **Format:** All `target_pages` are lists
- **Missing PDFs:** None (Adobe 2022 removed)

### Sequential Execution Strategy
```bash
# Pipeline A
python src/evaluate.py --pipeline A \
    --ground-truth data/ground_truth_v3.json \
    --output results/v3_pipeline_a.csv

# Clear GPU
python -c "import torch; torch.cuda.empty_cache()"
sleep 5

# Pipeline B (current)
python src/evaluate.py --pipeline B \
    --ground-truth data/ground_truth_v3.json \
    --output results/v3_pipeline_b.csv

# Clear GPU + Pipeline C (next)
```

### Error Monitoring
- Pipeline A: 61/70 errors (expected - image-only PDFs)
- Pipeline B: Monitoring for OOM (lesson from V2)
- Pipeline C: Will monitor for OOM

---

## 📈 Timeline

| Time | Event | Status |
|------|-------|--------|
| 13:09 | Pipeline A started | ✅ Complete |
| 13:12 | Pipeline A complete | ✅ 9/70 successful |
| 13:15 | Ground truth fixed | ✅ 70 questions |
| 13:17 | Pipeline A re-run | ✅ 9/70 successful |
| 13:24 | Pipeline B started | 🔄 Running |
| ~15:30 | Pipeline B expected done | ⏳ Pending |
| ~15:35 | Pipeline C start | ⏳ Pending |
| ~18:00 | Pipeline C expected done | ⏳ Pending |
| ~18:15 | LLM-as-judge start | ⏳ Pending |
| ~19:00 | V3 Complete! | ⏳ Pending |

**Total Est. Time:** ~6 hours from start

---

## 🎓 Lessons Applied from V2

### ✅ Successfully Applied
1. **Sequential Execution** - No simultaneous pipelines (prevents OOM)
2. **GPU Clearing** - Explicit cache clear between pipelines
3. **Error Handling** - Pipeline A correctly skips image-only PDFs
4. **Ground Truth Validation** - Fixed format issues before running

### 🔍 Monitoring
- Watching for CUDA OOM errors
- Checking process status periodically
- Will validate all results before judging

---

**Next Update:** After Pipeline B completes

**Repository:** https://github.com/Sij-Agentic/rag-benchmark  
**Last Updated:** 2026-09-01 13:30 UTC
