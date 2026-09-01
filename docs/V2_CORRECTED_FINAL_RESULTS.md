# V2 Corpus - CORRECTED Final Results

**Date:** 2026-08-31  
**Status:** ✅ Complete (Sequential Clean Run)  
**Evaluation Method:** LLM-as-Judge (Gemini 2.5 Flash)

---

## ⚠️ IMPORTANT: Previous Results Were Invalid

**Issue Discovered:** Pipeline B and C were run simultaneously, causing CUDA Out of Memory errors on Pipeline C's FinanceBench questions (7/8 failed with OOM).

**Solution:** Re-ran both pipelines sequentially (one at a time) to eliminate resource contention.

**Impact:** Results changed significantly!

---

## Final Accuracy (Sequential Clean Run)

| Pipeline | Overall | FinanceBench (Tables) | DocVQA (Forms) | DocLayNet |
|----------|---------|----------------------|----------------|-----------|
| **A (Naive)** | 8.3% (2/24) | 25.0% (2/8) | 0% (skipped) | 0% (0/1) |
| **B (LlamaParse)** | 66.7% (16/24) | 62.5% (5/8) | 66.7% (10/15) | 100% (1/1) |
| **C (VLM)** | **79.2% (19/24)** 🏆 | 62.5% (5/8) | **86.7% (13/15)** 🏆 | 100% (1/1) |

### Retrieval Performance (Clean Run)

| Pipeline | Overall Retrieval@5 |
|----------|---------------------|
| **A** | 37.5% (9/24) - Skipped image-only PDFs |
| **B** | **100% (24/24)** - Perfect retrieval |
| **C** | **100% (24/24)** - Perfect retrieval |

---

## Key Findings (CORRECTED)

### 1. Pipeline C (VLM) Wins Overall

**Overall Accuracy:**
- Pipeline C: 79.2% (best)
- Pipeline B: 66.7%
- Pipeline A: 8.3%

**Why C Wins:**
- Excels at DocVQA forms (86.7% vs B's 66.7%)
- Ties with B on FinanceBench (both 62.5%)
- Perfect retrieval (100%)

### 2. FinanceBench: Both Layout-Aware Methods Tie

**FinanceBench (Multi-column Financial Tables):**
- Naive (A): 25% → **Layout-unaware fails**
- LlamaParse (B): 62.5% → **Markdown helps (+37.5%)**
- VLM (C): 62.5% → **Visual understanding helps (+37.5%)**

**Key Insight:** Both layout-aware approaches provide similar improvement over naive baseline on financial tables. No clear winner between LlamaParse and VLM for this task.

### 3. DocVQA: VLM Dominates

**DocVQA (Forms with Spatial Layouts):**
- Naive (A): 0% → Image-only PDFs, cannot process
- LlamaParse (B): 66.7% → OCR works, but loses spatial context
- VLM (C): 86.7% → **Reads visual layout natively (+20%)**

**Key Insight:** VLM has a clear advantage on forms where spatial positioning matters.

### 4. Retrieval vs Extraction

Both B and C achieved **perfect 100% retrieval** when run without resource contention. The differentiation is entirely in the **extraction phase**:

- **Retrieval bottleneck:** Pipeline A (37.5%) - cannot handle image-only PDFs
- **Extraction bottleneck:** All pipelines - even with correct pages, layout affects answer quality

---

## Comparison: Invalid vs Corrected Results

### Previous Results (Invalid - OOM Contaminated)

| Pipeline | Overall | FinanceBench | DocVQA |
|----------|---------|--------------|--------|
| B | 75.0% | **87.5%** | 66.7% |
| C | 58.3% | **0%** (OOM!) | 86.7% |

### Corrected Results (Sequential Clean Run)

| Pipeline | Overall | FinanceBench | DocVQA |
|----------|---------|--------------|--------|
| B | 66.7% | **62.5%** | 66.7% |
| C | **79.2%** | **62.5%** | **86.7%** |

**Changes:**
- Pipeline B FinanceBench: 87.5% → 62.5% (also had 2 non-deterministic answer changes)
- Pipeline C FinanceBench: 0% → 62.5% (OOM fixed)
- Pipeline C overall: 58.3% → 79.2% (winner!)

---

## Technical Details

### Why Results Changed

**1. CUDA OOM on FinanceBench (Pipeline C)**
- Running B and C simultaneously exhausted 15GB T4 GPU
- FinanceBench PDFs are large (10-K reports, 20-50 pages)
- Pipeline C hit OOM during Jina embedding on 7/8 FinanceBench questions
- **Fix:** Sequential execution (run B, clear GPU, run C)

**2. Non-Deterministic Generation (Pipeline B)**
- 2 questions (Netflix, PayPal) had different answers between runs
- LLM generation not fully deterministic despite temperature=0
- 7/8 (previous) vs 5/8 (clean) on FinanceBench
- **Note:** This highlights evaluation variability

### Corpus Composition

**Total: 24 Questions**

1. **FinanceBench (8 questions)** - Multi-column financial tables
   - 3M, Netflix, PepsiCo, CVS Health, PayPal, JPMorgan, Verizon, Ulta Beauty

2. **DocVQA (15 questions)** - Forms/invoices with spatial layouts
   - All image-only PDFs (no text layer)
   - Field-value relationships depend on visual positioning

3. **DocLayNet (1 question)** - Scientific article
   - Multi-column layout with equation extraction

### Pipeline Configurations

**Pipeline A (Naive):**
- PyPDFLoader (no layout awareness)
- Cannot handle image-only PDFs
- Jina embeddings + FAISS + Gemini

**Pipeline B (LlamaParse):**
- LlamaParse with OCR
- Markdown representation preserves structure
- Jina embeddings (GPU) + FAISS (CPU) + Gemini

**Pipeline C (VLM):**
- Pixelshot (PDF → page images)
- Jina text embeddings for retrieval
- Gemini 2.5 Flash VLM for generation
- Native visual understanding

---

## Updated Recommendations

### For Practitioners

**When to use each approach:**

1. **Naive baseline (PyPDF):**
   - Simple documents with clean text
   - **Accuracy ceiling:** ~25% on complex layouts
   - **Skip if:** Image-only PDFs, complex tables, forms

2. **LlamaParse:**
   - Multi-column tables
   - Financial statements
   - Text-heavy documents
   - **Best for:** Structured data extraction (ties with VLM at 62.5% on tables)
   - **Weakness:** Loses spatial context on forms (66.7% vs VLM's 86.7%)

3. **VLM (Vision-Language Models):**
   - Forms with spatial layouts **(BEST: 86.7%)**
   - Image-only documents
   - Visual field-value relationships
   - **Best for:** Overall accuracy (79.2%)
   - **Strength:** Competitive on tables (62.5%), dominant on forms (86.7%)

### Updated Conclusion

**Previous Conclusion (WRONG):**
> LlamaParse dominates on tables (87.5%), VLM dominates on forms (86.7%)

**Corrected Conclusion:**
> **VLM is the overall winner (79.2%)** - it TIES with LlamaParse on tables (62.5%) and DOMINATES on forms (86.7%). Unless you specifically need markdown-based processing or have constraints on image rendering, VLM approach provides the best overall accuracy.

---

## Detailed Results by Question

### FinanceBench Performance (Both B and C: 5/8 = 62.5%)

| Question | Topic | A | B | C | Winner |
|----------|-------|---|---|---|---------|
| 3M | Cash flow capex | ✗ | ✓ | ✓ | Tie (B/C) |
| Netflix | Total liabilities | ✓ | ✗ | ✗ | A only |
| PepsiCo | EBITDA - capex | ✗ | ✓ | ✓ | Tie (B/C) |
| CVS Health | Asset turnover | ✗ | ✓ | ✗ | B only |
| PayPal | Working capital | ✓ | ✗ | ✓ | Tie (A/C) |
| JPMorgan | Gross margins | ✗ | ✗ | ✓ | C only |
| Verizon | Capital intensive | ✗ | ✓ | ✓ | Tie (B/C) |
| Ulta Beauty | Acquisitions | ✗ | ✓ | ✗ | B only |

**Pattern:** No clear dominance. Both B and C get different subsets correct, suggesting complementary strengths even on the same document type.

### DocVQA Performance

| Question Type | A | B | C | Winner |
|---------------|---|---|---|---------|
| Numerical fields (6 questions) | ✗ | 4/6 | 6/6 | **C** |
| Text fields (6 questions) | ✗ | 4/6 | 5/6 | **C** |
| Brand names (3 questions) | ✗ | 2/3 | 2/3 | Tie |

**Pattern:** VLM (C) wins decisively on DocVQA overall (13/15 vs 10/15).

---

## Lessons Learned

### 1. Resource Contention Matters

Running pipelines simultaneously on shared GPU can cause:
- OOM errors (7/8 FinanceBench in Pipeline C)
- Invalid conclusions (0% looks like "VLM can't handle tables")
- **Always run sequentially for accurate benchmarking**

### 2. Non-Determinism in LLM Generation

Despite temperature=0:
- 2/8 FinanceBench answers changed between Pipeline B runs
- Evaluation requires multiple runs for confidence
- Report results as ranges or averages when possible

### 3. Retrieval vs Extraction Separation

Both B and C: 100% retrieval, but 66.7% and 79.2% accuracy
- **Retrieval is "solved"** for these document types with good embeddings
- **Extraction quality** is the real differentiator
- Benchmark both separately to diagnose failures

### 4. V2 Corpus Design Success

**V1 Problems:**
- 82% too simple (single-column tables)
- Only 18% showed differentiation

**V2 Success:**
- 100% show differentiation (25% → 62-79%)
- Clear separation between naive and layout-aware
- Mix of table-heavy and form-heavy questions

---

## Files & Artifacts

### Clean Sequential Run Results
- [`results/v2_pipeline_a_fixed.csv`](results/v2_pipeline_a_fixed.csv) - Pipeline A
- [`results/v2_pipeline_b_clean.csv`](results/v2_pipeline_b_clean.csv) - Pipeline B (clean)
- [`results/v2_pipeline_c_clean.csv`](results/v2_pipeline_c_clean.csv) - Pipeline C (clean)
- [`results/v2_pipeline_b_clean_judged.csv`](results/v2_pipeline_b_clean_judged.csv) - With LLM judge
- [`results/v2_pipeline_c_clean_judged.csv`](results/v2_pipeline_c_clean_judged.csv) - With LLM judge

### Invalid Results (OOM Contaminated)
- [`results/v2_pipeline_b_fixed.csv`](results/v2_pipeline_b_fixed.csv) - First B run
- [`results/v2_pipeline_c_fixed.csv`](results/v2_pipeline_c_fixed.csv) - First C run (OOM errors)

---

## Final Conclusion

**Mission Accomplished:** V2 corpus successfully demonstrates clear differentiation between layout-aware and naive approaches.

**Corrected Key Takeaway:** 
- **VLM (Pipeline C) is the overall winner** with 79.2% accuracy
- VLM ties with LlamaParse on financial tables (62.5% each)
- VLM dominates on forms (86.7% vs 66.7%)
- Both layout-aware methods crush naive baseline (62-79% vs 25%)

**Production Recommendation:**
- **Default to VLM** unless you have specific constraints
- Use LlamaParse if markdown output is required
- Never use naive PyPDF for complex layouts

**Next Steps:**
1. ✅ V2 evaluation complete (clean sequential run)
2. ✅ Resource contention issue identified and fixed
3. ✅ Corrected results documented
4. 📝 Write comprehensive article with lessons learned
5. 🚀 Share findings with community

---

**Repository:** https://github.com/Sij-Agentic/rag-benchmark  
**Date:** 2026-08-31  
**Status:** CORRECTED FINAL RESULTS
