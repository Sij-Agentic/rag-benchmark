# V2 Corpus Evaluation - In Progress

**Date:** 2026-08-31  
**Status:** Running all 3 pipelines on V2 corpus

---

## Issues Fixed

### 1. DocVQA PDF Format Issue
**Problem:** DocVQA PDFs are image-only (no text layer)  
**Impact:** PyPDFLoader returns empty documents  
**Solution:** 
- Added error handling in pipeline_a.py to skip image-only PDFs
- Added error handling in pipeline_b.py (but LlamaParse has built-in OCR)
- Pipeline C uses VLM and handles images natively

**Result:**
- Pipeline A: Skips DocVQA gracefully (9/24 questions: FinanceBench + DocLayNet only)
- Pipeline B: Processes DocVQA via OCR (expected 24/24 questions)
- Pipeline C: Processes DocVQA via VLM (expected 24/24 questions)

### 2. CLI --ground-truth Flag
**Problem:** evaluate.py didn't accept custom ground truth path  
**Solution:** Added `--ground-truth` argument to argparse  
**Result:** Can now specify `data/ground_truth_v2.json`

---

## V2 Corpus Composition

**Total:** 24 questions
- **FinanceBench (hard):** 8 questions - Multi-column tables where naive extraction fails
- **DocLayNet:** 1 question - Scientific article with complex layout
- **DocVQA:** 15 questions - Forms/invoices with spatial field-value relationships

---

## Pipeline Runs

### Pipeline A ✓ Complete
**Command:**
```bash
python src/evaluate.py --pipeline A --ground-truth data/ground_truth_v2.json \
    --output results/v2_pipeline_a_fixed.csv
```

**Results:**
- **Total:** 24 questions
- **Successful:** 9 (37.5%)
- **By dataset:**
  - FinanceBench: 8/8 (100%)
  - DocLayNet: 1/1 (100%)
  - DocVQA: 0/15 (0% - all image-only, skipped)
- **Retrieval Hit@5:** 9/9 (100%)
- **Timing:** Avg ingest 3.9s, Avg query 3.9s

**Key Finding:** Pipeline A correctly skips all image-only PDFs and processes text PDFs perfectly.

### Pipeline B ⏳ Running
**Command:**
```bash
python src/evaluate.py --pipeline B --ground-truth data/ground_truth_v2.json \
    --output results/v2_pipeline_b_fixed.csv
```

**Expected:**
- Should process all 24/24 questions
- LlamaParse has built-in OCR for image-only PDFs
- From previous run: successfully extracted text from DocVQA

### Pipeline C ⏳ Running
**Command:**
```bash
python src/evaluate.py --pipeline C --ground-truth data/ground_truth_v2.json \
    --output results/v2_pipeline_c_fixed.csv
```

**Expected:**
- Should process all 24/24 questions
- VLM reads images directly
- From previous run: successfully answered DocVQA questions

---

## Next Steps

1. **Wait for Pipeline B & C** (running now, ~10-15 min total)
2. **Run LLM-as-Judge:**
   ```bash
   python src/llm_judge_v2.py
   ```
3. **Analyze results** - Compare accuracy across pipelines
4. **Expected differentiation:**
   - Pipeline A: Low accuracy on FinanceBench (naive extraction fails)
   - Pipeline B: Higher accuracy (markdown preserves table structure)
   - Pipeline C: Competitive accuracy (VLM reads visual layout)
5. **Commit final results** and update documentation
6. **Begin article writing** - comprehensive comparison with validated results

---

## Files Created

### Source Code
- [src/pipeline_a.py](src/pipeline_a.py) - Added image-only PDF handling
- [src/pipeline_b.py](src/pipeline_b.py) - Added image-only PDF handling
- [src/evaluate.py](src/evaluate.py) - Added --ground-truth CLI flag
- [src/llm_judge_v2.py](src/llm_judge_v2.py) - V2-specific LLM judge
- [analyze_v2_results.py](analyze_v2_results.py) - Quick stats analysis

### Data
- [data/ground_truth_v2.json](data/ground_truth_v2.json) - 24 hard questions
- [data/raw_pdfs/docvqa/*.pdf](data/raw_pdfs/docvqa/) - 15 image-only PDFs

### Results
- [results/v2_pipeline_a_fixed.csv](results/v2_pipeline_a_fixed.csv) ✓ Complete
- results/v2_pipeline_b_fixed.csv ⏳ In progress
- results/v2_pipeline_c_fixed.csv ⏳ In progress

---

## Technical Insights

### Image-Only PDFs
DocVQA images were converted to PDF format but lack a text layer. This is actually **perfect** for testing:
- **Pipeline A failure mode:** Cannot extract text → baseline benchmark
- **Pipeline B capability:** LlamaParse OCR → tests OCR quality
- **Pipeline C capability:** VLM reading → tests visual understanding

### Layout Complexity Spectrum
```
Simple ───────────────────────────────────────────> Complex
│                    │                    │                    │
Single-column    Multi-column        Forms with           Charts/
tables             tables            spatial layout        graphs
(Pipeline A       (Requires         (Requires           (Requires
works fine)       structure)        visual/spatial)     vision)
                  
                  [Pipeline B wins]                [Pipeline C wins]
```

### Evaluation Progress
- ✓ V1: 50 questions → discovered 82% too simple
- ✓ V2: 24 questions → only hard documents included
- ⏳ V2 evaluation running → expect clear differentiation
- ⏳ LLM-as-judge → final accuracy numbers
- 📝 Article writing → comprehensive analysis

---

**Last Updated:** 2026-08-31 19:30 UTC
