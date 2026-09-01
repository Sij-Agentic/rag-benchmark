# V3 Corpus - Design Document

**Goal:** Scale up evaluation to 80-100 questions while maintaining hard document selection criteria

**Date:** 2026-08-31  
**Status:** Design Phase

---

## V3 Objectives

1. **Validate V2 findings at scale** (25-79% differentiation)
2. **Expand document diversity** (more datasets, document types)
3. **Apply lessons learned** (sequential execution, error monitoring)
4. **Statistical confidence** (larger N for more reliable conclusions)
5. **Production-ready benchmark** (reproducible, well-documented)

---

## Corpus Design

### Target Size: 80-100 Questions

**Composition (Proposed):**
- **FinanceBench:** 20 questions (currently 8) - Multi-column financial tables
- **DocVQA:** 30 questions (currently 15) - Forms with spatial layouts
- **DocLayNet:** 10 questions (currently 1) - Scientific/technical documents
- **ChartQA:** 20 questions (currently 0) - Chart/graph reading
- **TOTAL:** 80 questions

**Why these numbers:**
- 20-30 questions per dataset for statistical significance
- Maintains balance between table-heavy and form-heavy questions
- Tests diverse layout complexities

### Hard Document Selection Criteria

From V2 lessons, only include documents where:

1. **Multi-column layouts:** ≥3 columns, nested structures
2. **Dense tables:** Cell count > 50, cross-references required
3. **Spatial dependencies:** Field-value pairs where position matters
4. **Image-only PDFs:** Test OCR vs VLM capabilities
5. **Visual elements:** Charts/graphs requiring visual reading

**Exclusion Criteria:**
- Single-column tables (too easy for naive)
- Simple key-value lists
- Plain paragraph text
- Documents with < 3 layout elements

---

## Dataset Expansion Plan

### 1. FinanceBench (8 → 20 questions)

**Current:** 8 hard questions proven to show differentiation

**Expansion Strategy:**
- Review original FinanceBench corpus (100+ questions)
- Filter for multi-column tables (≥3 columns)
- Prioritize cross-statement calculations (balance sheet + income statement)
- Include ratio calculations requiring multiple cells
- **Goal:** Add 12 more questions

**Validation:**
- Test sample questions with naive baseline
- Ensure accuracy < 40% for naive (proves it's hard)

### 2. DocVQA (15 → 30 questions)

**Current:** 15 image-only forms with spatial layouts

**Expansion Strategy:**
- Use streaming dataset to avoid downloading all 40K examples
- Filter criteria:
  - Question length < 120 chars (focused questions)
  - NOT yes/no questions (need extractive answers)
  - Forms with visible field labels (not just text blocks)
- Diverse form types: invoices, receipts, schedules, budgets, applications
- **Goal:** Add 15 more questions

**Quality Check:**
- Visual inspection: ensure spatial layout matters
- Avoid pure text documents that happen to be images

### 3. DocLayNet (1 → 10 questions)

**Current:** 1 scientific article with equation extraction

**Expansion Strategy:**
- Sample from DocLayNet categories:
  - Scientific articles (multi-column with figures)
  - Financial reports (complex layouts)
  - Government forms (structured layouts)
  - Patents (technical diagrams + text)
- Focus on questions requiring:
  - Figure/table cross-reference
  - Multi-column text extraction
  - Equation understanding
- **Goal:** Add 9 questions

**Sampling:**
```python
# Diverse document types
categories = ['scientific_articles', 'financial_reports', 
              'government_tenders', 'laws_and_regulations']
# 2-3 questions per category
```

### 4. ChartQA (0 → 20 questions) - NEW

**Challenge:** V2 had image format issues ("'bytes' object has no attribute 'convert'")

**Expansion Strategy:**
- Fix image handling in `build_v3_corpus.py`
- Handle both PIL Image and bytes objects
- Convert to PDF properly:
```python
if isinstance(image, bytes):
    image = Image.open(io.BytesIO(image))
if image.mode != 'RGB':
    image = image.convert('RGB')
```

**Question Types:**
- Bar charts (read values from visual bars)
- Line graphs (trend questions)
- Pie charts (percentage questions)
- Multi-series charts (comparison questions)

**Goal:** Add 20 chart questions to test pure visual reading

---

## Execution Strategy

### Sequential Pipeline Execution 🔴 CRITICAL

**Lesson from V2:** Simultaneous execution caused CUDA OOM errors

**V3 Plan:**
```bash
# Run pipelines one at a time
python src/evaluate.py --pipeline A --ground-truth data/ground_truth_v3.json \
    --output results/v3_pipeline_a.csv

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
sleep 5

python src/evaluate.py --pipeline B --ground-truth data/ground_truth_v3.json \
    --output results/v3_pipeline_b.csv

# Clear GPU memory  
python -c "import torch; torch.cuda.empty_cache()"
sleep 5

python src/evaluate.py --pipeline C --ground-truth data/ground_truth_v3.json \
    --output results/v3_pipeline_c.csv
```

**Safety Measures:**
1. Run each pipeline with `timeout` command (max 2 hours)
2. Monitor GPU memory after each question batch
3. Add explicit error handling for OOM
4. Save intermediate results every 10 questions

### Error Monitoring

**Proactive Monitoring:**
```python
# In evaluate.py, add after each question:
if 'CUDA out of memory' in error_msg:
    torch.cuda.empty_cache()
    log_warning(f"OOM on {item_id}, cleared cache")
    
if 'ERROR' in predicted_answer:
    log_error(f"{item_id}: {predicted_answer[:200]}")
```

**Post-Run Validation:**
```bash
# Check for errors
grep "ERROR" results/v3_*.csv | wc -l

# Check for OOM specifically  
grep "CUDA out of memory" results/v3_*.csv

# Count successful vs failed
python analyze_errors.py results/v3_*.csv
```

### Multiple Runs for Confidence

**Optional:** Run each pipeline 2-3 times to measure variance

**Reason:** V2 showed 2/8 FinanceBench answers changed in Pipeline B

**Plan:**
- If time permits, run B and C twice
- Report: mean ± std accuracy
- Identify high-variance questions (investigate why)

---

## Implementation Tasks

### Phase 1: Corpus Building (2-3 hours)

1. **Create `src/build_v3_corpus.py`**
   - Based on `build_v2_corpus.py`
   - Add ChartQA with fixed image handling
   - Expand FinanceBench/DocVQA/DocLayNet sampling
   - Generate `data/ground_truth_v3.json`

2. **Quality Control:**
   - Visual inspection of 10% sample
   - Verify ground truth answers
   - Check PDF conversions (especially DocVQA/ChartQA)

3. **Baseline Test:**
   - Run Pipeline A on 10-question sample
   - Confirm accuracy < 40% (proves corpus is hard)

### Phase 2: Pipeline Execution (4-6 hours)

**Estimated Timing (80 questions):**
- Pipeline A: ~1 hour (some will skip image-only)
- Pipeline B: ~2 hours (OCR + embeddings)
- Pipeline C: ~2 hours (VLM generation)
- **Total:** ~5 hours sequential + breaks

**Execution Script:**
```bash
#!/bin/bash
# run_v3_eval.sh

set -e  # Exit on error

echo "Starting V3 Evaluation - Sequential Execution"
date

# Pipeline A
echo "=== Pipeline A ==="
python src/evaluate.py --pipeline A \
    --ground-truth data/ground_truth_v3.json \
    --output results/v3_pipeline_a.csv
    
echo "Pipeline A complete. Clearing GPU..."
python -c "import torch; torch.cuda.empty_cache()"
sleep 10

# Pipeline B  
echo "=== Pipeline B ==="
python src/evaluate.py --pipeline B \
    --ground-truth data/ground_truth_v3.json \
    --output results/v3_pipeline_b.csv
    
echo "Pipeline B complete. Clearing GPU..."
python -c "import torch; torch.cuda.empty_cache()"
sleep 10

# Pipeline C
echo "=== Pipeline C ==="
python src/evaluate.py --pipeline C \
    --ground-truth data/ground_truth_v3.json \
    --output results/v3_pipeline_c.csv
    
echo "All pipelines complete!"
date
```

### Phase 3: LLM-as-Judge (1-2 hours)

**Estimated:** 80 questions × 3 pipelines = 240 judgments @ 1s each = 4 minutes per pipeline

```bash
# Judge all V3 results
python src/llm_judge_v3.py
```

**Output:**
- `results/v3_pipeline_a_judged.csv`
- `results/v3_pipeline_b_judged.csv`
- `results/v3_pipeline_c_judged.csv`
- `results/v3_summary.json` - Aggregate statistics

### Phase 4: Analysis & Documentation (1-2 hours)

1. **Compare with V2 results**
2. **Statistical significance tests** (if variance available)
3. **Per-dataset breakdown**
4. **Update README with V3 findings**
5. **Commit all results**

---

## Success Criteria

### Must Have ✅

1. **All 3 pipelines complete** on 80+ questions
2. **Zero OOM errors** (sequential execution works)
3. **Clear differentiation** maintained (naive < 40%, layout-aware > 60%)
4. **LLM-as-judge evaluation** for all questions
5. **Results committed** to GitHub

### Should Have 🎯

1. **ChartQA working** (20 questions successfully processed)
2. **Error rate < 5%** (< 4 failures per pipeline)
3. **Variance analysis** (if multiple runs completed)
4. **Statistical significance** (p-value for A vs B, A vs C)

### Nice to Have ⭐

1. **Multiple runs** (2-3 per pipeline for confidence intervals)
2. **Performance profiling** (identify bottlenecks)
3. **Cost analysis** (API calls, GPU hours)
4. **Ablation studies** (e.g., retrieval k=3 vs k=5)

---

## Risk Mitigation

### Risk 1: ChartQA Image Format Issues

**Mitigation:**
- Test on 5 samples before full build
- Add comprehensive error handling
- Fallback: Skip ChartQA, expand other datasets to compensate

### Risk 2: Long Execution Time (> 8 hours)

**Mitigation:**
- Start with 50-question subset if needed
- Run overnight
- Use longer timeouts

### Risk 3: OOM Despite Sequential Execution

**Mitigation:**
- Add per-question GPU clearing
- Reduce batch sizes
- Monitor memory after each question
- Emergency: Run on CPU (slow but works)

### Risk 4: Ground Truth Quality Issues

**Mitigation:**
- Sample 10% for manual review
- Cross-check with dataset source
- Document any questionable answers
- Flag for manual review post-evaluation

---

## Timeline

**Total Estimated Time:** 8-12 hours

| Phase | Task | Time | Status |
|-------|------|------|--------|
| 1 | Corpus Building | 2-3h | Pending |
| 2 | Pipeline A | 1h | Pending |
| 2 | Pipeline B | 2h | Pending |
| 2 | Pipeline C | 2h | Pending |
| 3 | LLM-as-Judge | 1h | Pending |
| 4 | Analysis | 1-2h | Pending |

**Recommended Approach:**
- Day 1: Corpus building + quality check
- Day 2: Run pipelines (morning: A, afternoon: B+C)
- Day 3: Judge + analyze + document

---

## Next Steps

1. ✅ Review and approve this design document
2. 🔄 Create `src/build_v3_corpus.py`
3. 🔄 Build V3 corpus (80 questions)
4. 🔄 Quality check (sample inspection)
5. 🔄 Run pipelines sequentially
6. 🔄 LLM-as-judge evaluation
7. 🔄 Analysis and documentation

---

**Ready to proceed?** Once approved, we'll start with Phase 1: Corpus Building.
