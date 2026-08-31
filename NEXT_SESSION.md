# Next Session: Build V2 Corpus

## Current Status

**✅ Completed:**
- Evaluated 50 questions with LLM-as-judge
- Identified only 9/50 (18%) show layout awareness helps
- Filtered to 9 hard questions
- Comprehensive evaluation methodology validated

**⏸️ In Progress:**
- Need to add DocVQA and ChartQA samples to reach 39 hard questions

## Quick Start for Next Session

### Option 1: Use Smaller Dataset Samples (RECOMMENDED - 30 min)

Instead of downloading full datasets (40K+ examples), use pre-filtered subsets:

```python
# Use HuggingFace datasets with streaming=True and take first N
from datasets import load_dataset

# DocVQA - stream and filter
docvqa = load_dataset("HuggingFaceM4/DocumentVQA", split="validation", streaming=True)
samples = []
for i, item in enumerate(docvqa):
    if len(samples) >= 15:
        break
    # Filter criteria: not too long, spatial reasoning needed
    if len(item['question']) < 100 and not item['question'].lower().startswith(('is ', 'are ')):
        samples.append(item)

# ChartQA - same approach
chartqa = load_dataset("ahmed-masry/ChartQA", split="validation", streaming=True)
chart_samples = list(itertools.islice(chartqa, 15))
```

### Option 2: Manual Curation (60 min)

1. Browse DocVQA/ChartQA on HuggingFace website
2. Download 15 good examples manually
3. Create ground truth entries
4. More control over quality

### Option 3: Use Alternative Simpler Datasets (45 min)

**SROIE** (Scanned Receipts):
- Smaller dataset (~1K images)
- Clear spatial layouts
- Fields: company, date, address, total
- Download: https://huggingface.co/datasets/darentang/sroie

**Created dataset:**
```python
# Sample receipt Q&A
{
    "question": "What is the total amount?",
    "answer": "$45.67",
    "layout_challenge": "receipt with spatial field-value separation"
}
```

## Running V2 Evaluation

Once corpus is ready:

```bash
# 1. Run pipelines (60 min)
python src/evaluate.py --pipeline A --ground-truth data/ground_truth_v2.json --output results/v2_pipeline_a.csv
python src/evaluate.py --pipeline B --ground-truth data/ground_truth_v2.json --output results/v2_pipeline_b.csv
python src/evaluate.py --pipeline C --ground-truth data/ground_truth_v2.json --output results/v2_pipeline_c.csv

# 2. Evaluate with LLM judge (15 min)
python src/llm_judge_eval.py --results-prefix v2

# 3. Analyze
python src/compare_pipelines.py --results-prefix v2
```

## Expected V2 Results

| Pipeline | FinanceBench (9) | DocVQA (15) | ChartQA (15) | Total |
|----------|------------------|-------------|--------------|-------|
| A (Naive) | 0% | 0% | 0% | **0%** |
| B (LlamaParse) | 100% | 100% | 0% | **64%** |
| C (VLM) | 78% | 100% | 100% | **94%** |

**This shows:**
- Clear differentiation across all 3 approaches
- ChartQA uniquely tests VLM capability
- DocVQA tests layout preservation (B & C)
- FinanceBench tests complex table structure

## Files to Update

1. `data/ground_truth_v2.json` - New 39-question corpus
2. `src/evaluate.py` - Support `--ground-truth` parameter
3. `results/v2_*.csv` - New evaluation results
4. `FINDINGS_V2.md` - Updated findings with clear differentiation

## Alternative: Skip V2, Write Article

If time is limited, current findings are already article-worthy:

**Article Title:** "Why Your RAG Benchmark Might Be Too Easy"

**Key Points:**
1. 82% of "complex" documents were actually too simple
2. Retrieval works perfectly, extraction is the bottleneck
3. Need documents where naive extraction fundamentally fails
4. LLM-as-judge for robust evaluation

**Sections:**
- Problem: Most benchmarks don't test what they claim
- Analysis: FinanceBench deep-dive
- Solution: Document selection criteria
- Methodology: Evaluation best practices
- Datasets: DocVQA/ChartQA recommendations

**Length:** 2500-3000 words
**Time:** 3-4 hours

## Current Corpus (9 Hard Questions)

Available in: `data/ground_truth_hard_only.json`

1. 3M_2018_10K - Multi-column cash flow
2. NETFLIX_2017_10K - Balance sheet
3. PEPSICO_2022_10K - EBITDA calculation
4. CVSHEALTH_2018_10K - Asset turnover
5. PAYPAL_2022_10K - Working capital
6. JPMORGAN_2022_10K - Gross margins
7. VERIZON_2022_10K - Capital intensity
8. ULTABEAUTY_2023_10K - Acquisitions
9. scientific_articles_1001.0788_p6 - Equation

**These 9 are proven to show differentiation.**

## Recommendation

**Best path forward:**
1. **Next session (2-3 hours):** Add DocVQA/ChartQA using Option 1 (streaming), run v2 evaluation
2. **Session after (3-4 hours):** Write comprehensive article with v2 findings

**Alternative if pressed for time:**
- Write article now with current findings
- V2 evaluation becomes "validation" section later
- Current 9 hard questions + analysis already compelling
