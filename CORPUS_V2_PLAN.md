# Corpus V2: Hard Layout-Aware Documents

## Goal

Build a corpus where **naive text extraction fundamentally fails** and layout-aware methods show clear differentiation.

## Composition

| Source | Count | Why Included | Expected Behavior |
|--------|-------|--------------|-------------------|
| **FinanceBench (hard)** | 8 | Multi-column complex tables | Naive fails (columns mixed), Markdown succeeds |
| **DocLayNet (hard)** | 1 | Scientific equation | Naive fails (symbols wrong), Markdown succeeds |
| **DocVQA** | 15 | Forms/receipts/invoices | Naive fails (field-value scrambled), Layout succeeds |
| **ChartQA** | 15 | Charts (bar/line/pie) | Naive fails (no data points), VLM succeeds |
| **Total** | **39** | All hard documents | Clear differentiation across all 3 pipelines |

## Expected Performance Matrix

| Pipeline | FinanceBench (8) | DocLayNet (1) | DocVQA (15) | ChartQA (15) | Total |
|----------|------------------|---------------|-------------|--------------|-------|
| **A (Naive)** | 0% | 0% | 0% | 0% | **0%** |
| **B (LlamaParse)** | 100% | 100% | 100% | 0% | **64%** |
| **C (VLM)** | 75% | 100% | 100% | 100% | **96%** |

### Rationale:

**Pipeline A (Naive PyPDF):**
- **FinanceBench:** Multi-column tables → columns get mixed → LLM can't extract
- **DocVQA:** Field-value pairs scrambled → LLM can't map correctly
- **ChartQA:** Text extraction gets axis labels but not data points → LLM can't answer

**Pipeline B (LlamaParse Markdown):**
- **FinanceBench:** ✓ Perfect table structure preserved
- **DocVQA:** ✓ Form structure preserved as markdown tables
- **ChartQA:** ✗ Still just text, can't read visual chart values

**Pipeline C (Gemini VLM):**
- **FinanceBench:** ✓ Can read table structure visually (mostly)
- **DocVQA:** ✓ Can see spatial field-value relationships
- **ChartQA:** ✓ Can READ values directly from visual charts (unique capability)

## Key Questions Each Document Type Tests

### FinanceBench (Multi-column Tables)
```
Example: 3M Cash Flow Statement
Question: What is FY2018 capital expenditure?

Table:
                    2018      2017      2016
Capex             (1,577)   (1,373)   (1,420)

Naive → "Capex 2018 1577 2017 1373 2016 1420" (can't match year→value)
Markdown → "| Capex | 1,577 | 1,373 | 1,420 |" (✓ preserves columns)
```

### DocVQA (Forms)
```
Example: Invoice
Question: What is the total amount?

Layout:
Invoice #: 12345        Date: 01/15/2024
Item: Widget            Qty: 10
Unit Price: $5.00       Total: $50.00

Naive → "Invoice 12345 Date 01/15/2024 Item Widget..." (field-value mixed)
Markdown → Preserves structure as table (✓)
VLM → Sees spatial layout directly (✓)
```

### ChartQA (Charts)
```
Example: Bar Chart
Question: What is the value for Category A?

[Chart showing bars: A=45, B=30, C=60]

Naive → "Category A B C" (no values extracted from visual bars)
Markdown → Same as naive (can't see visual)
VLM → Reads bar height = 45 (✓ UNIQUE CAPABILITY)
```

## Success Criteria

1. **Overall differentiation:** Pipelines show 0% → 64% → 96% progression
2. **ChartQA differentiates C from B:** Only VLM can read charts
3. **DocVQA shows all layout methods work:** B and C both succeed
4. **FinanceBench validates table preservation:** B succeeds, A fails

## Implementation Plan

### Phase 1: Collect Data (CURRENT)
- [x] Extract 9 hard questions from current corpus
- [ ] Download 15 DocVQA samples (forms/receipts/invoices)
- [ ] Download 15 ChartQA samples (bar/line/pie charts)
- [ ] Create ground_truth_v2.json

### Phase 2: Run Pipelines
- [ ] Run Pipeline A on 39 questions
- [ ] Run Pipeline B on 39 questions
- [ ] Run Pipeline C on 39 questions

### Phase 3: Evaluate
- [ ] LLM-as-judge on all 39 questions
- [ ] Verify expected performance matrix
- [ ] Analyze where each pipeline excels

### Phase 4: Document
- [ ] Update findings with v2 results
- [ ] Write article section on document selection
- [ ] Commit final results

## Timeline

- Data collection: 30 min (downloading now)
- Pipeline runs: 60 min (3 pipelines × 39 questions)
- Evaluation: 15 min (LLM judge)
- Analysis: 30 min
- **Total: 2.5 hours**

## Files Created

```
data/
  ground_truth_v2.json          # New corpus (39 questions)
  ground_truth_hard_only.json   # Filtered FinanceBench (9)
  raw_pdfs/
    financebench/               # Keep only 8 hard PDFs
    doclaynet/                  # Keep only 1 hard PDF
    docvqa/                     # 15 forms/receipts/invoices
    chartqa/                    # 15 charts

results/
  v2_pipeline_a.csv             # Pipeline A on v2 corpus
  v2_pipeline_b.csv             # Pipeline B on v2 corpus
  v2_pipeline_c.csv             # Pipeline C on v2 corpus
  v2_llm_judge.csv              # LLM judge evaluation
```
