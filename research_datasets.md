# Datasets for Layout-Aware Document QA

## Existing Datasets to Consider

### **DocVQA** (Document Visual Question Answering)
- **Content**: Scanned documents, forms, receipts, invoices, reports
- **Questions**: Visual reasoning required (reading from spatial layouts)
- **Size**: ~50K question-answer pairs on ~12K documents
- **Format**: Images + annotations
- **Pros**: Exactly what we need - spatial layouts with Q&A
- **Cons**: Might need conversion to PDF format
- **Source**: https://rrc.cvc.uab.es/?ch=17

### **ChartQA**
- **Content**: Charts (bar, line, pie charts from papers/web)
- **Questions**: Reading values from visual charts
- **Size**: ~28K questions on ~20K charts
- **Pros**: Perfect for testing VLM vs text (text extraction from charts is terrible)
- **Cons**: Narrow domain (only charts)
- **Source**: https://github.com/vis-nlp/ChartQA

### **InfographicVQA**
- **Content**: Infographics with mixed text, charts, diagrams
- **Questions**: Extractive and abstractive QA
- **Size**: ~30K questions on 5K infographics
- **Pros**: Complex layouts, visual reasoning
- **Cons**: Might be too complex (multiple hops)
- **Source**: https://docvqa.org/datasets/infographicvqa

### **TAT-QA** (Table-And-Text QA)
- **Content**: Hybrid tables + text from financial reports
- **Questions**: Financial reasoning (similar to FinanceBench)
- **Size**: ~16K questions on 2,700 documents
- **Pros**: Financial domain, complex tables
- **Cons**: Still might not test layout enough if tables are simple
- **Source**: https://nextplusplus.github.io/TAT-QA/

### **SlideVQA**
- **Content**: Presentation slides (PowerPoint converted to images)
- **Questions**: Reading from slide layouts
- **Pros**: Complex layouts with bullets, images, text boxes
- **Cons**: Might be hard to source original PDFs

### **SROIE** (Scanned Receipts)
- **Content**: Scanned receipt images
- **Task**: Key information extraction (not Q&A)
- **Pros**: Real-world forms with spatial layouts
- **Cons**: No question-answer pairs (would need to create)

## Recommendation

**Best immediate option: DocVQA**
- Has the right document types (forms, invoices, receipts)
- Has question-answer pairs
- Tests spatial reasoning
- Well-established benchmark

**Best supplement: ChartQA**
- Charts are WHERE text extraction completely fails
- Clear differentiation expected
- Clean domain

**Action plan:**
1. Sample 50-100 questions from DocVQA (focus on forms/invoices)
2. Add 20 questions from ChartQA
3. Ensure questions require spatial understanding, not just keyword search

## Creating Our Own Dataset

**If existing datasets don't fit:**

1. **IRS Tax Forms** (W-2, 1040, 1099)
   - Public domain
   - Well-defined fields
   - Create Q&A: "What is line 1 (wages)?" → extract value

2. **Public Company Earnings Reports**
   - More complex tables than FinanceBench
   - Multi-column layouts
   - Charts and graphs

3. **Scientific Papers with Figures**
   - arXiv papers with charts/diagrams
   - Questions about figure values
   - "What is the accuracy at epoch 10 in Figure 2?"

4. **Invoice Datasets**
   - Use public invoice datasets
   - Create extraction questions
   - "What is the total amount?" "What is the vendor name?"
