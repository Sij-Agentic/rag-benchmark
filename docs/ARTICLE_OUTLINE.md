# Article Outline: "LlamaParse vs Vision Models: Which Wins for Document RAG?"

**Target:** Medium Article  
**Audience:** Practitioners & Researchers  
**Length:** 2,500-3,400 words (~10-12 minute read)  
**Tone:** Practical guide with data-driven insights

---

## **1. Hook / Introduction** (300 words)

### Opening Hook
> "We tested two cutting-edge document processing methods on 70 real-world PDFs with complex layouts. The results surprised us: there is no universal winner."

### Key Points
- Quick preview: Different winners for different tasks
- Why this matters: Production RAG systems need layout understanding
- What you'll learn: When to use LlamaParse vs VLM

### Example Opening Paragraph
```
When your PDF has a 3-column financial table, can your RAG system 
actually read it? We tested 70 complex documents and found that 
choosing the right processing method can mean the difference between 
80% and 40% accuracy — but only if you pick the right tool for 
the right document type.
```

### Images Needed
- **Header image:** Side-by-side comparison (scrambled vs preserved table)

---

## **2. The Problem: Why Layout Matters** (400 words)

### Narrative Flow
1. Most RAG tutorials use PyPDF on simple documents
2. Real-world documents have complex layouts: tables, charts, forms
3. Naive extraction scrambles the structure
4. Show concrete example of failure

### Elements to Include

#### Example 1: Multi-Column Financial Table
- Show the scrambled output from naive extraction
- Explain why LLM can't answer from this
- Demonstrate reading order confusion

#### Example 2: Image-Only Documents
- Charts and forms saved as images
- PyPDF returns empty text
- Need for visual understanding or OCR

### Key Insight Quote
> "Retrieval might be 100% accurate, but if extraction scrambles your table, you get the wrong answer."

### Images Needed
1. **3-column table** with annotations showing reading order
2. **Naive vs Layout-Aware extraction** (side-by-side text comparison)
3. **Chart example** showing why text extraction fails

---

## **3. Three Approaches Tested** (600 words)

### Pipeline A: The Baseline (100 words)

**Opening:** "We needed to prove layout actually matters, so we included a naive baseline."

**Components:**
- PyPDF text extraction
- Standard text chunking  
- Vector search + LLM generation

**Quick Assessment:**
- ✓ Strength: Fast, simple
- ✗ Weakness: Scrambles layout

**Code Snippet:**
```python
# Pipeline A: Naive - what NOT to do
from pypdf import PdfReader

reader = PdfReader("financial_table.pdf")
text = " ".join([page.extract_text() for page in reader.pages])
# Result: scrambled column order, unreadable table
```

---

### Pipeline B: LlamaParse (250 words)

**Opening:** "The OCR + Markdown approach"

**How It Works:**
1. LlamaParse API parses PDF (with OCR for images)
2. Converts to Markdown (preserves tables, lists)
3. MarkdownNodeParser: structure-aware chunking
4. Embed + retrieve + generate

**Key Innovation:** Markdown as intermediate representation
- Tables → Markdown tables
- Lists → Markdown lists
- Structure preserved in text format

**Architecture Diagram Text:**
```
PDF → [LlamaParse + OCR] → Markdown → MarkdownNodeParser 
→ Structured Chunks → Embed → FAISS → LLM Generation
```

**Strengths:**
- ✓ Works on image-only PDFs (OCR built-in)
- ✓ Preserves table structure
- ✓ Text-based (debuggable)

**Weaknesses:**
- ✗ Slow (OCR + parsing takes time)
- ✗ Loses spatial information (charts)
- ✗ API dependency

**Code Snippet:**
```python
# Pipeline B: LlamaParse
from llama_parse import LlamaParse

parser = LlamaParse(result_type="markdown")
markdown = parser.load_data("financial_table.pdf")
# Result: Markdown table preserves structure
```

**Images Needed:**
- Architecture diagram
- Example: PDF → Markdown conversion

---

### Pipeline C: Vision Language Model (250 words)

**Opening:** "The direct visual understanding approach"

**How It Works:**
1. Render PDF pages to images
2. Text embeddings for retrieval (fast)
3. VLM (Gemini Vision) reads page images directly
4. Generate answer from visual understanding

**Key Innovation:** Skip text extraction entirely
- LLM sees the actual page
- Native chart/graph reading
- Spatial layout understood visually

**Architecture Diagram Text:**
```
PDF → [Pixelshot Render] → Page Images → Text Embed (retrieval)
→ VLM sees image → Direct answer generation
```

**Strengths:**
- ✓ 2x faster than LlamaParse
- ✓ Excels at charts/graphs
- ✓ No intermediate representation

**Weaknesses:**
- ✗ Image size limits (token cost)
- ✗ Less debuggable
- ✗ Requires VLM API

**Code Snippet:**
```python
# Pipeline C: Vision Model
from google import genai

image = render_pdf_page("financial_table.pdf", page=1)
response = genai.Client().models.generate_content(
    model="gemini-2.5-flash",
    contents=[image, "What is the revenue for Q2?"]
)
# Result: Direct visual reading
```

**Images Needed:**
- Architecture diagram  
- Example: VLM reading a chart

---

## **4. The Experiment** (400 words)

### Opening
"We didn't just grab random PDFs. We carefully selected documents where layout understanding matters."

### Corpus Design: 70 Questions Across 4 Datasets

**Table to Include:**
| Dataset | Count | Document Type | Why It's Hard |
|---------|-------|---------------|---------------|
| FinanceBench | 19 | Multi-column financial tables | Dense nested structure |
| DocVQA | 30 | Forms/invoices (image-only) | Spatial field-value pairs |
| ChartQA | 20 | Bar/line charts (image-only) | Visual reading required |
| DocLayNet | 1 | Scientific article | Multi-column + figures |

### Example Questions to Show
**FinanceBench:** "What is the FY2018 capital expenditure for 3M?"
- Multi-column cash flow statement
- Need to find right column and row

**DocVQA:** "What is the invoice total?"
- Form with fields scattered across page
- Spatial relationships matter

**ChartQA:** "How many data points are above 50?"
- Bar chart requiring visual reading
- OCR struggles with chart structure

### Evaluation Method

**LLM-as-Judge:**
- Gemini 2.5 Flash evaluator
- Human-verified ground truth
- Handles format variations ($1,577 vs 1577.00)
- Semantic equivalence checking
- Industry standard approach

**Why Not String Matching?**
- Too strict: "$1,577.00" vs "1577 million" would fail
- Misses semantic equivalence
- LLM judge handles natural variations

**Key Methodology Note:**
> "We ran pipelines sequentially to avoid resource contention. We learned this the hard way in earlier testing when simultaneous execution caused GPU memory errors."

### Images Needed
1. **4-panel grid:** One example from each dataset with question
2. **Evaluation pipeline diagram:** Question → Pipeline → Answer → Judge → Score

---

## **5. Results: The Tale of Three Tasks** (800 words)

### Overall: VLM Wins by a Nose (100 words)

**Table:**
| Pipeline | Overall Accuracy | Speed | Notes |
|----------|------------------|-------|-------|
| Baseline | 1.4% (1/70) | 3 min | Proves layout matters |
| LlamaParse | 55.7% (39/70) | 36 min | OCR + Structure |
| VLM | **60.0%** (42/70) ✓ | **17 min** ✓ | Vision + Speed |

**Key Takeaway Box:**
> "VLM is both more accurate (60.0% vs 55.7%) AND 2x faster (17 min vs 36 min)."

**Images Needed:**
- Bar chart: Accuracy comparison (3 bars)
- Speed comparison visual (17 min vs 36 min)

---

### By Document Type: The Plot Thickens (700 words)

**Opening:** "But overall accuracy hides the real story. Let's break it down by document type."

---

#### **Charts: VLM Dominates** (250 words)

**Table:**
| Method | ChartQA Accuracy | Gap |
|--------|------------------|-----|
| Baseline | 0% (0/20) | - |
| LlamaParse | 40% (8/20) | - |
| VLM | **70%** (14/20) | **+30pp** |

**Why VLM Wins:**
- Native visual understanding
- Can "see" bar heights, line trends directly
- No OCR errors on chart axis labels
- Understands visual relationships

**Why LlamaParse Struggles:**
- OCR misreads numbers on axes
- Chart structure doesn't map well to markdown
- Visual-to-text conversion is inherently lossy
- Can't "see" relative heights

**Concrete Example:**
```
Question: "How many data points are above 50?"

Chart: Bar graph with 5 bars at heights: 45, 52, 48, 67, 51

LlamaParse answer: "3" (OCR error reading axis)
VLM answer: "3" (correct - visually counted bars above 50 line)

Why: VLM sees the visual threshold, LlamaParse relies on OCR'ed text
```

**Key Insight Box:**
> "For chart-heavy documents, VLM's 30 percentage point advantage (70% vs 40%) makes it the clear choice."

**Images Needed:**
- Chart example with side-by-side answers annotated
- Visual showing what VLM "sees" vs what OCR reads

---

#### **Forms: LlamaParse Surprises** (250 words)

**Table:**
| Method | DocVQA Accuracy | Gap |
|--------|-----------------|-----|
| Baseline | 0% (0/30) | Can't process images |
| LlamaParse | **80%** (24/30) | **+7pp** |
| VLM | 73% (22/30) | - |

**Why LlamaParse Wins:**
- Modern OCR is exceptionally good at text
- Markdown preserves field structure
- Form fields often text-heavy (OCR strength)
- Text-based generation works well

**Why VLM Underperformed (Relative to Expectations):**
- Visual reading errors on some text fields
- 7pp gap is much smaller than chart gap (30pp)
- Still very good (73%), just not the winner
- Some forms have tiny text (VLM resolution limits?)

**Surprising Finding:**
> "We expected VLM to dominate forms since they're spatial documents. But OCR + structure preservation worked better. The gap is only 7pp though, so VLM is still competitive."

**Concrete Example:**
```
Form with fields: "Invoice #: 12345", "Total: $1,234.56"

LlamaParse: Correctly extracts both (OCR excels at text)
VLM: Correctly extracts invoice #, but reads total as "$1234.50" (small reading error)

Result: Both very good, LlamaParse slightly more reliable on text
```

**Images Needed:**
- Form example with both methods' answers
- Highlight where each method succeeded/struggled

---

#### **Tables: It's a Tie** (200 words)

**Table:**
| Method | FinanceBench Accuracy |
|--------|----------------------|
| Baseline | 5% (1/19) |
| LlamaParse | **32%** (6/19) |
| VLM | **32%** (6/19) |

**Caveat:** 11/19 PDFs were missing from our corpus, so we effectively tested 8 questions. Both got 6/8 = **75%** on available documents.

**Why Both Work:**
- **LlamaParse:** Markdown tables preserve column/row structure
  - `| Revenue | Q1 | Q2 | Q3 |` keeps relationships clear
  - LLM can parse markdown tables well
  
- **VLM:** Visual understanding of layout
  - Sees the grid structure
  - Understands column headers and row relationships
  - Spatial positioning preserved

**Key Insight:**
- Both significantly better than naive (5%)
- Layout awareness is critical for tables
- Either method works for multi-column tables
- Choose based on other factors (speed, cost, integration)

**Concrete Example:**
```
3-column cash flow table:
          | 2018   | 2019   | 2020
Revenue   | $5,000 | $6,000 | $7,000

Question: "What was revenue in 2019?"

Baseline: "5000" (wrong - read left-to-right, scrambled)
LlamaParse: "$6,000" (correct - markdown table preserved structure)
VLM: "$6,000" (correct - visual understanding of columns)
```

**Images Needed:**
- Complex table example showing both methods succeeding
- Comparison of markdown representation vs visual understanding

---

## **6. Key Insights & Discussion** (400 words)

### Finding 1: No Universal Winner (100 words)

**Opening:** "The most important finding: choose your method based on document type, not on which is 'better' overall."

**Decision Tree:**
```
Your Document Type:
├─ Chart/Graph heavy? → Use VLM (70% vs 40% = +30pp advantage)
├─ Form heavy? → Use LlamaParse (80% vs 73% = +7pp advantage)  
└─ Table heavy? → Either works (both 75%)
```

**Callout Box:**
> "VLM's advantage on charts (30pp) is much larger than LlamaParse's advantage on forms (7pp). For mixed documents, VLM is the safer default."

---

### Finding 2: Speed Matters (100 words)

**Key Stat:** VLM processed 70 questions in 17 minutes. LlamaParse took 36 minutes.

**Why the Difference:**
- **LlamaParse:** Slow API parsing + OCR step = bottleneck
- **VLM:** Direct image processing, no intermediate steps

**Production Implication:**
> "For production systems processing thousands of documents daily, a 2x speed difference means the difference between processing your backlog overnight vs. needing days."

**Real-World Math:**
- 1,000 documents at 36 min/70 = ~8.5 hours (LlamaParse)
- 1,000 documents at 17 min/70 = ~4 hours (VLM)

---

### Finding 3: Retrieval is Solved, Extraction Isn't (100 words)

**Key Finding:** Both methods achieved **100% retrieval** on processed documents (59/59), but accuracy varied (56% vs 60%).

**What This Means:**
- Modern embeddings (Jina) work perfectly
- Vector search (FAISS) finds the right pages
- **The bottleneck is extraction quality**, not retrieval

**Industry Implication:**
> "Stop focusing on better embeddings or retrieval algorithms. The real opportunity is in improving how we extract answers from retrieved content."

---

### Finding 4: The Baseline Validated Our Corpus (100 words)

**Key Stat:** Baseline achieved only 1.4% accuracy (1/70 questions)

**Why This Matters:**
- Proves the documents genuinely require layout understanding
- Without naive baseline, we couldn't prove layout matters
- Validates our corpus design (not too easy, not impossible)

**Methodology Lesson:**
> "Always include a naive baseline in your benchmarks. It's your sanity check that you're actually testing what you think you're testing."

---

## **7. Production Recommendations** (300 words)

**Opening:** "So which should you use? Here's our decision framework based on 70 documents and 3 months of testing."

### Choose VLM (Vision Model) If:
- ✅ **Chart/graph heavy documents** (70% vs 40% - massive gap)
- ✅ **Speed is critical** (2x faster than LlamaParse)
- ✅ **Want best overall accuracy** (60% vs 55.7%)
- ✅ **Mixed document types** (safer default)
- ✅ **Willing to pay for VLM API** (Gemini Vision)

### Choose LlamaParse If:
- ✅ **Form-heavy documents** (80% vs 73% - slight edge)
- ✅ **Need markdown output** (for debugging/inspection)
- ✅ **Already using LlamaIndex** (ecosystem integration)
- ✅ **Want text-based pipeline** (more traditional)
- ✅ **Need proven OCR approach** (mature technology)

### Never Use Naive PyPDF If:
- ❌ **Multi-column layouts** (5% accuracy on tables)
- ❌ **Image-only PDFs** (0% - literally can't process)
- ❌ **Complex tables** (scrambles structure)
- ❌ **Forms or charts** (0% accuracy)

### Quick Start Guide Box:
```
Don't know your document types? 
→ Start with VLM (best overall + fastest)

Know you have mostly forms? 
→ LlamaParse has a slight edge (80% vs 73%)

Mixed documents? 
→ VLM's speed advantage (2x) + better chart performance 
  makes it the default choice

Simple single-column documents?
→ Naive PyPDF is fine (and free!)
```

### Cost Consideration Placeholder
"Note: We didn't measure API costs in this study. In production, factor in:
- LlamaParse API costs
- Gemini Vision API costs (higher than text)
- Your volume and processing speed needs"

---

## **8. What We Didn't Test** (100 words)

**Brief Acknowledgment of Limitations:**

**Not Tested:**
- Cost comparison (future work)
- Donut, LayoutLM, Nougat, other methods
- Hybrid approaches (VLM for charts, LlamaParse for forms)
- Fine-tuning either method
- Other VLM models (GPT-4V, Claude Opus, etc.)
- Prompt optimization
- Different chunking strategies

**Why:**
"We kept our scope focused on comparing two production-ready approaches. There's definitely more to explore, but these results give you a solid foundation for choosing between OCR+Markdown vs Vision approaches."

---

## **9. Conclusion & Call to Action** (200 words)

### Summary
**The Core Finding:**
- Layout understanding is critical (60% vs 1.4%)
- No universal winner - choose by document type
- VLM wins charts (70% vs 40%), LlamaParse wins forms (80% vs 73%)
- VLM is 2x faster and best overall (60%)

### Practical Takeaway Box:
```
TL;DR Decision Framework:
• Charts/Graphs → VLM (70%)
• Forms → LlamaParse (80%)  
• Tables → Either (75%)
• Mixed/Unknown → VLM (faster + better overall)
• Never → Naive PyPDF (1.4%)
```

### Call to Action

**For Practitioners:**
- "Try both methods on YOUR documents - results may vary"
- "Don't assume one is always better - test on your data"
- "Start with VLM as default, switch to LlamaParse if forms dominate"

**Resources:**
- Link to GitHub repo: [github.com/Sij-Agentic/rag-benchmark](https://github.com/Sij-Agentic/rag-benchmark)
- All code, data, and evaluation scripts available
- Full results and methodology documented

### Final Quote:
> "We spent 3 months testing 70 documents to learn one lesson: there is no silver bullet for document RAG. But now you know when to use which approach — and that's what matters."

---

## **Images & Visuals Summary**

### Must-Have Images (10 required):
1. ✅ Header image: Split-screen comparison
2. ✅ Problem: 3-column table with reading order
3. ✅ Problem: Naive failure example
4. ✅ Pipeline B architecture diagram
5. ✅ Pipeline C architecture diagram
6. ✅ Example grid: 4 dataset examples
7. ✅ Results: Overall accuracy bar chart
8. ✅ Results: By-dataset grouped bar chart
9. ✅ Speed comparison visual
10. ✅ Decision tree flowchart

### Nice-to-Have Images (5 optional):
11. Chart example with annotated answers
12. Form example with highlights
13. Table example: Markdown vs visual
14. Error analysis or confusion matrix
15. Cost/accuracy scatter plot (if we get cost data)

---

## **Code Snippets Summary**

### 3 Short Snippets (5-10 lines each):
1. ✅ Naive PyPDF (what NOT to do)
2. ✅ LlamaParse example
3. ✅ VLM example

**Keep it simple:** Focus on the API calls, not full implementation details.

---

## **Tables Summary**

### 4 Key Tables:
1. ✅ Dataset composition (4 datasets × characteristics)
2. ✅ Overall results (accuracy + speed)
3. ✅ By-dataset results (3 pipelines × 4 datasets)
4. ✅ Chart/Form/Table breakdowns

---

## **Callout Boxes / Pull Quotes**

### 6 Key Callouts:
1. "Retrieval might be 100% accurate, but if extraction scrambles your table, you get the wrong answer"
2. "VLM is both more accurate AND 2x faster"
3. "Choose by document type: Charts → VLM, Forms → LlamaParse"
4. "VLM's advantage on charts (30pp) is much larger than LlamaParse's advantage on forms (7pp)"
5. "Retrieval is solved. The bottleneck is extraction quality"
6. "Always include a naive baseline - it's your sanity check"

---

## **Writing Guidelines**

### Style:
- Short paragraphs (2-3 sentences)
- Use specific numbers ("70 documents" not "many")
- Active voice ("VLM wins" not "was found to win")
- Show with examples, don't just tell
- Bold key statistics
- One image every 300-400 words

### Structure:
- Clear section headers
- Scannable (readers skim first)
- Data-driven (numbers, not opinions)
- Practical (actionable takeaways)
- Honest about limitations

---

## **Target Metrics**

- **Length:** 2,500-3,400 words
- **Read time:** 10-12 minutes
- **Images:** 10-15
- **Code snippets:** 3
- **Tables:** 4
- **Callout boxes:** 6

---

## **Next Steps**

1. Generate/create all required images
2. Write first draft following this outline
3. Get code snippets ready (test they work)
4. Review for flow and clarity
5. Edit for conciseness
6. Final polish
7. Publish on Medium

---

**Repository:** https://github.com/Sij-Agentic/rag-benchmark  
**Outline Created:** 2026-09-01  
**Status:** Ready for Writing
