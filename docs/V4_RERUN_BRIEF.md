# V4 Re-run Brief: Fix V3 Defects and Re-run LlamaParse vs PixelRAG

**For:** the implementation agent on the original run machine (`/home/ubuntu/rag-benchmark`, Ubuntu 24.04, Tesla T4, conda env `rag-benchmark`).
**Goal:** produce publication-grade results for a Medium article comparing **LlamaParse (Pipeline B)** with a **PixelRAG-style vision pipeline (Pipeline C)**. Pipeline A (naive PyPDF) is only a baseline.

---

## 1. Why we are re-running

An audit of the V3 results (`results/v3_*`) found defects that make several article claims unsupportable:

| # | Defect | Where | Effect |
|---|---|---|---|
| D1 | FinanceBench items added in V3 store raw `evidence` dicts in `target_pages`. They also have empty IDs (`AMCOR_2023_10K_`) and duplicate questions. | `src/build_v3_corpus.py:71,87` | 7 questions crash with `unhashable type: 'dict'` in every pipeline |
| D2 | V3 FinanceBench PDFs point to full, untrimmed 10-Ks that were never downloaded | `src/build_v3_corpus.py:86` | 4 questions fail with `PDF not found` |
| D3 | When `k` exceeds the number of indexed vectors, FAISS returns `-1`, and `chunks[-1]` duplicates the last chunk | `query()` in `pipeline_a/b/c.py` | C sends the same page image up to 5× per query, inflating tokens, cost and latency |
| D4 | The Jina model is reloaded with `from_pretrained` on every query | `query()` in `pipeline_b.py`, `pipeline_c.py` (check A too) | Query times are dominated by model loading and aren't usable |
| D5 | The PDF is re-ingested for every question, and LlamaParse cache hits aren't recorded | `src/evaluate.py:121`, `pipeline_b.py` cache | Ingestion and query times are mixed, with cold and warm LlamaParse calls combined |
| D6 | No token usage is logged. C's `context_length` measures a description string, not tokens. | all pipelines | No cost comparison is possible |
| D7 | Generation doesn't set `temperature` | `generate_content` calls in `pipeline_a/b/c.py` | Adds to run-to-run non-determinism |
| D8 | No source-image identity is stored for DocVQA and ChartQA | `src/build_v3_corpus.py` | Questions that share one image can't be grouped. In V3, 10 of 30 DocVQA questions came from a single expense table. |
| D9 | Pipeline C maps `page_num` to the pixelshot **tile index** and pairs it with PyPDF text by that index. DPI defaults to 150, but the docs say 200. | `pipeline_c.py:74-99,126,158-170` | Tiles may not correspond to pages. Retrieval text may not match the image. |
| D10 | LlamaParse is unpinned and its parse mode isn't recorded | `requirements.txt`, `pipeline_b.py:159` | Readers will ask which mode was used. Right now we can't say. |

Net effect: only **59 of 70** V3 questions actually ran. The claimed timing difference and "retrieval is solved" can't be supported.

---

## 2. What to change

Keep the experiment itself the same: same datasets, same prompts, same models (`gemini-2.5-flash` for generation and judging, `jinaai/jina-embeddings-v5-omni-small` for embeddings), and the per-document indexing protocol. **Fix the defects, add instrumentation, and change nothing else.** If you think something else needs changing, list it under "Proposed but not applied" in the report. Don't apply it.

### 2.1 Corpus (`src/build_v4_corpus.py`, new file copied from `build_v3_corpus.py`)
1. **FinanceBench:** don't rewrite the logic. Reuse `build_financebench()` and its helpers from `src/dataset.py`. That code already downloads the PDFs, resolves evidence pages by text overlap (not the unreliable `evidence_page_num` hint), and trims each PDF to evidence pages ± context. Every FinanceBench item, old and new, must go through this same path. Then:
   - Use `financebench_id` as the item ID.
   - Remove duplicate questions.
   - `target_pages` must be a `list[int]` of 1-based page numbers in the **trimmed** PDF.
   - Aim for **19** FinanceBench questions. If some documents can't be downloaded or resolved, pick replacements with the same "hard" keyword filter (`build_v3_corpus.py:64`). Log every substitution.
2. **DocVQA / ChartQA:** for each item, add `source_image_sha256` (a hash of the image bytes before PDF conversion). Also keep the dataset's own document or image identifier if it has one (for example `docId` or `imgname`). Keep the same 30 DocVQA and 20 ChartQA questions as V3. If you can't reproduce the same selection, log which questions differ.
3. **DocLayNet:** keep the one V3 question unchanged.
4. Write the corpus to `data/ground_truth_v4.json`. Validate it with the checks below and fail loudly if any check fails:
   - All IDs are unique.
   - Every `target_pages` is a non-empty `list[int]`.
   - Every `pdf_path` exists.
   - Every target page is ≤ the PDF's page count.

### 2.2 Pipelines (edit in place; the V3 CSVs are the record of old behaviour)
1. **D3:** in every `query()`, set `k_eff = min(k, index.faiss_index.ntotal)` and drop any `-1` indices. Record `k_eff` in the results.
2. **D4:** load the Jina model once per process, with a module-level cache or by passing it in. Model loading must not count towards query time.
3. **D7:** pass `temperature=0.0` to every generation call in A, B and C. Change nothing else in the prompts.
4. **D6:** after each Gemini call, record `usage_metadata.prompt_token_count`, `candidates_token_count` and `total_token_count`. Do the same for the judge.
5. **D9:**
   - Check whether pixelshot's tiles map 1:1 to PDF pages.
   - If they don't, render one image per PDF page and set `page_num` to the true page number. Pair each image with the PyPDF text for the same page. Record `images_sent` (the number of images in each query) and the image pixel dimensions.
   - Set DPI explicitly to **150** and record it. Don't switch to 200. V3 used 150, and changing it would add a new variable.
6. **D10:** pin `llama-parse` (and `llama-index-core`) in `requirements.txt` to the versions currently installed. Record the effective parse mode and every non-default option in the run manifest. **Keep the default markdown mode.** Don't turn on premium, multimodal or LLM parsing modes.

### 2.3 Evaluator (`src/evaluate.py`)
1. **D5:** split the timings and record each one separately:
   - `parse_time_s`: LlamaParse call, or pixelshot render, or PyPDF load
   - `embed_time_s`
   - `retrieval_time_s`
   - `generation_time_s`
   - `llamaparse_cache_hit` (bool)
2. Do **one cold LlamaParse pass**: move `data/cache/llamaparse/` aside first. Don't delete it. That pass gives a true parse time. Later repeat runs may use the cache.
3. **Repeats:** run the query and generation step for B and C **3 times** (`run_id` 1–3), reusing the cached ingestion. Run A once. This measures non-determinism, which matters with so few questions.
4. Keep running pipelines strictly one at a time, with `torch.cuda.empty_cache()` between pipelines (see `run_v3_eval.sh`). Never run B and C at the same time. That's what corrupted V2.

### 2.4 Judge (`src/llm_judge_v4.py`, copied from v3)
Use the same prompt and model as V3 (`gemini-2.5-flash`, temperature 0), and log token usage. Judge every row of every run.

---

## 3. Order of work

1. Run `scripts/dry_run.py`, then record `pip freeze` and `nvidia-smi` output.
2. Build and validate `ground_truth_v4.json`.
3. **Smoke test:** run A, B and C on 6 questions (2 FinanceBench, 2 DocVQA, 2 ChartQA) and check:
   - No `ERROR` rows.
   - `k_eff` ≤ page count.
   - `images_sent` equals the number of distinct pages retrieved.
   - Token columns are filled in.
   - Stop and report if any check fails.
4. Run in full: A ×1, then B cold + 2 cached repeats, then C ×3. One pipeline at a time.
5. Judge all rows.
6. Write the outputs listed in section 4.

If any row errors in the full run (for example a 503 or an OOM), retry that row up to 3 times with backoff. If it still fails, keep the error row and list it in the report. **Never drop rows silently.**

---

## 4. Outputs we need

Put everything under `results/v4/`. **Don't modify or delete any `results/v3_*` files.**

| File | Contents |
|---|---|
| `manifest.json` | git SHA, `pip freeze`, GPU, model IDs, LlamaParse version, mode and options, DPI, k, temperature, run dates |
| `pipeline_{a,b,c}_run{N}.csv` | one row per question per run (columns below) |
| `pipeline_{a,b,c}_run{N}_judged.csv` | the same rows plus `judge_correct`, `judge_reasoning` and judge token counts |
| `summary.json` | aggregates (schema below) |
| `disagreements_b_vs_c.csv` | every question where B and C differ in run 1. Columns: `item_id`, `source_dataset`, `source_image_sha256`, `question`, `gold_answer`, `answer_b`, `answer_c`, `judge_b`, `judge_c`, `judge_reasoning_b`, `judge_reasoning_c`, plus empty `human_b` and `human_c` columns for hand-checking |
| `REPORT.md` | human-readable report (section 5) |
| `logs/` | full stdout and stderr for each run |

**Required columns in each result row:** `item_id`, `run_id`, `pipeline`, `source_dataset`, `doc_id`, `source_image_sha256`, `question`, `gold_answer`, `predicted_answer`, `refused` (the answer contains "Cannot determine"), `target_pages`, `retrieved_pages`, `k_eff`, `retrieval_hit`, `pdf_page_count`, `images_sent`, `parse_time_s`, `embed_time_s`, `retrieval_time_s`, `generation_time_s`, `llamaparse_cache_hit`, `prompt_tokens`, `output_tokens`, `total_tokens`, `error`.

**`summary.json` schema:**
```json
{
  "B": {
    "runs": [{"run_id": 1, "total": 70, "valid": 70, "correct": 0, "refused": 0,
              "by_dataset": {"financebench": {"total": 19, "valid": 19, "correct": 0, "refused": 0}, "...": {}}}],
    "mean_accuracy": 0.0, "min_accuracy": 0.0, "max_accuracy": 0.0,
    "questions_correct_in_all_runs": 0, "questions_flipping_across_runs": 0,
    "timing_s": {"parse_cold_mean": 0, "parse_cached_mean": 0, "embed_mean": 0, "retrieval_mean": 0, "generation_mean": 0},
    "tokens": {"prompt_mean": 0, "output_mean": 0, "prompt_total": 0}
  },
  "B_vs_C_run1": {"both_correct": 0, "both_wrong": 0, "b_only": 0, "c_only": 0, "sign_test_p_two_sided": 0.0,
                  "by_dataset": {"chartqa": {"b_only": 0, "c_only": 0, "p": 0.0}, "...": {}}}
}
```
Include the same blocks for A and C. Also add `distinct_source_images` per dataset.

---

## 5. What `REPORT.md` must contain

Present numbers only, with no conclusions or narrative. The article agent will interpret them.

1. **Corpus table:** per dataset, the number of questions, the number of distinct source documents or images, and any substitutions compared with V3.
2. **Accuracy table:** per dataset × pipeline, give correct/valid for run 1, then the mean and min–max across runs.
3. **B vs C disagreement table:** per dataset, the `b_only` and `c_only` counts and the sign-test p-value.
4. **Grouped by source image:** accuracy for B and C where each source image counts once (its share of correct questions). This shows whether any single document dominates a result.
5. **ChartQA split:** questions that refer to a series by colour vs those that don't, with accuracy for B and C in each group. Tag a question as colour-referenced if it contains a colour word (red, blue, green, yellow, orange, purple, grey/gray, brown, light blue, colour/color). List the tagged question IDs so the tagging can be checked.
6. **Failure types for B and C:** counts of refusals, wrong values, and errors.
7. **Timing table:** mean parse time (cold), parse time (cached), embed, retrieval and generation, per pipeline.
8. **Token and cost table:** mean prompt and output tokens per query and per judge call. Estimate cost from Google's and LlamaIndex's current published prices, and cite the price used and the date you checked it.
9. **Retrieval note:** the distribution of `pdf_page_count` and `k_eff`. Be explicit that when `k_eff` equals the page count, retrieval is trivial.
10. **Deviations and anomalies:** anything that didn't go to plan, including retried or failed rows, substitutions, and whether the pixelshot tiles matched pages.
11. **Proposed but not applied:** anything you think should change but didn't.

---

## 6. Done when

- [ ] `ground_truth_v4.json` passes validation. FinanceBench has 19 unique questions, or fewer with every missing one explained.
- [ ] Every run CSV has one row per corpus question. After retries, no errors remain, or each remaining error is listed in the report.
- [ ] No row has `k_eff` greater than `pdf_page_count`, and in Pipeline C, `images_sent` equals the number of distinct retrieved pages.
- [ ] Token and split-timing columns are filled in for every non-error row.
- [ ] Per-dataset counts add up to the totals in `summary.json`. This is checked in code, and the check's output is printed in the report.
- [ ] B has one cold-parse run. B and C each have 3 runs, and all runs are judged.
- [ ] `disagreements_b_vs_c.csv` exists with empty `human_*` columns.
- [ ] `results/v3_*` are unchanged (`git diff --stat results/` shows only new files).
- [ ] All changes are committed on a branch named `v4-rerun`. Don't push or merge until the user asks.

## 7. Out of scope (don't do these)

- Swapping Pipeline C's retrieval for PixelRAG's own Qwen3-VL-Embedding retriever. With one PDF per index, retrieval is trivial, so it wouldn't change results. It only matters for a future shared-corpus experiment.
- Changing prompts, models, chunking, k, DPI or the LlamaParse mode.
- Adding datasets or questions beyond replacing broken FinanceBench items.
- Writing article prose or drawing conclusions.
