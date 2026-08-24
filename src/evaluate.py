"""Execution harness -- run every pipeline over ground_truth.json, log metrics.

NOT YET IMPLEMENTED -- scaffold only.

Planned behaviour: for each item in `data/ground_truth.json`, run each enabled
pipeline and append one row per (pipeline, question) to `results/benchmark.csv`.

Metrics worth separating, because they fail independently:

  retrieval_hit@k   did any retrieved chunk/tile come from a page in
                    `target_pages`?  Isolates ingestion quality from the LLM.
  answer_correct    LLM-as-judge over (question, gold answer, prediction).
                    FinanceBench answers are mostly numeric, so also do an
                    exact/normalised numeric match as a judge-free cross-check.
  latency_s         wall clock, split into ingest (amortised) and query.
  cost_usd          tokens x price, plus LlamaParse page credits. The whole
                    point of a 50-100 page corpus is that this stays small.

Report results broken down by `source_dataset` and `layout_challenge`, not just
as a single average -- a vision pipeline can win decisively on tables while
losing on plain multi-column prose, and one blended number would hide that.
"""

raise NotImplementedError("The evaluation harness is not implemented yet.")
