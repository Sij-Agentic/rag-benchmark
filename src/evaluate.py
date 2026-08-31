"""Evaluation harness — run pipelines over ground_truth.json, compute metrics.

Usage:
    python src/evaluate.py --pipeline A --output results/pipeline_a.csv
    python src/evaluate.py --pipeline A --financebench-only  # Quick test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
GROUND_TRUTH = DATA_DIR / "ground_truth.json"


@dataclass
class EvalResult:
    """One evaluation result (pipeline × question)."""

    # Identifiers
    pipeline: str
    item_id: str
    source_dataset: str
    doc_id: str

    # Question & answer
    question: str
    gold_answer: str | None
    predicted_answer: str

    # Retrieval metrics
    retrieved_pages: list[int]
    target_pages: list[int]
    retrieval_hit_at_k: bool
    retrieval_k: int

    # Timing
    ingest_time_s: float
    query_time_s: float

    # Context & metadata
    layout_challenge: str
    qa_source: str
    context_length: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "pipeline": self.pipeline,
            "item_id": self.item_id,
            "source_dataset": self.source_dataset,
            "doc_id": self.doc_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "predicted_answer": self.predicted_answer,
            "retrieved_pages": ",".join(map(str, self.retrieved_pages)),
            "target_pages": ",".join(map(str, self.target_pages)),
            "retrieval_hit_at_k": self.retrieval_hit_at_k,
            "retrieval_k": self.retrieval_k,
            "ingest_time_s": round(self.ingest_time_s, 3),
            "query_time_s": round(self.query_time_s, 3),
            "layout_challenge": self.layout_challenge,
            "qa_source": self.qa_source,
            "context_length": self.context_length,
        }
        d.update(self.extra)
        return d


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------


def load_pipeline(name: str):
    """Import the specified pipeline module."""
    import importlib.util
    import sys

    pipeline_file = PROJECT_ROOT / "src" / f"pipeline_{name.lower()}.py"
    if not pipeline_file.exists():
        raise ValueError(f"Pipeline file not found: {pipeline_file}")

    # Load module from file path
    spec = importlib.util.spec_from_file_location(f"pipeline_{name.lower()}", pipeline_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load pipeline {name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_item(pipeline_module, item: dict, k: int = 5, config: dict | None = None) -> EvalResult:
    """Run one (pipeline, question) evaluation."""
    pdf_path = PROJECT_ROOT / item["pdf_path"]
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    question = item["question"]
    if not question:
        raise ValueError(f"Item {item['id']} has no question (qa_source={item['qa_source']})")

    # Ingest
    t0 = time.time()
    index = pipeline_module.ingest([pdf_path], config=config)
    ingest_time = time.time() - t0

    # Query
    t0 = time.time()
    answer = pipeline_module.query(index, question, k=k)
    query_time = time.time() - t0

    # Check retrieval hit
    target_set = set(item["target_pages"])
    retrieved_set = set(answer.retrieved_pages)
    hit = bool(target_set & retrieved_set)

    return EvalResult(
        pipeline=pipeline_module.__name__.split("_")[-1].upper(),
        item_id=item["id"],
        source_dataset=item["source_dataset"],
        doc_id=item["doc_id"],
        question=question,
        gold_answer=item["answer"],
        predicted_answer=answer.text,
        retrieved_pages=answer.retrieved_pages,
        target_pages=item["target_pages"],
        retrieval_hit_at_k=hit,
        retrieval_k=k,
        ingest_time_s=ingest_time,
        query_time_s=query_time,
        layout_challenge=item["layout_challenge"],
        qa_source=item["qa_source"],
        context_length=len(answer.context),
        extra={
            "pdf_path": item["pdf_path"],
            "page_match_confidence": item.get("page_match_confidence"),
        },
    )


def run_evaluation(
    pipeline_name: str,
    output_path: Path,
    financebench_only: bool = False,
    k: int = 5,
    ground_truth_path: Path = GROUND_TRUTH,
) -> pd.DataFrame:
    """Run full evaluation and save results.

    Args:
        pipeline_name: "A", "B", or "C"
        output_path: Where to save CSV results
        financebench_only: If True, skip DocLayNet items
        k: Number of chunks to retrieve per query
        ground_truth_path: Path to ground truth JSON file

    Returns:
        DataFrame with results
    """
    # Load ground truth
    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"{ground_truth_path} not found. Run `python src/dataset.py` first."
        )

    with open(ground_truth_path) as f:
        gt = json.load(f)

    items = gt["items"]
    if financebench_only:
        items = [i for i in items if i["source_dataset"] == "financebench"]
        print(f"[FinanceBench only] {len(items)} questions")
    else:
        print(f"[Full corpus] {len(items)} questions")

    # Filter out items without questions
    items = [i for i in items if i.get("question")]
    if len(items) < len(gt["items"]):
        skipped = len(gt["items"]) - len(items)
        print(f"  (skipped {skipped} items with no question)")

    # Load pipeline
    print(f"[Pipeline {pipeline_name.upper()}]")
    pipeline = load_pipeline(pipeline_name)

    # Pipeline-specific config
    config = {}
    if pipeline_name.upper() == "B":
        # Pipeline B: Use GPU for embeddings (fast) but CPU for FAISS (avoid OOM)
        config["use_gpu_embeddings"] = True
        config["use_gpu_faiss"] = False
        print("  [Config] Pipeline B: GPU embeddings, CPU FAISS (avoiding FAISS OOM)")
    elif pipeline_name.upper() == "C":
        # Pipeline C: Vision-based, may need different config
        pass

    # Run evaluation
    results: list[EvalResult] = []
    for i, item in enumerate(items, 1):
        print(f"\n[{i}/{len(items)}] {item['id']}")
        print(f"  doc: {item['doc_id']}")
        print(f"  question: {item['question'][:80]}...")
        try:
            result = evaluate_item(pipeline, item, k=k, config=config)
            print(f"  retrieval: {'HIT' if result.retrieval_hit_at_k else 'MISS'} "
                  f"(retrieved pages: {result.retrieved_pages})")
            print(f"  answer: {result.predicted_answer[:120]}...")
            print(f"  timing: ingest={result.ingest_time_s:.2f}s, query={result.query_time_s:.2f}s")
            results.append(result)
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            # Create a failure record
            results.append(
                EvalResult(
                    pipeline=pipeline_name.upper(),
                    item_id=item["id"],
                    source_dataset=item["source_dataset"],
                    doc_id=item["doc_id"],
                    question=item["question"],
                    gold_answer=item["answer"],
                    predicted_answer=f"ERROR: {exc}",
                    retrieved_pages=[],
                    target_pages=item["target_pages"],
                    retrieval_hit_at_k=False,
                    retrieval_k=k,
                    ingest_time_s=0.0,
                    query_time_s=0.0,
                    layout_challenge=item["layout_challenge"],
                    qa_source=item["qa_source"],
                    context_length=0,
                    extra={"error": str(exc)},
                )
            )

    # Save results
    df = pd.DataFrame([r.to_dict() for r in results])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to {output_path}")
    print(f"{'='*70}")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Pipeline: {pipeline_name.upper()}")
    print(f"Questions evaluated: {len(results)}")
    print(f"\nRetrieval Hit@{k}: {df['retrieval_hit_at_k'].sum()}/{len(df)} "
          f"({100*df['retrieval_hit_at_k'].mean():.1f}%)")
    print(f"\nTiming:")
    print(f"  Avg ingest: {df['ingest_time_s'].mean():.2f}s")
    print(f"  Avg query:  {df['query_time_s'].mean():.2f}s")
    print(f"  Total:      {df['ingest_time_s'].sum() + df['query_time_s'].sum():.1f}s")

    if "source_dataset" in df.columns and len(df["source_dataset"].unique()) > 1:
        print(f"\nBy dataset:")
        for ds in sorted(df["source_dataset"].unique()):
            subset = df[df["source_dataset"] == ds]
            print(
                f"  {ds}: {subset['retrieval_hit_at_k'].sum()}/{len(subset)} "
                f"({100*subset['retrieval_hit_at_k'].mean():.1f}%)"
            )

    print(f"{'='*70}\n")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--pipeline",
        required=True,
        choices=["A", "B", "C", "a", "b", "c"],
        help="Pipeline to evaluate (A, B, or C)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: results/pipeline_X.csv)",
    )
    ap.add_argument(
        "--financebench-only",
        action="store_true",
        help="Evaluate only FinanceBench questions (skip DocLayNet)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    ap.add_argument(
        "--ground-truth",
        type=Path,
        default=GROUND_TRUTH,
        help=f"Ground truth JSON file (default: {GROUND_TRUTH})",
    )
    args = ap.parse_args(argv)

    # Default output path
    if not args.output:
        args.output = (
            PROJECT_ROOT / "results" / f"pipeline_{args.pipeline.lower()}.csv"
        )

    try:
        run_evaluation(
            pipeline_name=args.pipeline.upper(),
            output_path=args.output,
            financebench_only=args.financebench_only,
            k=args.k,
            ground_truth_path=args.ground_truth,
        )
        return 0
    except Exception as exc:
        print(f"\nFATAL ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
