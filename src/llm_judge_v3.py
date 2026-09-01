#!/usr/bin/env python
"""LLM-as-Judge evaluation for V3 corpus results (71 questions)."""

import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

JUDGE_PROMPT = """You are evaluating whether a predicted answer matches the gold answer.

**Question:** {question}

**Gold Answer:** {gold_answer}

**Predicted Answer:** {predicted_answer}

Determine if the predicted answer is CORRECT. Consider:
- Number format variations ($1,577 vs $1577.00 vs 1577 million are equivalent)
- Semantic equivalence (Yes vs "positive working capital")
- Extracting the answer from explanations (if explanation contains correct value, accept it)
- Small rounding differences (17.98 vs 18.0 are approximately equal)
- For yes/no questions, focus on the verdict, not the explanation
- For "Cannot determine" answers, mark as INCORRECT unless gold also says cannot determine
- For chart/visual questions, accept reasonable approximations from visual reading

Return a JSON object with:
{{
  "correct": true/false,
  "reasoning": "Brief explanation of why it's correct or incorrect"
}}

IMPORTANT: Be lenient with formatting but strict with factual correctness."""


def judge_answer(question: str, gold_answer: str, predicted_answer: str) -> dict:
    """Use LLM to judge if predicted answer matches gold answer."""

    # Skip obvious failures
    if not predicted_answer or "ERROR" in str(predicted_answer):
        return {"correct": False, "reasoning": "Error or empty prediction"}

    if pd.isna(predicted_answer) or str(predicted_answer).strip() == "":
        return {"correct": False, "reasoning": "Empty prediction"}

    # Call Gemini
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = JUDGE_PROMPT.format(
        question=question, gold_answer=gold_answer, predicted_answer=predicted_answer
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.0,
            },
        )
        result = json.loads(response.text)
        return result
    except Exception as e:
        print(f"  ⚠ Judge API error: {e}")
        return {"correct": False, "reasoning": f"Judge API error: {e}"}


def evaluate_pipeline(csv_path: Path) -> pd.DataFrame:
    """Evaluate one pipeline's results with LLM judge."""

    df = pd.read_csv(csv_path)
    pipeline = df["pipeline"].iloc[0]

    print(f"\n{'='*80}")
    print(f"Pipeline {pipeline}: Judging {len(df)} answers...")
    print(f"{'='*80}\n")

    results = []
    correct_count = 0

    for idx, row in df.iterrows():
        item_id = row.get("item_id", row.get("id", f"row_{idx}"))
        question = row["question"]
        gold = row.get("gold_answer", row.get("answer"))
        pred = row["predicted_answer"]

        # Skip if prediction has error
        if pd.notna(row.get("error")):
            results.append(
                {
                    "correct": False,
                    "reasoning": f"Pipeline error: {row['error']}",
                }
            )
            print(f"  [{idx+1}/{len(df)}] {item_id}: ✗ (pipeline error)")
            continue

        # Judge
        result = judge_answer(question, gold, pred)
        results.append(result)

        if result["correct"]:
            correct_count += 1
            print(f"  [{idx+1}/{len(df)}] {item_id}: ✓")
        else:
            reasoning_short = result["reasoning"][:60] + "..." if len(result["reasoning"]) > 60 else result["reasoning"]
            print(f"  [{idx+1}/{len(df)}] {item_id}: ✗ - {reasoning_short}")

        # Rate limit
        time.sleep(0.2)

    # Add results to dataframe
    df["judge_correct"] = [r["correct"] for r in results]
    df["judge_reasoning"] = [r["reasoning"] for r in results]

    # Summary
    accuracy = correct_count / len(df) * 100
    print(f"\n{'='*80}")
    print(f"Pipeline {pipeline} Accuracy: {correct_count}/{len(df)} ({accuracy:.1f}%)")
    print(f"{'='*80}\n")

    return df


def main():
    results_dir = Path("results")

    # Find all V3 results
    csv_files = sorted(results_dir.glob("v3_pipeline_*.csv"))
    if not csv_files:
        print("No V3 results found! Run pipelines first:")
        print("  python src/evaluate.py --pipeline A --ground-truth data/ground_truth_v3.json --output results/v3_pipeline_a.csv")
        return

    all_results = []
    summary_by_pipeline = {}

    for csv_file in csv_files:
        print(f"\n{'='*80}")
        print(f"Processing: {csv_file.name}")
        print(f"{'='*80}")

        df = evaluate_pipeline(csv_file)
        all_results.append(df)

        # Save individual judged results
        output_path = csv_file.with_name(csv_file.stem + "_judged.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}\n")

        # Store summary
        pipeline = df["pipeline"].iloc[0]
        total = len(df)
        correct = df["judge_correct"].sum()
        accuracy = correct / total * 100

        summary_by_pipeline[pipeline] = {
            "total": total,
            "correct": int(correct),
            "accuracy": accuracy,
            "by_dataset": {}
        }

        # By dataset breakdown
        for dataset in df["source_dataset"].unique():
            subset = df[df["source_dataset"] == dataset]
            subset_correct = subset["judge_correct"].sum()
            summary_by_pipeline[pipeline]["by_dataset"][dataset] = {
                "total": len(subset),
                "correct": int(subset_correct),
                "accuracy": subset_correct / len(subset) * 100
            }

    # Combined summary
    print(f"\n{'='*80}")
    print("V3 CORPUS FINAL SUMMARY (71 Questions)")
    print(f"{'='*80}\n")

    print(f"{'Pipeline':<12} {'Overall':<15} {'FinanceBench':<15} {'DocVQA':<15} {'ChartQA':<15} {'DocLayNet':<15}")
    print("-" * 87)

    for pipeline in sorted(summary_by_pipeline.keys()):
        stats = summary_by_pipeline[pipeline]
        overall = f"{stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)"

        # Get per-dataset stats
        by_ds = stats["by_dataset"]
        fb = by_ds.get("financebench", {})
        dv = by_ds.get("docvqa", {})
        cq = by_ds.get("chartqa", {})
        dl = by_ds.get("doclaynet", {})

        fb_str = f"{fb.get('correct', 0)}/{fb.get('total', 0)} ({fb.get('accuracy', 0):.0f}%)" if fb else "N/A"
        dv_str = f"{dv.get('correct', 0)}/{dv.get('total', 0)} ({dv.get('accuracy', 0):.0f}%)" if dv else "N/A"
        cq_str = f"{cq.get('correct', 0)}/{cq.get('total', 0)} ({cq.get('accuracy', 0):.0f}%)" if cq else "N/A"
        dl_str = f"{dl.get('correct', 0)}/{dl.get('total', 0)} ({dl.get('accuracy', 0):.0f}%)" if dl else "N/A"

        print(f"{'Pipeline ' + pipeline:<12} {overall:<15} {fb_str:<15} {dv_str:<15} {cq_str:<15} {dl_str:<15}")

    print("\n" + "="*80)

    # Save summary JSON
    summary_path = results_dir / "v3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_by_pipeline, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
