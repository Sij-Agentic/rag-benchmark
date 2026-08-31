#!/usr/bin/env python
"""LLM-as-Judge evaluation for V2 corpus results."""

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
- Semantic equivalence (Yes vs Yes, positive working capital)
- Extracting the answer from explanations (if explanation contains correct value, accept it)
- Small rounding differences (17.98 vs 18.0 are approximately equal)
- For yes/no questions, focus on the verdict, not the explanation
- For "Cannot determine" answers, mark as INCORRECT unless gold also says cannot determine

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

    print(f"\n{'='*60}")
    print(f"Pipeline {pipeline}: Judging {len(df)} answers...")
    print(f"{'='*60}\n")

    results = []
    correct_count = 0

    for idx, row in df.iterrows():
        item_id = row["item_id"]
        question = row["question"]
        gold = row["gold_answer"]
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
            print(f"  [{idx+1}/{len(df)}] {item_id}: ✗ - {result['reasoning'][:50]}")

        # Rate limit
        time.sleep(0.2)

    # Add results to dataframe
    df["judge_correct"] = [r["correct"] for r in results]
    df["judge_reasoning"] = [r["reasoning"] for r in results]

    # Summary
    accuracy = correct_count / len(df) * 100
    print(f"\n{'='*60}")
    print(f"Pipeline {pipeline} Accuracy: {correct_count}/{len(df)} ({accuracy:.1f}%)")
    print(f"{'='*60}\n")

    return df


def main():
    results_dir = Path("results")

    # Find all V2 fixed results
    csv_files = sorted(results_dir.glob("v2_pipeline_*_fixed.csv"))
    if not csv_files:
        print("No V2 fixed results found!")
        return

    all_results = []
    for csv_file in csv_files:
        df = evaluate_pipeline(csv_file)
        all_results.append(df)

        # Save individual judged results
        output_path = csv_file.with_name(csv_file.stem + "_judged.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}\n")

    # Combined summary
    print(f"\n{'='*60}")
    print("V2 CORPUS FINAL SUMMARY")
    print(f"{'='*60}\n")

    for df in all_results:
        pipeline = df["pipeline"].iloc[0]
        total = len(df)
        correct = df["judge_correct"].sum()
        accuracy = correct / total * 100
        print(f"Pipeline {pipeline}: {correct}/{total} ({accuracy:.1f}%)")

        # By dataset
        for dataset in df["source_dataset"].unique():
            subset = df[df["source_dataset"] == dataset]
            subset_correct = subset["judge_correct"].sum()
            print(f"  {dataset}: {subset_correct}/{len(subset)} ({subset_correct/len(subset)*100:.1f}%)")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
