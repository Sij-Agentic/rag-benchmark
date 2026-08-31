"""Compare pipeline results and identify hard vs easy documents.

Analyzes answer correctness across pipelines A, B, C to find:
- Easy documents: All pipelines get correct answer
- Hard documents: Pipeline A fails, B/C succeed (or all fail)
"""

import json
import re
from pathlib import Path

import pandas as pd


def normalize_number(text: str) -> str:
    """Normalize numbers for comparison."""
    if not text:
        return ""
    # Remove common formatting
    text = text.replace(",", "").replace("$", "").replace("%", "").strip()
    # Extract first number found
    match = re.search(r"-?\d+\.?\d*", text)
    return match.group(0) if match else text


def check_answer_correct(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    if not predicted or not gold:
        return False

    # Exact match (case-insensitive)
    if predicted.strip().lower() == gold.strip().lower():
        return True

    # Normalized number match
    pred_num = normalize_number(predicted)
    gold_num = normalize_number(gold)

    if pred_num and gold_num:
        try:
            # Try comparing as floats (handles 1577 vs 1577.00)
            return abs(float(pred_num) - float(gold_num)) < 0.01
        except ValueError:
            return pred_num == gold_num

    # Contains match (gold is substring of predicted)
    if gold.strip().lower() in predicted.strip().lower():
        return True

    return False


def main():
    # Load ground truth
    gt_path = Path("data/ground_truth.json")
    with open(gt_path) as f:
        gt_data = json.load(f)

    gt_items = {item["id"]: item for item in gt_data["items"]}

    # Load pipeline results
    results = {}
    for pipeline in ["A", "B", "C"]:
        csv_path = Path(f"results/pipeline_{pipeline.lower()}_full50.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            results[pipeline] = df
            print(f"Loaded Pipeline {pipeline}: {len(df)} rows")
        else:
            print(f"Pipeline {pipeline} results not found: {csv_path}")

    if not results:
        print("No results to compare!")
        return

    # Compute answer correctness for each pipeline
    for pipeline, df in results.items():
        correctness = []
        for _, row in df.iterrows():
            item_id = row["item_id"]
            predicted = str(row.get("predicted_answer", ""))
            gold = gt_items[item_id]["answer"]
            correct = check_answer_correct(predicted, gold)
            correctness.append(correct)

        df[f"answer_correct_{pipeline}"] = correctness
        acc = sum(correctness) / len(correctness) * 100
        print(f"Pipeline {pipeline} accuracy: {acc:.1f}% ({sum(correctness)}/{len(correctness)})")

    # Merge all results
    if len(results) == 1:
        combined = list(results.values())[0]
    else:
        combined = results["A"].copy()
        for pipeline in ["B", "C"]:
            if pipeline in results:
                suffix = f"_{pipeline}"
                merge_cols = ["item_id"]
                combined = combined.merge(
                    results[pipeline][["item_id", "predicted_answer"]].rename(
                        columns={"predicted_answer": f"predicted_answer{suffix}"}
                    ),
                    on="item_id",
                    how="outer"
                )
                combined = combined.merge(
                    results[pipeline][["item_id", f"answer_correct_{pipeline}"]],
                    on="item_id",
                    how="outer"
                )

    # Classify documents by difficulty
    print("\n" + "=" * 70)
    print("DOCUMENT CLASSIFICATION")
    print("=" * 70)

    if "answer_correct_A" in combined.columns:
        # Easy: All pipelines correct (or at least A is correct)
        if all(f"answer_correct_{p}" in combined.columns for p in ["A", "B", "C"]):
            easy_mask = (
                combined["answer_correct_A"] &
                combined["answer_correct_B"] &
                combined["answer_correct_C"]
            )
            hard_mask = ~combined["answer_correct_A"] & (
                combined["answer_correct_B"] | combined["answer_correct_C"]
            )
            very_hard_mask = (
                ~combined["answer_correct_A"] &
                ~combined["answer_correct_B"] &
                ~combined["answer_correct_C"]
            )
        elif "answer_correct_B" in combined.columns:
            easy_mask = combined["answer_correct_A"] & combined["answer_correct_B"]
            hard_mask = ~combined["answer_correct_A"] & combined["answer_correct_B"]
            very_hard_mask = ~combined["answer_correct_A"] & ~combined["answer_correct_B"]
        else:
            easy_mask = combined["answer_correct_A"]
            hard_mask = ~combined["answer_correct_A"]
            very_hard_mask = pd.Series([False] * len(combined))

        easy_docs = combined[easy_mask]
        hard_docs = combined[hard_mask]
        very_hard_docs = combined[very_hard_mask]

        print(f"\n✓ EASY ({len(easy_docs)}): All pipelines correct")
        print(f"⚠ HARD ({len(hard_docs)}): A fails, B/C succeed")
        print(f"✗ VERY HARD ({len(very_hard_docs)}): All pipelines fail")

        # Show hard documents
        if len(hard_docs) > 0:
            print("\n" + "=" * 70)
            print("HARD DOCUMENTS (where layout awareness helps)")
            print("=" * 70)
            for _, row in hard_docs.iterrows():
                item = gt_items[row["item_id"]]
                print(f"\n{row['item_id']}")
                print(f"  Document: {item['doc_id']}")
                print(f"  Layout: {item['layout_challenge']}")
                print(f"  Question: {item['question'][:80]}...")
                print(f"  Gold: {item['answer']}")
                print(f"  Pipeline A: {row.get('predicted_answer', 'N/A')[:60]}...")
                if "predicted_answer_B" in row:
                    print(f"  Pipeline B: {row.get('predicted_answer_B', 'N/A')[:60]}...")
                if "predicted_answer_C" in row:
                    print(f"  Pipeline C: {row.get('predicted_answer_C', 'N/A')[:60]}...")

        # Show very hard documents
        if len(very_hard_docs) > 0:
            print("\n" + "=" * 70)
            print("VERY HARD DOCUMENTS (all pipelines struggle)")
            print("=" * 70)
            for _, row in very_hard_docs.iterrows():
                item = gt_items[row["item_id"]]
                print(f"\n{row['item_id']}")
                print(f"  Document: {item['doc_id']}")
                print(f"  Question: {item['question'][:80]}...")

        # Show easy documents to discard
        if len(easy_docs) > 15:
            print(f"\n✓ {len(easy_docs)} easy documents (can be discarded)")
            print("Sample easy docs:")
            for _, row in easy_docs.head(10).iterrows():
                item = gt_items[row["item_id"]]
                print(f"  - {item['doc_id']} ({item['layout_challenge']})")

        # Save classification
        output_path = Path("results/document_classification.csv")
        classification = []
        for _, row in combined.iterrows():
            item = gt_items[row["item_id"]]
            if easy_mask[row.name]:
                difficulty = "easy"
            elif hard_mask[row.name]:
                difficulty = "hard"
            elif very_hard_mask[row.name]:
                difficulty = "very_hard"
            else:
                difficulty = "unknown"

            classification.append({
                "item_id": row["item_id"],
                "doc_id": item["doc_id"],
                "difficulty": difficulty,
                "layout_challenge": item["layout_challenge"],
                "source_dataset": item["source_dataset"],
                "A_correct": row.get("answer_correct_A", None),
                "B_correct": row.get("answer_correct_B", None),
                "C_correct": row.get("answer_correct_C", None),
            })

        pd.DataFrame(classification).to_csv(output_path, index=False)
        print(f"\nClassification saved to: {output_path}")


if __name__ == "__main__":
    main()
