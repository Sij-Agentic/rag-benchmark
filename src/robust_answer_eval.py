"""Robust answer evaluation that handles formatting variations and semantic equivalence."""

import json
import re
from pathlib import Path

import pandas as pd


def extract_all_numbers(text):
    """Extract all numbers from text, normalized."""
    if not text:
        return []
    text = str(text).replace(",", "").replace("$", "").replace("%", "")
    # Find all numbers including decimals and negatives
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return [float(n) for n in numbers if n and n != '.']


def check_number_match(pred, gold):
    """Check if predicted and gold answers contain matching numbers."""
    pred_nums = extract_all_numbers(pred)
    gold_nums = extract_all_numbers(gold)

    if not pred_nums or not gold_nums:
        return False

    # Check if the main number matches (usually first or largest)
    gold_main = max(gold_nums, key=abs) if gold_nums else None

    for pred_num in pred_nums:
        if gold_main is not None:
            # Allow small relative error for rounding
            if abs(pred_num - gold_main) < 0.01 * abs(gold_main) + 0.01:
                return True
            # Also check exact match after rounding
            if abs(pred_num - gold_main) < 1.0:
                return True

    return False


def check_semantic_match(pred, gold):
    """Check semantic equivalence for yes/no and qualitative answers."""
    pred_lower = str(pred).lower().strip()
    gold_lower = str(gold).lower().strip()

    # Exact match
    if pred_lower == gold_lower:
        return True

    # Yes/No questions
    if any(word in gold_lower for word in ["yes", "no", "true", "false"]):
        # Extract the yes/no from both
        pred_has_yes = "yes" in pred_lower or "positive" in pred_lower or "true" in pred_lower
        pred_has_no = "no" in pred_lower or "negative" in pred_lower or "false" in pred_lower or "not" in pred_lower

        gold_has_yes = "yes" in gold_lower or "positive" in gold_lower or "true" in gold_lower
        gold_has_no = "no" in gold_lower or "negative" in gold_lower or "false" in gold_lower or "not" in gold_lower

        if gold_has_yes and pred_has_yes and not pred_has_no:
            return True
        if gold_has_no and pred_has_no and not pred_has_yes:
            return True

    # Check if gold answer is substring of prediction (common for short answers)
    if len(gold_lower) > 5 and gold_lower in pred_lower:
        return True

    return False


def robust_answer_check(pred, gold):
    """
    Robust answer checking that handles:
    - Number formatting variations
    - Semantic equivalence
    - Extracting answers from explanations
    - Multiple phrasings of the same answer
    """
    if not pred or not gold:
        return False

    # Skip errors
    if "ERROR" in str(pred) or "error" in str(pred).lower():
        return False

    # Cannot determine = wrong
    if "cannot determine" in str(pred).lower():
        return False

    # Check number match
    if check_number_match(pred, gold):
        return True

    # Check semantic match
    if check_semantic_match(pred, gold):
        return True

    return False


def evaluate_all_pipelines():
    """Re-evaluate all three pipelines with robust answer checking."""
    # Load data
    gt = json.load(open("data/ground_truth.json"))
    items = {i["id"]: i for i in gt["items"]}

    a = pd.read_csv("results/pipeline_a_full50.csv")
    b = pd.read_csv("results/pipeline_b_full50.csv")
    c = pd.read_csv("results/pipeline_c_full50.csv")

    # Evaluate each pipeline
    results = []
    a_correct = 0
    b_correct = 0
    c_correct = 0

    for item_id in a["item_id"]:
        item = items[item_id]
        gold = item["answer"]

        a_pred = a[a["item_id"] == item_id].iloc[0]["predicted_answer"]
        b_pred = b[b["item_id"] == item_id].iloc[0]["predicted_answer"]
        c_pred = c[c["item_id"] == item_id].iloc[0]["predicted_answer"]

        a_ok = robust_answer_check(a_pred, gold)
        b_ok = robust_answer_check(b_pred, gold)
        c_ok = robust_answer_check(c_pred, gold)

        if a_ok:
            a_correct += 1
        if b_ok:
            b_correct += 1
        if c_ok:
            c_correct += 1

        results.append({
            "item_id": item_id,
            "doc_id": item["doc_id"],
            "question": item["question"][:60] + "...",
            "gold": gold,
            "A_correct": a_ok,
            "B_correct": b_ok,
            "C_correct": c_ok,
            "A_pred": str(a_pred)[:100],
            "B_pred": str(b_pred)[:100],
            "C_pred": str(c_pred)[:100],
        })

    df = pd.DataFrame(results)

    print("="*80)
    print("ROBUST ANSWER EVALUATION")
    print("="*80)
    print(f"\nTotal questions: 50")
    print(f"\nAccuracy:")
    print(f"  Pipeline A: {a_correct}/50 ({100*a_correct/50:.1f}%)")
    print(f"  Pipeline B: {b_correct}/50 ({100*b_correct/50:.1f}%)")
    print(f"  Pipeline C: {c_correct}/50 ({100*c_correct/50:.1f}%)")
    print(f"\nImprovement over naive (Pipeline A):")
    print(f"  Pipeline B: {b_correct - a_correct:+d} ({100*(b_correct - a_correct)/50:+.1f}%)")
    print(f"  Pipeline C: {c_correct - a_correct:+d} ({100*(c_correct - a_correct)/50:+.1f}%)")

    # Analyze where layout helps
    layout_helps = df[(~df["A_correct"]) & (df["B_correct"] | df["C_correct"])]
    print(f"\n✓ Questions where layout awareness helps: {len(layout_helps)}/50")

    if len(layout_helps) > 0:
        print("\nDetails:")
        for _, row in layout_helps.iterrows():
            print(f"\n  {row['item_id']}: {row['doc_id']}")
            print(f"    Q: {row['question']}")
            print(f"    Gold: {row['gold']}")
            print(f"    A={row['A_correct']} B={row['B_correct']} C={row['C_correct']}")

    # All succeed
    all_succeed = df[df["A_correct"] & df["B_correct"] & df["C_correct"]]
    print(f"\n✓ Questions all pipelines solve: {len(all_succeed)}/50")

    # All fail
    all_fail = df[(~df["A_correct"]) & (~df["B_correct"]) & (~df["C_correct"])]
    print(f"\n✗ Questions all pipelines fail: {len(all_fail)}/50")

    # Save results
    df.to_csv("results/robust_evaluation.csv", index=False)
    print(f"\nResults saved to: results/robust_evaluation.csv")

    return df


if __name__ == "__main__":
    df = evaluate_all_pipelines()
