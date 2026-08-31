"""Detailed per-question comparison across all three pipelines."""

import json
import re
from pathlib import Path

import pandas as pd


def normalize_number(text):
    if not text:
        return ""
    text = str(text).replace(",", "").replace("$", "").replace("%", "").strip()
    match = re.search(r"-?\d+\.?\d*", text)
    return match.group(0) if match else text


def check_correct(pred, gold):
    if not pred or not gold or "ERROR" in str(pred):
        return False
    if str(pred).strip().lower() == str(gold).strip().lower():
        return True
    pred_num, gold_num = normalize_number(pred), normalize_number(gold)
    if pred_num and gold_num:
        try:
            return abs(float(pred_num) - float(gold_num)) < 0.01
        except:
            return pred_num == gold_num
    return str(gold).strip().lower() in str(pred).strip().lower()


# Load results
a = pd.read_csv("results/pipeline_a_full50.csv")
b = pd.read_csv("results/pipeline_b_full50.csv")
c = pd.read_csv("results/pipeline_c_full50.csv")

gt = json.load(open("data/ground_truth.json"))
items = {i["id"]: i for i in gt["items"]}

# Compute correctness for each
results = []
for item_id in a["item_id"]:
    gold = items[item_id]["answer"]

    a_row = a[a["item_id"] == item_id].iloc[0]
    b_row = b[b["item_id"] == item_id].iloc[0]
    c_row = c[c["item_id"] == item_id].iloc[0]

    a_pred = a_row["predicted_answer"]
    b_pred = b_row["predicted_answer"]
    c_pred = c_row["predicted_answer"]

    a_correct = check_correct(a_pred, gold)
    b_correct = check_correct(b_pred, gold)
    c_correct = check_correct(c_pred, gold)

    results.append({
        "item_id": item_id,
        "doc_id": items[item_id]["doc_id"],
        "question": items[item_id]["question"][:80] + "...",
        "gold": gold,
        "A_correct": "✓" if a_correct else "✗",
        "B_correct": "✓" if b_correct else "✗",
        "C_correct": "✓" if c_correct else "✗",
        "A_pred": str(a_pred)[:60] if not a_correct else "",
        "B_pred": str(b_pred)[:60] if not b_correct else "",
        "C_pred": str(c_pred)[:60] if not c_correct else "",
    })

df = pd.DataFrame(results)

print("="*80)
print("DETAILED PIPELINE COMPARISON")
print("="*80)

# Summary
a_total = (df["A_correct"] == "✓").sum()
b_total = (df["B_correct"] == "✓").sum()
c_total = (df["C_correct"] == "✓").sum()

print(f"\nAccuracy:")
print(f"  Pipeline A: {a_total}/50 ({100*a_total/50:.1f}%)")
print(f"  Pipeline B: {b_total}/50 ({100*b_total/50:.1f}%)")
print(f"  Pipeline C: {c_total}/50 ({100*c_total/50:.1f}%)")

# Questions where all fail
all_fail = df[(df["A_correct"] == "✗") & (df["B_correct"] == "✗") & (df["C_correct"] == "✗")]
print(f"\n✗ ALL FAIL: {len(all_fail)}/50 questions")

# Questions where B or C succeeds but A fails
layout_helps = df[(df["A_correct"] == "✗") & ((df["B_correct"] == "✓") | (df["C_correct"] == "✓"))]
print(f"\n✓ LAYOUT HELPS: {len(layout_helps)}/50 questions")
if len(layout_helps) > 0:
    print("\nQuestions where layout awareness helped:")
    for _, row in layout_helps.iterrows():
        print(f"\n  {row['item_id']}: {row['doc_id']}")
        print(f"    Q: {row['question']}")
        print(f"    Gold: {row['gold']}")
        print(f"    A={row['A_correct']} B={row['B_correct']} C={row['C_correct']}")

# Questions where all succeed
all_succeed = df[(df["A_correct"] == "✓") & (df["B_correct"] == "✓") & (df["C_correct"] == "✓")]
print(f"\n✓ ALL SUCCEED: {len(all_succeed)}/50 questions (easy)")

# Save detailed CSV
df.to_csv("results/detailed_comparison.csv", index=False)
print(f"\nDetailed results saved to: results/detailed_comparison.csv")
