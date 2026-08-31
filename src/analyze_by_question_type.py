"""Analyze pipeline performance by question type: simple extraction vs calculation."""

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


def classify_question(item):
    """Classify question as simple extraction or calculation."""
    q = item["question"].lower()

    # Calculation indicators (require computing something)
    calc_indicators = [
        " average ", " ratio", "calculate", " growth rate",
        "3 year", "2 year", "turnover",
        # Specific financial ratios
        " roa", " roe", " dpo", " dso",
        # Operations on numbers
        " less ", " plus ", " divided by", " minus "
    ]

    # Simple extraction indicators
    simple_indicators = [
        "what is the fy", "what was the fy",
        "how much", "how many",
        "according to", "which", "what are",
        "does ", "is ", "did "
    ]

    # Check for calculation
    for indicator in calc_indicators:
        if indicator in q:
            return "calculation"

    # If question asks for a metric that needs formula (margin, ratio, etc.)
    # but doesn't explicitly say "calculate"
    if any(word in q for word in ["margin", "ebitda"]) and "what is" in q:
        # Check if it's asking to calculate or just extract
        # "What is the operating margin for 2018?" could be either
        # If it mentions a formula or multiple years, it's calculation
        if any(year in q for year in ["fy2015", "fy2016", "fy2017", "fy2018", "fy2019", "fy2020", "fy2021", "fy2022"]):
            # Count how many years mentioned
            years_mentioned = sum(1 for year in ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"] if year in q)
            if years_mentioned > 1:
                return "calculation"

    return "simple_extraction"


# Load data
gt = json.load(open("data/ground_truth.json"))
items = {i["id"]: i for i in gt["items"]}

a = pd.read_csv("results/pipeline_a_full50.csv")
b = pd.read_csv("results/pipeline_b_full50.csv")
c = pd.read_csv("results/pipeline_c_full50.csv")

# Classify and analyze
results = {
    "simple_extraction": {"items": [], "A": [], "B": [], "C": []},
    "calculation": {"items": [], "A": [], "B": [], "C": []}
}

for item_id in a["item_id"]:
    item = items[item_id]
    gold = item["answer"]
    qtype = classify_question(item)

    a_pred = a[a["item_id"] == item_id].iloc[0]["predicted_answer"]
    b_pred = b[b["item_id"] == item_id].iloc[0]["predicted_answer"]
    c_pred = c[c["item_id"] == item_id].iloc[0]["predicted_answer"]

    a_correct = check_correct(a_pred, gold)
    b_correct = check_correct(b_pred, gold)
    c_correct = check_correct(c_pred, gold)

    results[qtype]["items"].append(item_id)
    results[qtype]["A"].append(a_correct)
    results[qtype]["B"].append(b_correct)
    results[qtype]["C"].append(c_correct)

# Print results
print("=" * 80)
print("ANALYSIS BY QUESTION TYPE")
print("=" * 80)

for qtype in ["simple_extraction", "calculation"]:
    data = results[qtype]
    total = len(data["items"])
    a_acc = sum(data["A"])
    b_acc = sum(data["B"])
    c_acc = sum(data["C"])

    print(f"\n{qtype.upper().replace('_', ' ')} ({total} questions):")
    print(f"  Pipeline A: {a_acc}/{total} ({100*a_acc/total:.1f}%)")
    print(f"  Pipeline B: {b_acc}/{total} ({100*b_acc/total:.1f}%)")
    print(f"  Pipeline C: {c_acc}/{total} ({100*c_acc/total:.1f}%)")
    print(f"  Difference B-A: {b_acc - a_acc:+d} | C-A: {c_acc - a_acc:+d}")

# Show sample questions from each category
print("\n" + "=" * 80)
print("SAMPLE SIMPLE EXTRACTION QUESTIONS:")
print("=" * 80)
for item_id in results["simple_extraction"]["items"][:5]:
    item = items[item_id]
    print(f"\n{item_id}:")
    print(f"  Q: {item['question'][:80]}...")
    print(f"  A: {item['answer']}")

print("\n" + "=" * 80)
print("SAMPLE CALCULATION QUESTIONS:")
print("=" * 80)
for item_id in results["calculation"]["items"][:5]:
    item = items[item_id]
    print(f"\n{item_id}:")
    print(f"  Q: {item['question'][:80]}...")
    print(f"  A: {item['answer']}")

# Identify where layout helps in simple extraction
print("\n" + "=" * 80)
print("SIMPLE EXTRACTION: Where layout awareness helps")
print("=" * 80)

for i, item_id in enumerate(results["simple_extraction"]["items"]):
    a_correct = results["simple_extraction"]["A"][i]
    b_correct = results["simple_extraction"]["B"][i]
    c_correct = results["simple_extraction"]["C"][i]

    if not a_correct and (b_correct or c_correct):
        item = items[item_id]
        print(f"\n{item_id}: {item['doc_id']}")
        print(f"  Q: {item['question'][:70]}...")
        print(f"  Gold: {item['answer']}")
        print(f"  A=✗ B={'✓' if b_correct else '✗'} C={'✓' if c_correct else '✗'}")
