"""Debug why simple extraction questions are failing."""

import json
import re

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
    q = item['question'].lower()
    calc_indicators = [' average ', ' ratio', 'calculate', ' growth rate', '3 year', '2 year', 'turnover', ' roa', ' roe', ' dpo', ' dso', ' less ', ' plus ', ' divided by', ' minus ']
    for indicator in calc_indicators:
        if indicator in q:
            return 'calculation'
    if any(word in q for word in ['margin', 'ebitda']) and 'what is' in q:
        years_mentioned = sum(1 for year in ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'] if year in q)
        if years_mentioned > 1:
            return 'calculation'
    return 'simple_extraction'

# Load data
gt = json.load(open('data/ground_truth.json'))
items = {i['id']: i for i in gt['items']}
a = pd.read_csv('results/pipeline_a_full50.csv')
b = pd.read_csv('results/pipeline_b_full50.csv')
c = pd.read_csv('results/pipeline_c_full50.csv')

# Filter to simple extraction with no errors
simple_no_errors = []
for item_id in a['item_id']:
    item = items[item_id]
    if classify_question(item) == 'simple_extraction':
        a_pred = str(a[a['item_id'] == item_id].iloc[0]['predicted_answer'])
        b_pred = str(b[b['item_id'] == item_id].iloc[0]['predicted_answer'])
        c_pred = str(c[c['item_id'] == item_id].iloc[0]['predicted_answer'])

        if 'ERROR' not in a_pred and 'ERROR' not in b_pred and 'ERROR' not in c_pred:
            simple_no_errors.append(item_id)

# Calculate accuracy on clean subset
a_correct = 0
b_correct = 0
c_correct = 0

print("="*80)
print(f"SIMPLE EXTRACTION - CLEAN SUBSET ({len(simple_no_errors)} questions)")
print("="*80)

failures_all_three = []

for item_id in simple_no_errors:
    item = items[item_id]
    gold = item['answer']

    a_pred = a[a['item_id'] == item_id].iloc[0]['predicted_answer']
    b_pred = b[b['item_id'] == item_id].iloc[0]['predicted_answer']
    c_pred = c[c['item_id'] == item_id].iloc[0]['predicted_answer']

    a_ok = check_correct(a_pred, gold)
    b_ok = check_correct(b_pred, gold)
    c_ok = check_correct(c_pred, gold)

    if a_ok:
        a_correct += 1
    if b_ok:
        b_correct += 1
    if c_ok:
        c_correct += 1

    if not a_ok and not b_ok and not c_ok:
        failures_all_three.append({
            'id': item_id,
            'doc': item['doc_id'],
            'question': item['question'],
            'gold': gold,
            'a_pred': str(a_pred)[:80],
            'b_pred': str(b_pred)[:80],
            'c_pred': str(c_pred)[:80],
            'a_retrieved': a[a['item_id'] == item_id].iloc[0]['retrieval_hit_at_k'],
            'b_retrieved': b[b['item_id'] == item_id].iloc[0]['retrieval_hit_at_k'],
            'c_retrieved': c[c['item_id'] == item_id].iloc[0]['retrieval_hit_at_k'],
        })

print(f"\nAccuracy on {len(simple_no_errors)} clean simple extraction questions:")
print(f"  Pipeline A: {a_correct}/{len(simple_no_errors)} ({100*a_correct/len(simple_no_errors):.1f}%)")
print(f"  Pipeline B: {b_correct}/{len(simple_no_errors)} ({100*b_correct/len(simple_no_errors):.1f}%)")
print(f"  Pipeline C: {c_correct}/{len(simple_no_errors)} ({100*c_correct/len(simple_no_errors):.1f}%)")
print(f"\nAll three fail: {len(failures_all_three)} questions")

print("\n" + "="*80)
print("FAILURES ACROSS ALL THREE PIPELINES (first 5):")
print("="*80)

for fail in failures_all_three[:5]:
    print(f"\n{fail['id']}: {fail['doc']}")
    print(f"  Q: {fail['question'][:70]}...")
    print(f"  Gold: {fail['gold']}")
    print(f"  Retrieval: A={fail['a_retrieved']} B={fail['b_retrieved']} C={fail['c_retrieved']}")
    print(f"  A predicted: {fail['a_pred']}")
    print(f"  B predicted: {fail['b_pred']}")
    print(f"  C predicted: {fail['c_pred']}")

print(f"\n\nConclusion: {len(failures_all_three)}/{len(simple_no_errors)} simple extraction questions fail across ALL pipelines.")
print("This suggests the questions themselves may not have direct extractable answers,")
print("or the LLM is struggling even with correct table structure.")
