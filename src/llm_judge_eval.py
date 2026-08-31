"""LLM-as-judge evaluation for answer correctness.

Uses Gemini to evaluate whether predicted answers match gold answers,
handling format variations and semantic equivalence.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()


JUDGE_PROMPT = """You are evaluating whether a predicted answer matches the gold (correct) answer.

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


def judge_answer(question: str, gold_answer: str, predicted_answer: str, model: str = "gemini-2.5-flash") -> dict:
    """Use LLM to judge if predicted answer matches gold answer."""

    # Skip obvious failures
    if not predicted_answer or "ERROR" in str(predicted_answer):
        return {"correct": False, "reasoning": "Error or empty prediction"}

    if "cannot determine" in str(predicted_answer).lower() and "cannot determine" not in str(gold_answer).lower():
        return {"correct": False, "reasoning": "Predicted cannot determine but gold has answer"}

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "correct": {"type": "boolean"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["correct", "reasoning"]
                }
            }
        )

        result = json.loads(response.text)
        return result

    except Exception as e:
        print(f"Error judging answer: {e}")
        return {"correct": False, "reasoning": f"Judge error: {str(e)}"}


def evaluate_with_judge():
    """Re-evaluate all pipelines using LLM-as-judge."""

    # Load data
    gt = json.load(open("data/ground_truth.json"))
    items = {i["id"]: i for i in gt["items"]}

    a = pd.read_csv("results/pipeline_a_full50.csv")
    b = pd.read_csv("results/pipeline_b_full50.csv")
    c = pd.read_csv("results/pipeline_c_full50.csv")

    results = []
    a_correct = 0
    b_correct = 0
    c_correct = 0

    print("="*80)
    print("LLM-AS-JUDGE EVALUATION")
    print("="*80)
    print(f"\nEvaluating 50 questions with Gemini judge...\n")

    for i, item_id in enumerate(a["item_id"], 1):
        item = items[item_id]
        question = item["question"]
        gold = item["answer"]

        a_pred = str(a[a["item_id"] == item_id].iloc[0]["predicted_answer"])
        b_pred = str(b[b["item_id"] == item_id].iloc[0]["predicted_answer"])
        c_pred = str(c[c["item_id"] == item_id].iloc[0]["predicted_answer"])

        print(f"[{i}/50] {item_id[:30]}...", end=" ", flush=True)

        # Judge each pipeline
        a_judgment = judge_answer(question, gold, a_pred)
        time.sleep(0.5)  # Rate limiting

        b_judgment = judge_answer(question, gold, b_pred)
        time.sleep(0.5)

        c_judgment = judge_answer(question, gold, c_pred)
        time.sleep(0.5)

        a_ok = a_judgment["correct"]
        b_ok = b_judgment["correct"]
        c_ok = c_judgment["correct"]

        if a_ok:
            a_correct += 1
        if b_ok:
            b_correct += 1
        if c_ok:
            c_correct += 1

        print(f"A={'✓' if a_ok else '✗'} B={'✓' if b_ok else '✗'} C={'✓' if c_ok else '✗'}")

        results.append({
            "item_id": item_id,
            "doc_id": item["doc_id"],
            "question": question[:60] + "...",
            "gold": gold,
            "A_correct": a_ok,
            "B_correct": b_ok,
            "C_correct": c_ok,
            "A_reasoning": a_judgment["reasoning"],
            "B_reasoning": b_judgment["reasoning"],
            "C_reasoning": c_judgment["reasoning"],
            "A_pred": a_pred[:100],
            "B_pred": b_pred[:100],
            "C_pred": c_pred[:100],
        })

    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nAccuracy (judged by LLM):")
    print(f"  Pipeline A: {a_correct}/50 ({100*a_correct/50:.1f}%)")
    print(f"  Pipeline B: {b_correct}/50 ({100*b_correct/50:.1f}%)")
    print(f"  Pipeline C: {c_correct}/50 ({100*c_correct/50:.1f}%)")
    print(f"\nImprovement over A:")
    print(f"  Pipeline B: {b_correct - a_correct:+d} ({100*(b_correct - a_correct)/50:+.1f}%)")
    print(f"  Pipeline C: {c_correct - a_correct:+d} ({100*(c_correct - a_correct)/50:+.1f}%)")

    # Analyze
    layout_helps = df[(~df["A_correct"]) & (df["B_correct"] | df["C_correct"])]
    print(f"\n✓ Layout helps: {len(layout_helps)}/50 ({100*len(layout_helps)/50:.1f}%)")

    all_succeed = df[df["A_correct"] & df["B_correct"] & df["C_correct"]]
    print(f"✓ All succeed: {len(all_succeed)}/50 ({100*len(all_succeed)/50:.1f}%)")

    all_fail = df[(~df["A_correct"]) & (~df["B_correct"]) & (~df["C_correct"])]
    print(f"✗ All fail: {len(all_fail)}/50 ({100*len(all_fail)/50:.1f}%)")

    # Show where layout helps
    if len(layout_helps) > 0:
        print("\n" + "="*80)
        print("QUESTIONS WHERE LAYOUT AWARENESS HELPS:")
        print("="*80)
        for _, row in layout_helps.iterrows():
            print(f"\n{row['item_id']}: {row['doc_id']}")
            print(f"  Q: {row['question']}")
            print(f"  Gold: {row['gold']}")
            print(f"  A={'✓' if row['A_correct'] else '✗'} B={'✓' if row['B_correct'] else '✗'} C={'✓' if row['C_correct'] else '✗'}")
            if not row["A_correct"]:
                print(f"  Why A failed: {row['A_reasoning']}")
            if row["B_correct"]:
                print(f"  Why B succeeded: {row['B_reasoning']}")

    # Save
    df.to_csv("results/llm_judge_evaluation.csv", index=False)
    print(f"\nResults saved to: results/llm_judge_evaluation.csv")

    return df


if __name__ == "__main__":
    df = evaluate_with_judge()
