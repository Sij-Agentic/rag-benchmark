#!/usr/bin/env python
"""Build V3 corpus - 80-100 questions from multiple datasets.

Expansion from V2 (24 questions):
- FinanceBench: 8 → 20 questions
- DocVQA: 15 → 30 questions
- DocLayNet: 1 → 10 questions
- ChartQA: 0 → 20 questions (NEW, fix image issues)
Total: 80 questions

Lessons from V2:
- Only include hard documents (naive < 40%)
- Use streaming datasets to avoid downloading all
- Fix ChartQA image handling
- Sequential execution for evaluation
"""

import io
import json
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDFS = DATA_DIR / "raw_pdfs"


def collect_financebench(target: int = 20) -> list[dict]:
    """Collect FinanceBench questions (expand from 8 to 20).

    Strategy: Keep existing 8 hard questions, add 12 more with similar complexity.
    """
    print(f"\n{'='*80}")
    print(f"Collecting FinanceBench: Target {target} questions")
    print(f"{'='*80}")

    # Load existing V2 FinanceBench questions (already proven hard)
    v2_path = DATA_DIR / "ground_truth_v2.json"
    if v2_path.exists():
        with open(v2_path) as f:
            v2_data = json.load(f)
        existing = [item for item in v2_data["items"] if item["source_dataset"] == "financebench"]
        print(f"Keeping {len(existing)} existing hard FinanceBench questions")
    else:
        existing = []

    # Load full FinanceBench dataset
    dataset = load_dataset("PatronusAI/financebench", split="train", streaming=True)

    # Collect additional questions
    collected = []
    existing_ids = {item["id"] for item in existing}

    # Criteria for hard questions:
    # 1. Multi-table questions (requires multiple financial statements)
    # 2. Ratio/calculation questions (not simple lookup)
    # 3. Questions requiring cross-statement data

    keywords_hard = ["ratio", "calculate", "capex", "EBITDA", "turnover", "working capital",
                     "intensity", "gross margin", "operating margin", "multiple statements"]

    for item in dataset:
        if len(collected) + len(existing) >= target:
            break

        item_id = item["doc_name"] + "_" + str(item.get("unique_id", ""))

        # Skip if already in existing
        if item_id in existing_ids:
            continue

        # Check if question seems hard (contains calculation keywords)
        question = item["question"].lower()
        if any(keyword in question for keyword in keywords_hard):
            collected.append({
                "id": item_id,
                "source_dataset": "financebench",
                "doc_id": item["doc_name"],
                "question": item["question"],
                "answer": item["answer"],
                "pdf_path": f"data/raw_pdfs/financebench/{item['doc_name']}.pdf",
                "target_pages": item.get("evidence", []),
                "layout_challenge": "dense nested financial tables",
                "qa_source": "financebench-human"
            })

            if (len(collected) + len(existing)) % 5 == 0:
                print(f"  Collected {len(collected) + len(existing)}/{target}...")

    print(f"✓ Collected {len(collected)} new FinanceBench questions")
    return existing + collected


def collect_docvqa(target: int = 30) -> list[dict]:
    """Collect DocVQA questions (expand from 15 to 30).

    Strategy: Diverse form types with spatial layouts.
    """
    print(f"\n{'='*80}")
    print(f"Collecting DocVQA: Target {target} questions")
    print(f"{'='*80}")

    # Load existing V2 DocVQA questions
    v2_path = DATA_DIR / "ground_truth_v2.json"
    if v2_path.exists():
        with open(v2_path) as f:
            v2_data = json.load(f)
        existing = [item for item in v2_data["items"] if item["source_dataset"] == "docvqa"]
        print(f"Keeping {len(existing)} existing DocVQA questions")
    else:
        existing = []

    # Load DocVQA dataset (validation split, streaming)
    dataset = load_dataset("pixparse/docvqa-single-page-questions", split="validation", streaming=True)

    collected = []
    docvqa_dir = RAW_PDFS / "docvqa"
    docvqa_dir.mkdir(parents=True, exist_ok=True)

    existing_doc_ids = {item["id"] for item in existing}

    for idx, item in enumerate(dataset):
        if len(collected) + len(existing) >= target:
            break

        question = item["question"]

        # Filter criteria (from V2)
        if len(question) > 120:  # Too long
            continue
        if question.lower().startswith(("is ", "does ", "do ", "are ")):  # Yes/no
            continue
        if not item.get("answers"):  # No answer
            continue

        # Get answer (take first if multiple)
        answer = item["answers"][0] if isinstance(item["answers"], list) else item["answers"]

        doc_id = f"docvqa_{len(existing) + len(collected) + 1:03d}"

        # Skip if already exists
        if doc_id in existing_doc_ids:
            continue

        # Save image as PDF
        pdf_path = docvqa_dir / f"{doc_id}.pdf"

        try:
            image = item["image"]

            # Handle different image formats (FIX for ChartQA issue)
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif not isinstance(image, Image.Image):
                # Try to convert if it's some other format
                image = Image.open(io.BytesIO(image))

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Save as PDF
            image.save(pdf_path, "PDF", resolution=100.0)

            collected.append({
                "id": doc_id,
                "source_dataset": "docvqa",
                "doc_id": doc_id,
                "question": question,
                "answer": answer,
                "pdf_path": f"data/raw_pdfs/docvqa/{doc_id}.pdf",
                "target_pages": 1,
                "layout_challenge": "form with spatial field-value relationships",
                "qa_source": "docvqa-human"
            })

            if len(collected) % 5 == 0:
                print(f"  Saved {len(collected)} DocVQA PDFs...")

        except Exception as e:
            print(f"  Error saving {doc_id}: {e}")
            continue

    print(f"✓ Collected {len(collected)} new DocVQA questions")
    return existing + collected


def collect_doclaynet(target: int = 10) -> list[dict]:
    """Collect DocLayNet questions (expand from 1 to 10).

    Strategy: Diverse document types (scientific, financial, government).
    """
    print(f"\n{'='*80}")
    print(f"Collecting DocLayNet: Target {target} questions")
    print(f"{'='*80}")

    # Load existing V2 DocLayNet questions
    v2_path = DATA_DIR / "ground_truth_v2.json"
    if v2_path.exists():
        with open(v2_path) as f:
            v2_data = json.load(f)
        existing = [item for item in v2_data["items"] if item["source_dataset"] == "doclaynet"]
        print(f"Keeping {len(existing)} existing DocLayNet questions")
    else:
        existing = []

    # For now, we'll manually curate DocLayNet questions since they need
    # careful question generation (no built-in QA pairs)
    # This is a placeholder - in practice, you'd sample documents and
    # manually create questions about layout elements

    print(f"⚠ DocLayNet expansion requires manual curation")
    print(f"  Current: {len(existing)} questions")
    print(f"  Target: {target} questions")
    print(f"  TODO: Add {target - len(existing)} more questions manually")

    return existing


def collect_chartqa(target: int = 20) -> list[dict]:
    """Collect ChartQA questions (NEW - was 0 in V2).

    Strategy: Fix image handling issues from V2, collect diverse chart types.
    """
    print(f"\n{'='*80}")
    print(f"Collecting ChartQA: Target {target} questions (NEW)")
    print(f"{'='*80}")

    # Load ChartQA dataset (val split for quick testing)
    try:
        dataset = load_dataset("HuggingFaceM4/ChartQA", split="val", streaming=True)
    except Exception as e:
        print(f"✗ Error loading ChartQA: {e}")
        return []

    collected = []
    chartqa_dir = RAW_PDFS / "chartqa"
    chartqa_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(dataset):
        if len(collected) >= target:
            break

        # ChartQA uses 'query' and 'label' instead of 'question' and 'answer'
        question = item.get("query", "")
        answer = item.get("label", [])

        # Extract answer from list if needed
        if isinstance(answer, list) and answer:
            answer = answer[0]

        # Filter criteria
        if not question or len(question) > 150:  # Too long or empty
            continue
        if question.lower().startswith(("is ", "does ", "do ")):  # Prefer extractive
            continue
        if not answer:
            continue

        doc_id = f"chartqa_{len(collected) + 1:03d}"
        pdf_path = chartqa_dir / f"{doc_id}.pdf"

        try:
            # Get image
            image = item["image"]

            # CRITICAL FIX: Handle bytes vs PIL Image
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif hasattr(image, 'read'):  # File-like object
                image = Image.open(image)
            elif not isinstance(image, Image.Image):
                # Last resort: try to convert
                try:
                    image = Image.open(io.BytesIO(image))
                except:
                    print(f"  Error: Cannot convert image type {type(image)} for {doc_id}")
                    continue

            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Save as PDF
            image.save(pdf_path, "PDF", resolution=100.0)

            collected.append({
                "id": doc_id,
                "source_dataset": "chartqa",
                "doc_id": doc_id,
                "question": question,
                "answer": str(answer),
                "pdf_path": f"data/raw_pdfs/chartqa/{doc_id}.pdf",
                "target_pages": 1,
                "layout_challenge": "chart/graph requiring visual reading",
                "qa_source": "chartqa-human"
            })

            if len(collected) % 5 == 0:
                print(f"  Saved {len(collected)} ChartQA PDFs...")

        except Exception as e:
            print(f"  Error saving chartqa_{len(collected) + 1:03d}: {e}")
            continue

    print(f"✓ Collected {len(collected)} ChartQA questions")
    return collected


def build_v3_corpus():
    """Build complete V3 corpus."""
    print("="*80)
    print("BUILDING V3 CORPUS")
    print("="*80)

    # Collect from each dataset
    financebench_items = collect_financebench(target=20)
    docvqa_items = collect_docvqa(target=30)
    doclaynet_items = collect_doclaynet(target=10)
    chartqa_items = collect_chartqa(target=20)

    # Combine all items
    all_items = financebench_items + docvqa_items + doclaynet_items + chartqa_items

    # Shuffle for variety (but keep reproducible)
    random.seed(42)
    random.shuffle(all_items)

    # Create ground truth JSON
    ground_truth = {
        "version": "v3",
        "date_created": "2026-08-31",
        "description": "V3 corpus - 80 hard questions across 4 datasets",
        "items": all_items
    }

    # Save
    output_path = DATA_DIR / "ground_truth_v3.json"
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("V3 CORPUS COMPLETE")
    print("="*80)
    print(f"\nTotal questions: {len(all_items)}")

    by_dataset = {}
    for item in all_items:
        dataset = item["source_dataset"]
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1

    for dataset, count in sorted(by_dataset.items()):
        print(f"  - {dataset}: {count}")

    print(f"\nSaved to: {output_path}")
    print("\nNext step: Run pipelines with --ground-truth data/ground_truth_v3.json")


if __name__ == "__main__":
    build_v3_corpus()
