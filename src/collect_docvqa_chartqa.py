"""Download samples from DocVQA and ChartQA datasets.

DocVQA: Forms, receipts, invoices with spatial layouts
ChartQA: Bar/line/pie charts requiring visual reading
"""

import json
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def download_docvqa_samples(num_samples=15, output_dir="data/raw_pdfs/docvqa"):
    """Download sample questions from DocVQA dataset.

    DocVQA focuses on document VQA with forms, receipts, invoices.
    We want questions that require spatial understanding.
    """
    print(f"Loading DocVQA dataset...")

    # DocVQA is available on HuggingFace
    # Dataset: "lmms-lab/DocVQA" or similar
    # Let's try the validation set for quality

    try:
        dataset = load_dataset("HuggingFaceM4/DocumentVQA", split="validation")
        print(f"Loaded {len(dataset)} validation examples")
    except Exception as e:
        print(f"Error loading DocVQA: {e}")
        print("Trying alternative source...")
        # Alternative: Load from original source or use sample
        return []

    # Filter for good questions (not too long, not yes/no)
    filtered = []
    for item in dataset:
        question = item['question']

        # Skip overly complex or yes/no questions
        if len(question) > 150:
            continue
        if question.lower().startswith(('is ', 'are ', 'does ', 'did ')):
            continue

        filtered.append(item)

    # Sample randomly
    random.seed(42)
    samples = random.sample(filtered, min(num_samples, len(filtered)))

    # Download images and create entries
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    entries = []
    for i, item in enumerate(samples, 1):
        # Download image
        image = item['image']
        doc_id = f"docvqa_{i:03d}"

        # Save as PDF (convert from image)
        image_path = output_path / f"{doc_id}.png"
        image.save(image_path)

        # Convert to PDF
        pdf_path = output_path / f"{doc_id}.pdf"
        image.save(pdf_path, "PDF", resolution=100.0)

        # Create ground truth entry
        entry = {
            "id": f"docvqa_{i:03d}",
            "source_dataset": "docvqa",
            "doc_id": doc_id,
            "pdf_path": str(pdf_path),
            "question": item['question'],
            "answer": item['answers'][0],  # DocVQA has multiple answers, take first
            "target_pages": [1],  # Single-page documents
            "layout_challenge": "form with spatial field-value relationships",
            "qa_source": "docvqa-human"
        }
        entries.append(entry)

        print(f"  [{i}/{num_samples}] {doc_id}: {item['question'][:50]}...")

    return entries


def download_chartqa_samples(num_samples=15, output_dir="data/raw_pdfs/chartqa"):
    """Download sample questions from ChartQA dataset.

    ChartQA focuses on reading values from charts (bar, line, pie).
    Perfect for testing VLM vs text extraction.
    """
    print(f"\nLoading ChartQA dataset...")

    try:
        dataset = load_dataset("ahmed-masry/ChartQA", split="validation")
        print(f"Loaded {len(dataset)} validation examples")
    except Exception as e:
        print(f"Error loading ChartQA: {e}")
        return []

    # Filter for good questions
    filtered = []
    for item in dataset:
        question = item['query']

        # Skip overly long questions
        if len(question) > 150:
            continue

        filtered.append(item)

    # Sample randomly
    random.seed(43)
    samples = random.sample(filtered, min(num_samples, len(filtered)))

    # Download and save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    entries = []
    for i, item in enumerate(samples, 1):
        # Download chart image
        image = item['image']
        doc_id = f"chartqa_{i:03d}"

        # Save as PDF
        image_path = output_path / f"{doc_id}.png"
        image.save(image_path)

        pdf_path = output_path / f"{doc_id}.pdf"
        image.save(pdf_path, "PDF", resolution=100.0)

        # Create ground truth entry
        entry = {
            "id": f"chartqa_{i:03d}",
            "source_dataset": "chartqa",
            "doc_id": doc_id,
            "pdf_path": str(pdf_path),
            "question": item['query'],
            "answer": item['label'],
            "target_pages": [1],
            "layout_challenge": "chart requiring visual value reading",
            "qa_source": "chartqa-human"
        }
        entries.append(entry)

        print(f"  [{i}/{num_samples}] {doc_id}: {item['query'][:50]}...")

    return entries


def main():
    print("="*80)
    print("DOWNLOADING DOCVQA & CHARTQA SAMPLES")
    print("="*80)

    # Download samples
    docvqa_entries = download_docvqa_samples(num_samples=15)
    chartqa_entries = download_chartqa_samples(num_samples=15)

    # Load existing hard questions
    with open("data/ground_truth_hard_only.json") as f:
        hard_corpus = json.load(f)

    # Combine
    all_items = hard_corpus['items'] + docvqa_entries + chartqa_entries

    # Create new corpus
    new_corpus = {
        "metadata": {
            "seed": 20240831,
            "num_questions": len(all_items),
            "num_documents": len(all_items),  # Mostly 1-page docs
            "description": "Hard layout-aware corpus: FinanceBench (9) + DocVQA (15) + ChartQA (15)",
            "sources": {
                "financebench_hard": 8,
                "doclaynet_hard": 1,
                "docvqa": len(docvqa_entries),
                "chartqa": len(chartqa_entries)
            },
            "notes": [
                "Only includes documents where naive text extraction fails",
                "FinanceBench: Multi-column complex tables",
                "DocVQA: Forms with spatial field-value relationships",
                "ChartQA: Charts requiring visual value reading"
            ]
        },
        "items": all_items
    }

    # Save
    output_path = Path("data/ground_truth_v2.json")
    with open(output_path, "w") as f:
        json.dump(new_corpus, f, indent=2)

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total questions: {len(all_items)}")
    print(f"  - FinanceBench (hard): {hard_corpus['metadata']['num_questions']}")
    print(f"  - DocVQA: {len(docvqa_entries)}")
    print(f"  - ChartQA: {len(chartqa_entries)}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
