"""Debug Pipeline B: Check what context is being sent to LLM."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from pipeline_b import ingest, query

# Load 3M document
pdf_path = Path("data/raw_pdfs/financebench/3M_2018_10K.pdf")
question = "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"

print(f"="*70)
print(f"DEBUGGING PIPELINE B: {pdf_path.name}")
print(f"="*70)
print(f"Question: {question}\n")

# Ingest
print("Ingesting with LlamaParse + MarkdownNodeParser...")
index = ingest([pdf_path], config={"use_gpu_embeddings": True, "use_gpu_faiss": False})

print(f"\nCreated {len(index.chunks)} chunks:")
for i, chunk in enumerate(index.chunks):
    print(f"\n[Chunk {i}] Page {chunk.page_num}, {len(chunk.text)} chars")
    preview = chunk.text[:200].replace("\n", " ")
    print(f"  Preview: {preview}...")

# Query
print(f"\n{'='*70}")
print("QUERYING...")
print(f"{'='*70}")
answer = query(index, question, k=5)

print(f"\nRetrieved {len(answer.retrieved_chunks)} chunks:")
for i, chunk in enumerate(answer.retrieved_chunks):
    print(f"\n[Retrieved {i}] Page {chunk.page_num}")

    # Check if this chunk contains the answer
    if "1,577" in chunk.text or "1577" in chunk.text:
        print("  ✓ CONTAINS THE ANSWER!")

    # Show the full chunk text
    print(f"\nFull chunk text ({len(chunk.text)} chars):")
    print("-" * 70)
    print(chunk.text)
    print("-" * 70)

print(f"\n{'='*70}")
print(f"LLM ANSWER:")
print(f"{'='*70}")
print(answer.text)
