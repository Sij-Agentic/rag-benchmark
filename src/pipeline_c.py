"""Pipeline C — Simplified visual retrieval with Gemini VLM.

PDF → page images → text embedding for retrieval → Gemini VLM for answer generation

Simplified approach:
- Render each PDF page as an image
- Use text embeddings for retrieval (same as Pipeline A/B)
- BUT: Send retrieved PAGE IMAGES to Gemini VLM (not text)
- This tests: Can VLM extract from visual layout better than text extraction?

Interface (shared with pipeline_a and pipeline_b):
    ingest(pdf_paths, config) -> Index
    query(index, question, k) -> Answer
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from transformers import AutoModel

# Load environment
load_dotenv()


@dataclass
class Chunk:
    """One page image with text for embedding."""

    image_path: Path
    text: str  # Extracted text for embedding
    doc_id: str
    page_num: int
    chunk_id: int


@dataclass
class Index:
    """Searchable vector index plus page images."""

    faiss_index: faiss.Index
    chunks: list[Chunk]
    embed_model: str
    embed_dim: int


@dataclass
class Answer:
    """Query result: VLM-generated answer from page images."""

    text: str
    retrieved_pages: list[int]
    retrieved_chunks: list[Chunk]
    context: str


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_pdf_pages(pdf_path: Path, output_dir: Path, dpi: int = 150) -> list[dict]:
    """Render PDF pages as images using pixelshot."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["pixelshot", str(pdf_path), "--output", str(output_dir), "--dpi", str(dpi)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"pixelshot failed: {result.stderr}")

    # Pixelshot creates: <output>/<pdf_name>.png.tiles/ with tile_NNNN.jpg
    tiles_dir = output_dir / f"{pdf_path.stem}.png.tiles"

    if not tiles_dir.exists():
        raise RuntimeError(f"Tiles directory not found: {tiles_dir}")

    metadata = json.loads((tiles_dir / "tiles.json").read_text())

    pages = []
    for i, tile_name in enumerate(metadata["tiles"]):
        pages.append({
            "image_path": tiles_dir / tile_name,
            "page_num": i + 1
        })

    return pages


def _extract_text_from_pdf(pdf_path: Path) -> dict[int, str]:
    """Extract text from PDF using pypdf (for embedding only)."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    page_texts = {}

    for i, page in enumerate(reader.pages, 1):
        page_texts[i] = page.extract_text()

    return page_texts


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest(pdf_paths: list[Path], config: dict | None = None) -> Index:
    """Build FAISS index using text embeddings, but store page images.

    Strategy: Use text for retrieval (cheap, works), images for generation (VLM).
    """
    cfg = config or {}
    dpi = cfg.get("dpi", 150)
    embed_model = cfg.get("embed_model", "jinaai/jina-embeddings-v5-omni-small")
    use_gpu = cfg.get("use_gpu", True) and torch.cuda.is_available()

    print(f"[Pipeline C] Ingesting {len(pdf_paths)} PDFs")
    print(f"  Render pages → text embedding for retrieval → VLM for generation")
    print(f"  embed_model={embed_model}, dpi={dpi}")

    # 1. Render all pages as images
    cache_dir = Path("data/cache/pixelrag_tiles")
    chunks: list[Chunk] = []

    for pdf_path in pdf_paths:
        print(f"  {pdf_path.stem}: rendering pages...")

        # Render pages
        page_dir = cache_dir / pdf_path.stem
        tiles_subdir = page_dir / f"{pdf_path.stem}.png.tiles"

        if not tiles_subdir.exists() or not (tiles_subdir / "tiles.json").exists():
            pages = _render_pdf_pages(pdf_path, page_dir, dpi=dpi)
        else:
            # Load from cache
            metadata = json.loads((tiles_subdir / "tiles.json").read_text())
            pages = [
                {"image_path": tiles_subdir / t, "page_num": i + 1}
                for i, t in enumerate(metadata["tiles"])
                if (tiles_subdir / t).exists()
            ]
            print(f"    loaded {len(pages)} pages from cache")

        # Extract text for embedding
        page_texts = _extract_text_from_pdf(pdf_path)

        # Create chunks (one per page)
        for page in pages:
            chunks.append(
                Chunk(
                    image_path=page["image_path"],
                    text=page_texts.get(page["page_num"], ""),
                    doc_id=pdf_path.stem,
                    page_num=page["page_num"],
                    chunk_id=len(chunks),
                )
            )

        print(f"    → {len(pages)} pages")

    print(f"  rendered {len(chunks)} pages total")

    # 2. Embed text using Jina
    print(f"  embedding text from {len(chunks)} pages...")

    jina_model = AutoModel.from_pretrained(embed_model, trust_remote_code=True)
    if use_gpu:
        jina_model = jina_model.cuda()

    texts = [c.text for c in chunks]

    with torch.no_grad():
        embeddings_tensor = jina_model.encode(texts, task="retrieval")

    embeddings = embeddings_tensor.float().cpu().numpy()
    embed_dim = embeddings.shape[1]

    print(f"  embeddings shape: {embeddings.shape} (dim={embed_dim})")

    # 3. Build FAISS index (CPU to avoid OOM)
    index = faiss.IndexFlatIP(embed_dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print(f"  indexed {index.ntotal} vectors")

    return Index(
        faiss_index=index,
        chunks=chunks,
        embed_model=embed_model,
        embed_dim=embed_dim,
    )


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def query(index: Index, question: str, k: int = 5) -> Answer:
    """Retrieve pages via text embedding, generate answer with Gemini VLM."""
    llm_model = os.getenv("GEMINI_VLM_MODEL", "gemini-2.5-flash")

    # 1. Embed query as text
    jina_model = AutoModel.from_pretrained(index.embed_model, trust_remote_code=True)
    if torch.cuda.is_available():
        jina_model = jina_model.cuda()

    with torch.no_grad():
        q_tensor = jina_model.encode([question], task="retrieval")
    q_embedding = q_tensor.float().cpu().numpy()

    faiss.normalize_L2(q_embedding)

    # 2. Retrieve pages via FAISS
    distances, indices = index.faiss_index.search(q_embedding, k)
    retrieved_chunks = [index.chunks[i] for i in indices[0]]

    # 3. Generate answer with Gemini VLM using page IMAGES
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt_parts = [
        "Answer the following question based strictly on the provided document page images.",
        "Read the visual content carefully including any tables, charts, or layouts.",
        "If the answer cannot be determined, say 'Cannot determine from provided context.'",
        f"\nQuestion: {question}\n",
        "\nDocument pages:",
    ]

    # Add page images
    for i, chunk in enumerate(retrieved_chunks):
        with open(chunk.image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()

        prompt_parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img_data
            }
        })
        prompt_parts.append(f"\n[Page {i+1}: {chunk.doc_id}, page {chunk.page_num}]\n")

    prompt_parts.append("\nAnswer:")

    response = client.models.generate_content(model=llm_model, contents=prompt_parts)
    answer_text = (response.text or "").strip()

    # 4. Extract page numbers
    retrieved_pages = sorted({c.page_num for c in retrieved_chunks if c.page_num > 0})

    context_desc = "\n".join([
        f"Page {i+1}: {c.doc_id} page {c.page_num} (image: {c.image_path})"
        for i, c in enumerate(retrieved_chunks)
    ])

    return Answer(
        text=answer_text,
        retrieved_pages=retrieved_pages,
        retrieved_chunks=retrieved_chunks,
        context=context_desc,
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def ingest_one(pdf_path: Path, config: dict | None = None) -> Index:
    """Ingest a single PDF."""
    return ingest([pdf_path], config)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline_c_simple.py <pdf_path> [question]")
        sys.exit(1)

    pdf = Path(sys.argv[1])
    if not pdf.exists():
        print(f"Error: {pdf} not found")
        sys.exit(1)

    print(f"Testing Pipeline C with {pdf.name}\n")
    idx = ingest_one(pdf)
    print(f"\nIndexed {len(idx.chunks)} pages")

    if len(sys.argv) > 2:
        q = " ".join(sys.argv[2:])
        print(f"\nQuery: {q}")
        ans = query(idx, q, k=5)
        print(f"\nRetrieved pages: {ans.retrieved_pages}")
        print(f"\nAnswer:\n{ans.text}")
