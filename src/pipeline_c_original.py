"""Pipeline C — PixelRAG vision-based retrieval.

pixelshot (tile rendering) → Qwen vision embeddings → FAISS → Gemini VLM generation

The bet: Skip text extraction entirely. Render PDFs to image tiles, embed the
pixels, retrieve visually similar tiles, and generate answers from images with
a VLM. Layout is preserved because we never convert to text.

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
from PIL import Image

# Load environment
load_dotenv()


@dataclass
class Chunk:
    """One visual tile with metadata."""

    image_path: Path
    doc_id: str  # PDF filename stem
    page_num: int  # 1-based page number
    chunk_id: int


@dataclass
class Index:
    """Searchable vector index plus visual tiles."""

    faiss_index: faiss.Index
    chunks: list[Chunk]
    embed_model: str
    embed_dim: int


@dataclass
class Answer:
    """Query result: VLM-generated answer + retrieval provenance."""

    text: str
    retrieved_pages: list[int]
    retrieved_chunks: list[Chunk]
    context: str  # Description of retrieved tiles


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_pdf_to_tiles(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[dict[str, Any]]:
    """Render PDF to image tiles using pixelshot.

    Returns list of dicts with {image_path, page_num} for each tile.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use pixelshot to render PDF
    # pixelshot outputs to <filename>.png.tiles/ subdirectory with tile_NNNN.jpg files
    cmd = [
        "pixelshot",
        str(pdf_path),
        "--output", str(output_dir),
        "--dpi", str(dpi),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"pixelshot failed: {result.stderr}")

    # Pixelshot creates subdirectory: <output>/<pdf_name>.png.tiles/
    tiles_dir = output_dir / f"{pdf_path.stem}.png.tiles"

    if not tiles_dir.exists():
        raise RuntimeError(f"Pixelshot tiles directory not found: {tiles_dir}")

    # Load metadata from tiles.json
    metadata_path = tiles_dir / "tiles.json"
    if not metadata_path.exists():
        raise RuntimeError(f"tiles.json not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    total_pages = metadata["total_pages"]

    # Collect tiles (one tile per page, named tile_0000.jpg, tile_0001.jpg, etc.)
    tiles = []
    for i, tile_name in enumerate(metadata["tiles"]):
        tile_path = tiles_dir / tile_name
        if tile_path.exists():
            tiles.append({
                "image_path": tile_path,
                "page_num": i + 1  # 1-based page numbers
            })

    return tiles


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest(pdf_paths: list[Path], config: dict | None = None) -> Index:
    """Build FAISS index from PDFs via visual tile embeddings.

    Args:
        pdf_paths: List of PDF file paths to index
        config: Optional configuration dict:
            dpi: Rendering DPI (default 200)
            embed_model: Vision embedding model (default "Qwen/Qwen3-VL-Embedding-2B")
            use_gpu: Whether to use GPU for embeddings (default True)

    Returns:
        Index with FAISS index and visual tiles
    """
    cfg = config or {}
    dpi = cfg.get("dpi", 200)
    embed_model = cfg.get("embed_model", "Qwen/Qwen3-VL-Embedding-2B")
    use_gpu = cfg.get("use_gpu", True) and torch.cuda.is_available()

    print(f"[Pipeline C] Ingesting {len(pdf_paths)} PDFs")
    print(f"  pixelshot → visual tiles (DPI={dpi}) → vision embeddings → FAISS")
    print(f"  embed_model={embed_model}")

    # 1. Render all PDFs to visual tiles
    cache_dir = Path("data/cache/pixelrag_tiles")
    chunks: list[Chunk] = []

    for pdf_path in pdf_paths:
        print(f"  {pdf_path.stem}: rendering...")
        tile_dir = cache_dir / pdf_path.stem

        # Check if already rendered (cache)
        tiles_subdir = tile_dir / f"{pdf_path.stem}.png.tiles"
        if not tiles_subdir.exists() or not (tiles_subdir / "tiles.json").exists():
            tiles = _render_pdf_to_tiles(pdf_path, tile_dir, dpi=dpi)
        else:
            # Load from cache
            metadata_path = tiles_subdir / "tiles.json"
            metadata = json.loads(metadata_path.read_text())
            tiles = [
                {"image_path": tiles_subdir / tile_name, "page_num": i + 1}
                for i, tile_name in enumerate(metadata["tiles"])
                if (tiles_subdir / tile_name).exists()
            ]
            print(f"    loaded {len(tiles)} tiles from cache")

        # Create chunks
        for tile in tiles:
            chunks.append(
                Chunk(
                    image_path=tile["image_path"],
                    doc_id=pdf_path.stem,
                    page_num=tile["page_num"],
                    chunk_id=len(chunks),
                )
            )

        print(f"    → {len(tiles)} tiles")

    print(f"  rendered {len(chunks)} tiles total")

    # 2. Load vision embedding model
    print(f"  loading vision embedding model: {embed_model}...")

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(embed_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(embed_model, trust_remote_code=True)

    if use_gpu:
        model = model.cuda()

    model.eval()

    # 3. Embed all visual tiles
    print(f"  embedding {len(chunks)} tiles...")

    embeddings_list = []
    batch_size = 8  # Process in small batches to avoid OOM

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_images = [Image.open(c.image_path).convert("RGB") for c in batch_chunks]

        # Process images
        inputs = processor(images=batch_images, return_tensors="pt")

        if use_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Get image embeddings (typically from last_hidden_state pooled)
            batch_embeds = outputs.last_hidden_state.mean(dim=1)  # Pool over sequence
            embeddings_list.append(batch_embeds.cpu().numpy())

        if (i + batch_size) % 40 == 0 or i + batch_size >= len(chunks):
            print(f"    embedded {min(i + batch_size, len(chunks))}/{len(chunks)} tiles")

    embeddings = np.vstack(embeddings_list).astype("float32")
    embed_dim = embeddings.shape[1]

    print(f"  embeddings shape: {embeddings.shape} (dim={embed_dim})")

    # 4. Build FAISS index (CPU only, to avoid GPU memory conflicts)
    index = faiss.IndexFlatIP(embed_dim)

    # Normalize for cosine similarity
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
    """Retrieve top-k visual tiles and generate answer with Gemini VLM.

    Args:
        index: Index from ingest()
        question: User question
        k: Number of tiles to retrieve

    Returns:
        Answer with VLM-generated text and provenance
    """
    llm_model = os.getenv("GEMINI_VLM_MODEL", "gemini-2.5-flash")

    # 1. Embed query text using same vision model
    # (Qwen3-VL can embed both images and text)
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(index.embed_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(index.embed_model, trust_remote_code=True)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    # Embed question as text
    inputs = processor(text=[question], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        q_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype("float32")

    faiss.normalize_L2(q_embedding)

    # 2. Search FAISS
    distances, indices = index.faiss_index.search(q_embedding, k)
    retrieved_chunks = [index.chunks[i] for i in indices[0]]

    # 3. Generate answer with Gemini VLM
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Build multimodal prompt with tile images
    prompt_parts = [
        "Answer the following question based strictly on the provided document images (visual tiles).",
        "The tiles show portions of PDF pages. Read the visual content carefully including any tables, charts, or layouts.",
        "If the answer cannot be determined from the visual context, say 'Cannot determine from provided context.'",
        f"\nQuestion: {question}\n",
        "\nDocument tiles:",
    ]

    # Add tile images
    for i, chunk in enumerate(retrieved_chunks):
        # Read and encode image
        with open(chunk.image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()

        prompt_parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": img_data
            }
        })
        prompt_parts.append(f"\n[Tile {i+1}: {chunk.doc_id}, page {chunk.page_num}]\n")

    prompt_parts.append("\nAnswer:")

    response = client.models.generate_content(model=llm_model, contents=prompt_parts)
    answer_text = (response.text or "").strip()

    # 4. Extract page numbers
    retrieved_pages = sorted({c.page_num for c in retrieved_chunks if c.page_num > 0})

    # Build context description
    context_desc = "\n".join([
        f"Tile {i+1}: {c.doc_id} page {c.page_num} ({c.image_path})"
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
        print("Usage: python pipeline_c.py <pdf_path> [question]")
        sys.exit(1)

    pdf = Path(sys.argv[1])
    if not pdf.exists():
        print(f"Error: {pdf} not found")
        sys.exit(1)

    print(f"Testing Pipeline C with {pdf.name}\n")
    idx = ingest_one(pdf)
    print(f"\nIndexed {len(idx.chunks)} visual tiles")

    if len(sys.argv) > 2:
        q = " ".join(sys.argv[2:])
        print(f"\nQuery: {q}")
        ans = query(idx, q, k=5)
        print(f"\nRetrieved pages: {ans.retrieved_pages}")
        print(f"\nAnswer:\n{ans.text}")
