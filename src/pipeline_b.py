"""Pipeline B — LlamaParse markdown extraction.

LlamaParse(result_type="markdown") → MarkdownNodeParser
    → Gemini text embeddings → FAISS → Gemini generation

The bet: Converting layout to *structured markdown* before chunking preserves
table rows and column boundaries that Pipeline A destroys. MarkdownNodeParser
splits on semantic structure (headers, tables) instead of blind character counts.

Interface (shared with pipeline_a):
    ingest(pdf_paths, config) -> Index
    query(index, question, k) -> Answer
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_parse import LlamaParse
from transformers import AutoModel

# Load environment
load_dotenv()

# Embedding backends
EmbedBackend = Literal["gemini", "jina"]


@dataclass
class Chunk:
    """One text chunk with metadata."""

    text: str
    doc_id: str  # PDF filename stem
    page_num: int  # 1-based page number (approximated from markdown)
    chunk_id: int  # Sequential chunk ID


@dataclass
class Index:
    """Searchable vector index plus chunk metadata."""

    faiss_index: faiss.Index
    chunks: list[Chunk]
    embed_backend: EmbedBackend
    embed_model: str
    embed_dim: int


@dataclass
class Answer:
    """Query result: generated answer + retrieval provenance."""

    text: str
    retrieved_pages: list[int]  # 1-based page numbers
    retrieved_chunks: list[Chunk]
    context: str  # concatenated context sent to LLM


# ---------------------------------------------------------------------------
# Caching layer for LlamaParse (avoid re-billing)
# ---------------------------------------------------------------------------


def _cache_key(pdf_path: Path, result_type: str = "markdown") -> str:
    """Generate cache key from PDF path and parse settings."""
    # Hash: filename + file size + mtime + result_type
    stat = pdf_path.stat()
    key_str = f"{pdf_path.name}_{stat.st_size}_{stat.st_mtime}_{result_type}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache_path(pdf_path: Path, cache_dir: Path) -> Path:
    """Get cache file path for parsed result."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(pdf_path)
    return cache_dir / f"{pdf_path.stem}_{key}.json"


def _load_from_cache(pdf_path: Path, cache_dir: Path) -> list[Document] | None:
    """Load cached LlamaParse result if available."""
    cache_path = _get_cache_path(pdf_path, cache_dir)
    if not cache_path.exists():
        return None

    try:
        data = json.loads(cache_path.read_text())
        docs = [Document(text=d["text"], metadata=d["metadata"]) for d in data]
        return docs
    except Exception:
        return None


def _save_to_cache(
    pdf_path: Path, cache_dir: Path, docs: list[Document]
) -> None:
    """Save LlamaParse result to cache."""
    cache_path = _get_cache_path(pdf_path, cache_dir)
    data = [{"text": doc.text, "metadata": doc.metadata} for doc in docs]
    cache_path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest(pdf_paths: list[Path], config: dict | None = None) -> Index:
    """Build FAISS index from PDFs via LlamaParse markdown extraction.

    Args:
        pdf_paths: List of PDF file paths to index
        config: Optional configuration dict:
            embed_backend: "gemini" | "jina" (default "jina")
            embed_model: str (backend-specific model name)
            use_gpu: bool (default True if available)
            cache_dir: Path to cache LlamaParse results (default: data/cache/llamaparse)

    Returns:
        Index with FAISS index, chunks, and metadata
    """
    cfg = config or {}
    embed_backend: EmbedBackend = cfg.get("embed_backend", "jina")
    embed_model = cfg.get("embed_model")
    use_gpu_embeddings = cfg.get("use_gpu_embeddings", True)  # For Jina model
    use_gpu_faiss = cfg.get("use_gpu_faiss", True)  # For FAISS index
    cache_dir = cfg.get("cache_dir") or Path("data/cache/llamaparse")

    # Set default model per backend
    if not embed_model:
        if embed_backend == "gemini":
            embed_model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2-preview")
        else:  # jina
            embed_model = "jinaai/jina-embeddings-v5-omni-small"

    print(f"[Pipeline B] Ingesting {len(pdf_paths)} PDFs")
    print(f"  LlamaParse → markdown → MarkdownNodeParser")
    print(f"  embed_backend={embed_backend}, model={embed_model}")

    # 1. Parse PDFs to markdown with LlamaParse (cached)
    all_docs: list[Document] = []
    llama_api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not llama_api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY not found in environment")

    parser = LlamaParse(
        api_key=llama_api_key, result_type="markdown", verbose=False
    )

    for pdf_path in pdf_paths:
        # Check cache first
        cached_docs = _load_from_cache(pdf_path, cache_dir)
        if cached_docs:
            print(f"  {pdf_path.stem}: loaded from cache")
            docs = cached_docs
        else:
            print(f"  {pdf_path.stem}: parsing with LlamaParse...")
            docs = parser.load_data(str(pdf_path))
            _save_to_cache(pdf_path, cache_dir, docs)

        # Skip empty documents
        if not docs or all(len(doc.text.strip()) == 0 for doc in docs):
            print(f"  ⚠ {pdf_path.stem}: no text content (image-only PDF?)")
            continue

        # Assign metadata: LlamaParse returns docs in page order, assign 1, 2, 3...
        for i, doc in enumerate(docs, start=1):
            doc.metadata["source_file"] = pdf_path.stem
            doc.metadata["page_num"] = i  # Sequential page number

        all_docs.extend(docs)

    if not all_docs:
        raise ValueError(
            f"No text content found in any of the {len(pdf_paths)} PDFs. "
            "All PDFs appear to be image-only."
        )

    print(f"  parsed {len(all_docs)} document(s) to markdown")

    # 2. Split markdown into chunks with MarkdownNodeParser
    # This parser understands markdown structure (headers, tables, lists)
    node_parser = MarkdownNodeParser()
    nodes = node_parser.get_nodes_from_documents(all_docs)
    print(f"  split into {len(nodes)} markdown chunks")

    # 3. Build Chunk objects with metadata (page numbers now inherited from parent docs)
    chunks: list[Chunk] = []
    for i, node in enumerate(nodes):
        # Get page number from parent document metadata (set above)
        page_num = node.metadata.get("page_num", 1)

        # If still missing, try to extract from ref_doc_id (which points back to parent)
        if page_num == 1 and node.ref_doc_id:
            # Find parent doc and copy its page_num
            parent_doc = next((d for d in all_docs if d.id_ == node.ref_doc_id), None)
            if parent_doc and "page_num" in parent_doc.metadata:
                page_num = parent_doc.metadata["page_num"]

        chunks.append(
            Chunk(
                text=node.text,
                doc_id=node.metadata.get("source_file", "unknown"),
                page_num=page_num,
                chunk_id=i,
            )
        )

    # 4. Embed all chunks
    print(f"  embedding {len(chunks)} chunks with {embed_backend}/{embed_model}...")

    if embed_backend == "gemini":
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        all_embeddings = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                time.sleep(0.1)  # Rate limit protection
            response = client.models.embed_content(model=embed_model, contents=chunk.text)
            all_embeddings.append(response.embeddings[0].values)
            if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                print(f"    embedded {i + 1}/{len(chunks)} chunks")
        embeddings = np.array(all_embeddings, dtype="float32")

    elif embed_backend == "jina":
        # Load Jina model on GPU (separate from FAISS GPU setting)
        jina_model = (
            AutoModel.from_pretrained(embed_model, trust_remote_code=True).cuda()
            if torch.cuda.is_available() and use_gpu_embeddings
            else AutoModel.from_pretrained(embed_model, trust_remote_code=True)
        )

        # Batch encode
        texts = [c.text for c in chunks]
        with torch.no_grad():
            embeddings_tensor = jina_model.encode(texts, task="retrieval")
        embeddings = embeddings_tensor.float().cpu().numpy()
        print(f"    embedded {len(chunks)}/{len(chunks)} chunks (GPU batch)")

    else:
        raise ValueError(f"Unknown embed_backend: {embed_backend}")

    embed_dim = embeddings.shape[1]
    print(f"  embeddings shape: {embeddings.shape} (dim={embed_dim})")

    # 5. Build FAISS index
    index = faiss.IndexFlatIP(embed_dim)

    # Move to GPU if available and requested
    if use_gpu_faiss and faiss.get_num_gpus() > 0:
        print("  moving index to GPU...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"  indexed {index.ntotal} vectors")

    return Index(
        faiss_index=index,
        chunks=chunks,
        embed_backend=embed_backend,
        embed_model=embed_model,
        embed_dim=embed_dim,
    )


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def query(index: Index, question: str, k: int = 5) -> Answer:
    """Retrieve top-k chunks and generate answer.

    Args:
        index: Index from ingest()
        question: User question
        k: Number of chunks to retrieve (default 5)

    Returns:
        Answer with generated text and retrieval provenance
    """
    llm_model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")

    # 1. Embed query using same backend as index
    if index.embed_backend == "gemini":
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        q_response = client.models.embed_content(
            model=index.embed_model, contents=[question]
        )
        q_embedding = np.array([q_response.embeddings[0].values], dtype="float32")

    elif index.embed_backend == "jina":
        # Always try GPU for query embedding (small, fast)
        jina_model = (
            AutoModel.from_pretrained(index.embed_model, trust_remote_code=True).cuda()
            if torch.cuda.is_available()
            else AutoModel.from_pretrained(index.embed_model, trust_remote_code=True)
        )
        with torch.no_grad():
            q_tensor = jina_model.encode([question], task="retrieval")
        q_embedding = q_tensor.float().cpu().numpy()

    else:
        raise ValueError(f"Unknown embed_backend: {index.embed_backend}")

    faiss.normalize_L2(q_embedding)

    # 2. Search FAISS
    distances, indices = index.faiss_index.search(q_embedding, k)
    retrieved_chunks = [index.chunks[i] for i in indices[0]]

    # 3. Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[Chunk {i+1} from {chunk.doc_id}, page ~{chunk.page_num}]\n{chunk.text}"
        )
    context = "\n\n".join(context_parts)

    # 4. Generate answer with Gemini
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = f"""Answer the following question based strictly on the provided context.
The context is in markdown format with preserved table structure.
If the answer cannot be determined from the context, say "Cannot determine from provided context."

Context:
{context}

Question: {question}

Answer:"""

    response = client.models.generate_content(model=llm_model, contents=prompt)
    answer_text = (response.text or "").strip()

    # 5. Extract unique page numbers from retrieved chunks
    retrieved_pages = sorted(
        {chunk.page_num for chunk in retrieved_chunks if chunk.page_num > 0}
    )

    return Answer(
        text=answer_text,
        retrieved_pages=retrieved_pages,
        retrieved_chunks=retrieved_chunks,
        context=context,
    )


# ---------------------------------------------------------------------------
# Convenience: single-PDF interface
# ---------------------------------------------------------------------------


def ingest_one(pdf_path: Path, config: dict | None = None) -> Index:
    """Ingest a single PDF."""
    return ingest([pdf_path], config)


# ---------------------------------------------------------------------------
# Main (for testing)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline_b.py <pdf_path> [question]")
        sys.exit(1)

    pdf = Path(sys.argv[1])
    if not pdf.exists():
        print(f"Error: {pdf} not found")
        sys.exit(1)

    print(f"Testing Pipeline B with {pdf.name}\n")
    idx = ingest_one(pdf)
    print(f"\nIndexed {len(idx.chunks)} chunks from {pdf.name}")

    if len(sys.argv) > 2:
        q = " ".join(sys.argv[2:])
        print(f"\nQuery: {q}")
        ans = query(idx, q, k=5)
        print(f"\nRetrieved pages: {ans.retrieved_pages}")
        print(f"\nAnswer:\n{ans.text}")
