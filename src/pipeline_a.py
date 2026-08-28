"""Pipeline A — Naive text baseline.

PyPDFLoader → RecursiveCharacterTextSplitter → Gemini text embeddings
             → FAISS → Gemini generation

This is the control arm: no layout awareness. Tables arrive as whitespace-
mangled text, multi-column pages are read in whatever order pypdf emits.

Interface (shared by all three pipelines):
    ingest(pdf_paths, config) -> Index
    query(index, question, k) -> Answer
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load environment
load_dotenv()


@dataclass
class Chunk:
    """One text chunk with metadata."""

    text: str
    doc_id: str  # PDF filename stem
    page_num: int  # 1-based page number within the PDF
    chunk_id: int  # Sequential chunk ID


@dataclass
class Index:
    """Searchable vector index plus chunk metadata."""

    faiss_index: faiss.Index
    chunks: list[Chunk]
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
# Ingestion
# ---------------------------------------------------------------------------


def ingest(pdf_paths: list[Path], config: dict | None = None) -> Index:
    """Build FAISS index from PDFs via naive text extraction.

    Args:
        pdf_paths: List of PDF file paths to index
        config: Optional configuration dict:
            chunk_size: int (default 1000)
            chunk_overlap: int (default 200)
            embed_model: str (default from GEMINI_EMBED_MODEL env)
            use_gpu: bool (default True if available)

    Returns:
        Index with FAISS index, chunks, and metadata
    """
    cfg = config or {}
    chunk_size = cfg.get("chunk_size", 1000)
    chunk_overlap = cfg.get("chunk_overlap", 200)
    embed_model = cfg.get("embed_model") or os.getenv(
        "GEMINI_EMBED_MODEL", "gemini-embedding-2-preview"
    )
    use_gpu = cfg.get("use_gpu", True)

    print(f"[Pipeline A] Ingesting {len(pdf_paths)} PDFs")
    print(f"  chunk_size={chunk_size}, overlap={chunk_overlap}")
    print(f"  embed_model={embed_model}")

    # 1. Load PDFs with PyPDF
    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        # Attach source filename for later attribution
        for doc in docs:
            doc.metadata["source_file"] = pdf_path.stem
        all_docs.extend(docs)
    print(f"  loaded {len(all_docs)} pages from {len(pdf_paths)} PDFs")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(all_docs)
    print(f"  split into {len(split_docs)} chunks")

    # 3. Build Chunk objects with metadata
    chunks: list[Chunk] = []
    for i, doc in enumerate(split_docs):
        chunks.append(
            Chunk(
                text=doc.page_content,
                doc_id=doc.metadata.get("source_file", "unknown"),
                page_num=doc.metadata.get("page", 0) + 1,  # pypdf is 0-based
                chunk_id=i,
            )
        )

    # 4. Batch embed all chunks
    print(f"  embedding {len(chunks)} chunks with {embed_model}...")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Batch embed in groups of 100 (API limit)
    batch_size = 100
    all_embeddings = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]
        response = client.models.embed_content(model=embed_model, contents=texts)
        batch_embeddings = [emb.values for emb in response.embeddings]
        all_embeddings.extend(batch_embeddings)
        if (start // batch_size + 1) % 5 == 0:
            print(f"    embedded {start + len(batch)}/{len(chunks)} chunks")

    embeddings = np.array(all_embeddings, dtype="float32")
    embed_dim = embeddings.shape[1]
    print(f"  embeddings shape: {embeddings.shape} (dim={embed_dim})")

    # 5. Build FAISS index
    # Use Inner Product for cosine similarity (embeddings are normalized)
    index = faiss.IndexFlatIP(embed_dim)

    # Move to GPU if available and requested
    if use_gpu and faiss.get_num_gpus() > 0:
        print("  moving index to GPU...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Normalize embeddings for cosine similarity with inner product
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
    """Retrieve top-k chunks and generate answer.

    Args:
        index: Index from ingest()
        question: User question
        k: Number of chunks to retrieve (default 5)

    Returns:
        Answer with generated text and retrieval provenance
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    llm_model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")

    # 1. Embed query
    q_response = client.models.embed_content(
        model=index.embed_model, contents=[question]
    )
    q_embedding = np.array([q_response.embeddings[0].values], dtype="float32")
    faiss.normalize_L2(q_embedding)

    # 2. Search FAISS
    distances, indices = index.faiss_index.search(q_embedding, k)
    retrieved_chunks = [index.chunks[i] for i in indices[0]]

    # 3. Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[Chunk {i+1} from {chunk.doc_id}, page {chunk.page_num}]\n{chunk.text}"
        )
    context = "\n\n".join(context_parts)

    # 4. Generate answer with Gemini
    prompt = f"""Answer the following question based strictly on the provided context.
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
        print("Usage: python pipeline_a.py <pdf_path> [question]")
        sys.exit(1)

    pdf = Path(sys.argv[1])
    if not pdf.exists():
        print(f"Error: {pdf} not found")
        sys.exit(1)

    print(f"Testing Pipeline A with {pdf.name}\n")
    idx = ingest_one(pdf)
    print(f"\nIndexed {len(idx.chunks)} chunks from {pdf.name}")

    if len(sys.argv) > 2:
        q = " ".join(sys.argv[2:])
        print(f"\nQuery: {q}")
        ans = query(idx, q, k=5)
        print(f"\nRetrieved pages: {ans.retrieved_pages}")
        print(f"\nAnswer:\n{ans.text}")
