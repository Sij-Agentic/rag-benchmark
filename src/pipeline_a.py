"""Pipeline A -- naive text baseline.

    PyPDFLoader -> RecursiveCharacterTextSplitter -> Gemini text embeddings
                -> FAISS -> Gemini generation

This is the control arm: no layout awareness at all. Tables arrive as
whitespace-mangled text and multi-column pages are read in whatever order
pypdf emits, which is exactly the failure mode Pipelines B and C aim to fix.

NOT YET IMPLEMENTED -- scaffold only. Planned interface, shared by all three
pipelines so `evaluate.py` can treat them interchangeably:

    def ingest(pdf_paths: list[Path]) -> Index
    def query(index: Index, question: str, k: int = 5) -> Answer

`Answer` carries the generated text plus the retrieved page numbers, so the
harness can score answer correctness and retrieval hit-rate separately.
"""

raise NotImplementedError("Pipeline A is not implemented yet.")
