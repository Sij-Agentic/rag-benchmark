"""Pipeline B -- extraction via LlamaParse.

    LlamaParse(result_type="markdown") -> MarkdownNodeParser
                -> Gemini text embeddings -> FAISS -> Gemini generation

The bet: converting layout to *structured markdown* before chunking preserves
table rows and column boundaries that Pipeline A destroys, and
`MarkdownNodeParser` then splits on semantic headings instead of blind
character counts.

NOT YET IMPLEMENTED -- scaffold only. Same interface as `pipeline_a`:

    def ingest(pdf_paths: list[Path]) -> Index
    def query(index: Index, question: str, k: int = 5) -> Answer

Note: LlamaParse is a paid network API. Cache parsed markdown to
`data/cache/llamaparse/` so re-running the harness does not re-bill every page.
"""

raise NotImplementedError("Pipeline B is not implemented yet.")
