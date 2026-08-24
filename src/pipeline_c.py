"""Pipeline C -- vision retrieval via PixelRAG.

    pixelshot (tile rendering) -> multimodal embeddings
                -> FAISS -> Gemini vision generation

The bet: skip text extraction entirely. Render each page to image tiles, embed
the pixels, retrieve tiles, and hand the images straight to a VLM -- so layout
is never lost in a text conversion step.

NOT YET IMPLEMENTED -- scaffold only. Same interface as `pipeline_a`:

    def ingest(pdf_paths: list[Path]) -> Index
    def query(index: Index, question: str, k: int = 5) -> Answer

Two things to settle before implementing:

1. Rendering. `pixelshot <pdf> --output <dir> --dpi 200` rasterises PDFs with
   PyMuPDF; the `--backend {cdp,playwright}` flag only affects URL/HTML inputs.
   So for a PDF corpus Chromium is never invoked. Tiles are tall JPEG strips
   (`--tile-height`, default 8192px), NOT one-image-per-page, so tile->page
   attribution has to be tracked to score retrieval hit-rate per page.

2. Embeddings. There is no image-embedding endpoint in the Gemini Developer
   API (`gemini-embedding-001` is text-only). The options are:
     a. PixelRAG's own default, `Qwen/Qwen3-VL-Embedding-2B`, run locally --
        needs `pip install 'pixelrag[embed]'` (torch + transformers, ~3GB) and
        fits the T4 in fp16. Keeps the pipeline self-contained and is the
        configuration PixelRAG is designed around.
     b. Vertex AI `multimodalembedding@001` -- an API call, but a different
        credential path (GCP project + service account, not GEMINI_API_KEY).
   (a) is the recommended default; (b) only if local weights are unacceptable.
"""

raise NotImplementedError("Pipeline C is not implemented yet.")
