import time

from .tool_base import BaseTool, ToolResult


class SemanticSearchTool(BaseTool):
    name = "semantic_search"
    description = "Search the fractal document tree using vector embeddings (ChromaDB). Returns semantically similar content with paths."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query text"},
            "top_k": {"type": "integer", "description": "Max results", "default": 5},
            "filter_path": {"type": "string", "description": "Optional path filter (e.g. '2' to search only topic 2)"},
        },
        "required": ["query"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        try:
            import chromadb
            from lib.vector_store import VectorStore, get_embedding_function

            ef = get_embedding_function()
            client = chromadb.PersistentClient(path="/chroma")
            collection_name = "system"
            existing = {c.name for c in client.list_collections()}

            if collection_name not in existing:
                return ToolResult.ok(self.name, {
                    "error": "Vector collection not found — run service first to index documents",
                    "results": [],
                }, params=kwargs, duration_s=time.time() - t0)

            collection = client.get_collection(name=collection_name, embedding_function=ef)
            vs = VectorStore(collection)

            filter_path = kwargs.get("filter_path", "")
            filter_criteria = None
            if filter_path:
                filter_criteria = {filter_path: {"$eq": filter_path}}

            results = vs.search(
                kwargs["query"],
                k=kwargs.get("top_k", 5),
                search_type="mmr",
                filter=filter_criteria,
            )

            clean = [
                {
                    "content": r["page_content"][:300],
                    "path": r["metadata"].get("path", ""),
                    "file": r["metadata"].get("file_path", ""),
                }
                for r in results
            ]

            duration = time.time() - t0
            return ToolResult.ok(self.name, {"results": clean, "count": len(clean)}, params=kwargs, duration_s=duration)

        except Exception as e:
            return ToolResult.fail(self.name, f"Search failed: {e}", params=kwargs, duration_s=time.time() - t0)
