import time

from .tool_base import BaseTool, ToolResult


class DialecticAnalyseTool(BaseTool):
    name = "dialectic_analyse"
    description = "Run ML dialectical analysis on texts: classify hierarchy, antithesis, and synthesis relations. Returns triangle subgraph."
    parameters = {
        "type": "object",
        "properties": {
            "texts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of text strings to analyse",
            },
            "epochs": {"type": "integer", "description": "Number of analysis epochs", "default": 10},
            "hash_id": {"type": "string", "description": "Optional session hash for state persistence"},
            "quick_mode": {"type": "boolean", "description": "Run only 1-2 epochs for faster results", "default": False},
        },
        "required": ["texts"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        try:
            from integrator.main import make_dialectics
            from integrator.serialize import serialize_graph_to_structure

            texts = kwargs["texts"]
            epochs = 2 if kwargs.get("quick_mode") else kwargs.get("epochs", 10)

            T = make_dialectics(texts, epochs=epochs, hash_id=kwargs.get("hash_id"))

            graph = T.max_score_triangle_subgraph(
                T.graph, return_start_node=True, start_with_sub=False
            )
            serialized = serialize_graph_to_structure(*graph)

            duration = time.time() - t0
            return ToolResult.ok(self.name, {
                "graph": serialized,
                "progress": T.progress(),
                "epochs_completed": epochs,
            }, params=kwargs, duration_s=duration)

        except ModuleNotFoundError as e:
            return ToolResult.fail(self.name, f"ML models not available: {e}. Run from within Docker container with classifier dependencies.", params=kwargs, duration_s=time.time() - t0)
        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)
