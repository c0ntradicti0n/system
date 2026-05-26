import time

from ._path import _project_root
from .tool_base import BaseTool, ToolResult


class GenerateTocPromptTool(BaseTool):
    name = "generate_toc_prompt"
    description = "Build an LLM prompt for extending the dialectical table of contents at a specific path. Does NOT call the LLM — returns instruction + prompt text."
    parameters = {
        "type": "object",
        "properties": {
            "toc_string": {"type": "string", "description": "Serialized TOC string (from post_process_tree)"},
            "tree_json": {"type": "object", "description": "Serialized tree dict"},
            "path_to_fill": {"type": "string", "description": "Path to fill (e.g. '1332')"},
            "features": {
                "type": "object",
                "description": "Feature toggles: block, shift, topic, antithesis, thesis, synthesis, inversion_antonym",
                "default": {"block": True, "shift": True, "topic": True, "antithesis": True, "thesis": True, "synthesis": True, "inversion_antonym": True},
            },
        },
        "required": ["toc_string", "tree_json", "path_to_fill"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        try:
            from philo.llm_update_toc import llm_update_toc
            features = kwargs.get("features", {})
            instruction, prompt = llm_update_toc(
                kwargs["toc_string"],
                features,
                kwargs["tree_json"],
                path_to_fill=kwargs["path_to_fill"],
            )
            duration = time.time() - t0
            return ToolResult.ok(self.name, {
                "instruction": instruction,
                "prompt": prompt,
                "prompt_length": len(prompt),
            }, params=kwargs, duration_s=duration)
        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)


class GenerateTextPromptTool(BaseTool):
    name = "generate_text_prompt"
    description = "Build an LLM prompt for writing text content for paths missing content. Does NOT call the LLM."
    parameters = {
        "type": "object",
        "properties": {
            "toc_string": {"type": "string", "description": "Serialized TOC string"},
            "tree_json": {"type": "object", "description": "Serialized tree dict"},
            "base_path": {"type": "string", "description": "Filesystem base path"},
            "features": {
                "type": "object",
                "default": {"block": True, "shift": False, "topic": True, "antithesis": True, "thesis": True, "synthesis": True, "inversion_antonym": True},
            },
        },
        "required": ["toc_string", "tree_json", "base_path"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        try:
            from philo.llm_update_text import llm_update_text
            features = kwargs.get("features", {})
            instruction, prompt = llm_update_text(
                kwargs["toc_string"],
                features,
                kwargs["tree_json"],
                kwargs["base_path"],
            )
            duration = time.time() - t0
            return ToolResult.ok(self.name, {
                "instruction": instruction,
                "prompt": prompt,
                "prompt_length": len(prompt),
            }, params=kwargs, duration_s=duration)
        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)
