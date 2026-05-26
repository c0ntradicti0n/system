import json
import logging
import time
from typing import Any, Dict, List, Optional

from lib.agent_tools.config import AGENT_MAX_STEPS, AGENT_DEFAULT_MODE
from lib.agent_tools.registry import ToolRegistry
from .visibility import VisibilityLogger
from .state import AgentState

log = logging.getLogger("agent_tools.orchestrator")

SYSTEM_PROMPT = """You are a dialectical content agent operating on a Hegelian fractal knowledge system.

The system stores knowledge as nested triples (thesis=1, antithesis=2, synthesis=3) with antonym relations (_) and topic titles (.).

You have tools to:
- Scan the filesystem tree at different depths
- Analyse triple completeness
- Find missing content
- Compare content for inconsistencies
- Lint and fix filenames
- Generate prompts for LLM content expansion
- Search semantically
- Run ML dialectical analysis
- Find analogies across tree levels

Plan your approach step by step. Call ONE tool at a time. After each result, decide the next step.
When the task is complete or you cannot proceed further, respond with a summary of what was done.

IMPORTANT: Always prefer shallow scans (depth=2) first, then drill deeper where needed."""


class Agent:
    def __init__(
        self,
        registry: ToolRegistry,
        llm_client: Any,
        llm_model: str,
        visibility: Optional[VisibilityLogger] = None,
        mode: str = "dynamic",
        max_steps: int = AGENT_MAX_STEPS,
    ):
        self.registry = registry
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.visibility = visibility or VisibilityLogger()
        self.mode = mode
        self.max_steps = max_steps
        self.state = AgentState()

    def run(self, intent: str) -> Dict[str, Any]:
        self.state.intent = intent
        self.state.step_number = 0

        if self.mode == "plan":
            return self._run_plan_mode(intent)
        return self._run_dynamic_mode(intent)

    def _run_dynamic_mode(self, intent: str) -> Dict[str, Any]:
        """Dynamic loop: LLM decides which tool to call each step."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": intent},
        ]
        tools = self.registry.schema_list(fmt="openai")

        if not tools:
            log.warning("No tools registered — returning LLM response directly")
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
            )
            return self._make_result(response.choices[0].message.content, 0)

        for step in range(self.max_steps):
            self.state.step_number = step + 1

            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
            except Exception as e:
                log.error("LLM call failed: %s", e)
                return self._make_result(f"LLM error at step {step + 1}: {e}", step)

            msg = response.choices[0].message

            if not msg.tool_calls:
                final = msg.content or "Task complete."
                log.info("[%d] LLM final: %s", step + 1, final[:200])
                return self._make_result(final, step + 1)
            # Collect all tool responses, then add assistant msg + all responses once
            messages.append(msg)
            all_responses = []

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                log.info("[%d] LLM calls: %s(%s)", step + 1, tool_name, tool_args)

                op_idx = self.visibility.log_start(tool_name, tool_args)
                t0 = time.time()

                result = self.registry.call(tool_name, **tool_args)

                duration = time.time() - t0
                summary = self._summarize_result(result)

                if result.success:
                    self.visibility.log_success(op_idx, summary, duration)
                    self.state.add_result(tool_name, result.data)
                else:
                    self.visibility.log_error(op_idx, result.error or "unknown error", duration)

                all_responses.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({
                        "success": result.success,
                        "data": self._truncate_for_llm(result.data),
                        "error": result.error,
                    }),
                })

            messages.extend(all_responses)

        # Max steps reached
        return self._make_result(
            f"Reached max steps ({self.max_steps}).", self.max_steps
        )

    def _run_plan_mode(self, intent: str) -> Dict[str, Any]:
        """Plan mode: LLM pre-plans all steps, executes sequentially."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nIMPORTANT: First, output a numbered plan of ALL tool calls needed. Then execute each one sequentially. After all steps, summarize results."},
            {"role": "user", "content": intent},
        ]
        tools = self.registry.schema_list(fmt="openai")

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        plan_steps = []
        if msg.content:
            log.info("Plan:\n%s", msg.content)

        if not msg.tool_calls:
            return {"result": msg.content or "No tool calls planned.", "steps": 0}

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            plan_steps.append((tool_name, tool_args, tc.id))

        all_responses = []
        for tool_name, tool_args, tc_id in plan_steps:
            op_idx = self.visibility.log_start(tool_name, tool_args)
            t0 = time.time()
            result = self.registry.call(tool_name, **tool_args)
            duration = time.time() - t0
            summary = self._summarize_result(result)

            if result.success:
                self.visibility.log_success(op_idx, summary, duration)
            else:
                self.visibility.log_error(op_idx, result.error or "unknown error", duration)

            all_responses.append({
                "tool": tool_name,
                "success": result.success,
                "summary": summary,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": json.dumps({
                    "success": result.success,
                    "data": self._truncate_for_llm(result.data),
                }),
            })

        # Get final summary from LLM
        final = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
        )

        return {
            "result": final.choices[0].message.content or "Plan executed.",
            "plan_steps": plan_steps,
            "results": all_responses,
            "steps": len(plan_steps),
            "history": self._history_dicts(),
        }

    def _summarize_result(self, result) -> str:
        if not result.success:
            return f"Error: {result.error}"
        if isinstance(result.data, dict):
            if "violation_count" in result.data:
                return f"{result.data['violation_count']} violations, {result.data['fixed_count']} fixed"
            if "files_created" in result.data:
                return f"Created {result.data['count']} files"
            if "results" in result.data:
                return f"{result.data.get('count', len(result.data['results']))} results"
            if "min_depth_paths" in result.data:
                return f"{len(result.data['min_depth_paths'])} shallow, {len(result.data['max_depth_paths'])} deep incomplete"
            if "prompt_length" in result.data:
                return f"Prompt built ({result.data['prompt_length']} chars)"
            if "inconsistency_count" in result.data:
                return f"{result.data['inconsistency_count']} inconsistencies (score={result.data['similarity_score']})"
            if "analogies" in result.data:
                return f"{len(result.data['analogies'])} analogies found"
            keys = list(result.data.keys())
            return f"keys={keys}"
        return str(result.data)[:120]

    def _make_result(self, result: str, steps: int) -> Dict:
        return {
            "result": result,
            "steps": steps,
            "history": self._history_dicts(),
        }

    def _history_dicts(self) -> List[Dict]:
        return [{
            "tool_name": h.tool_name,
            "status": h.status,
            "result_summary": h.result_summary,
            "duration_s": round(h.duration_s, 2),
        } for h in self.visibility.get_history()]

    def _truncate_for_llm(self, data: Any, max_chars: int = 3000) -> Any:
        """Truncate large data for LLM context window."""
        if data is None:
            return None
        s = json.dumps(data, default=str)
        if len(s) <= max_chars:
            return data
        return {"_truncated": True, "summary": s[:max_chars] + "..."}
