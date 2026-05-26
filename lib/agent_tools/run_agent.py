#!/usr/bin/env python3
"""CLI entry point for the dialectical agent tools.

Usage:
    python3 -m lib.agent_tools.run_agent "scan topic 131"
    python3 -m lib.agent_tools.run_agent --mode plan "find missing content in 131 and generate it"
    python3 -m lib.agent_tools.run_agent "lint filenames in /system" --no-dry-run
"""
import argparse
import json
import logging
import os
import sys

from openai import OpenAI

from lib.agent_tools.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, AGENT_DEFAULT_MODE, AGENT_HISTORY_FILE
from lib.agent_tools.registry import ToolRegistry
from lib.agent_tools.orchestrator.agent import Agent
from lib.agent_tools.orchestrator.visibility import VisibilityLogger

# Import all tools
from lib.agent_tools.tree_scan import ScanFilesystemTool
from lib.agent_tools.triple_analysis import AnalyseTriplesTool, MissingContentTool
from lib.agent_tools.toc_write import WriteTocToFilesystemTool, SanitizeTocTool
from lib.agent_tools.analogies import FindAnalogiesTool
from lib.agent_tools.prompt_gen import GenerateTocPromptTool, GenerateTextPromptTool
from lib.agent_tools.semantic_search import SemanticSearchTool
from lib.agent_tools.dialectic import DialecticAnalyseTool
from lib.agent_tools.lint_filenames import LintFilenamesTool
from lib.agent_tools.compare_contents import CompareContentsTool
from lib.agent_tools.search_concept import SearchConceptTool
from lib.agent_tools.rename_topic import RenameTopicTool
from lib.agent_tools.move_topic import MoveTopicTool
from lib.agent_tools.merge_topics import MergeTopicsTool
from lib.agent_tools.write_topic_content import WriteTopicContentTool
from lib.agent_tools.park_topic import ParkTopicTool, InsertTopicTool, ListParkedTool


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for tool_cls in [
        ScanFilesystemTool,
        AnalyseTriplesTool,
        MissingContentTool,
        WriteTocToFilesystemTool,
        SanitizeTocTool,
        FindAnalogiesTool,
        GenerateTocPromptTool,
        GenerateTextPromptTool,
        SemanticSearchTool,
        DialecticAnalyseTool,
        LintFilenamesTool,
        CompareContentsTool,
        SearchConceptTool,
        RenameTopicTool,
        MoveTopicTool,
        MergeTopicsTool,
        WriteTopicContentTool,
        ParkTopicTool,
        InsertTopicTool,
        ListParkedTool,
    ]:
        reg.register(tool_cls())
    return reg


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dialectical Content Agent — explore, analyse, and extend the fractal knowledge system"
    )
    parser.add_argument("intent", nargs="?", help="What should the agent do?")
    parser.add_argument("--mode", choices=["dynamic", "plan"], default=AGENT_DEFAULT_MODE)
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--history", default=AGENT_HISTORY_FILE, help="Path for operation history JSONL")
    parser.add_argument("--no-dry-run", action="store_true", help="Actually perform destructive operations (e.g. rename files)")

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Build registry
    registry = build_registry()

    # Set up LLM client
    api_key = LLM_API_KEY or os.environ.get("LLM_API_KEY")
    if not api_key:
        print("Error: LLM_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    llm_client = OpenAI(
        api_key=api_key,
        base_url=LLM_BASE_URL,
    )

    # If intent is from stdin (pipe mode)
    intent = args.intent
    if not intent and not sys.stdin.isatty():
        intent = sys.stdin.read().strip()
    if not intent:
        parser.print_help()
        sys.exit(1)

    # Inject --no-dry-run into intent if user wants real ops
    if args.no_dry_run:
        intent = intent + " (perform actual changes, not dry run)"

    visibility = VisibilityLogger(history_file=args.history)

    agent = Agent(
        registry=registry,
        llm_client=llm_client,
        llm_model=args.model,
        visibility=visibility,
        mode=args.mode,
    )

    result = agent.run(intent)

    # Print result
    print("\n=== Result ===")
    print(result.get("result", "No result"))

    print(f"\n=== Steps: {result.get('steps', 0)} ===")
    for h in result.get("history", []):
        status_icon = "✓" if h.get("status") == "success" else "✗"
        print(f"  {status_icon} {h.get('tool_name', '?')} ({h.get('duration_s', '?')}s) — {h.get('result_summary', '')}")

    if args.history:
        print(f"\nHistory logged to: {args.history}")


if __name__ == "__main__":
    main()
