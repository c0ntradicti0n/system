for ../dialectics the microservice framework in ./ has some outdated mcp like context aggregation to make or refactor more content for our "dialectical triangle fractal". Actually we failed then to make the triangle systematic as it should be.

We imagine a kind of agentic workflow, where an agent can:

1. Find inconsistencies by comparing contents
2. Derive fixes for the inconsistencies
3. Lint changes (so we want to have something that autofixes bad filenames containing chars like "'".)

We want this to have a mcp like tooling in an agent workflow.

It should still use our commands like:

    t = tree(
        basepath=config.system_path,
        startpath=location_path,
        sparse=True,
        info_radius=3,
        location="",
        pre_set_output_level=OutputLevel.FILENAMES,
        exclude=[".git", ".git.md", ".idea"],
        prefix_items=True,
        depth=100 if task == "text" else 2,
    )

But be more flexible in the way it looks around in multistep actions.

## Done

Built agent ops layer at `lib/agent_tools/`:

### Infrastructure
- `tool_base.py` — `ToolResult` + `BaseTool` protocol
- `registry.py` — `ToolRegistry` with schema generation
- `orchestrator/visibility.py` — `OperationLog` + file logging
- `orchestrator/state.py` — `AgentState` for accumulated context
- `orchestrator/agent.py` — `Agent` class with dynamic + plan modes
- `run_agent.py` — CLI entry point

### 13 Tools

| Tool | Wraps | Purpose |
|------|-------|---------|
| `scan_filesystem` | `lib.helper.tree()` | Multi-depth tree exploration with output levels |
| `analyse_triples` | `philo.analyser.analyse_toc()` | Find incomplete triples |
| `find_missing_content` | `philo.analyser.without_text()` | Find paths without text |
| `write_toc_to_filesystem` | `lib.helper.nested_dict_to_filesystem()` | Write tree → .md files |
| `sanitize_toc` | `lib.helper.sanitize_nested_dict()` | Fix key formatting |
| `find_analogies` | `lib.helper.get_analogue_from_nested_dict()` | Cross-level analogy mapping |
| `generate_toc_prompt` | `philo.llm_update_toc()` | Build TOC expansion prompt |
| `generate_text_prompt` | `philo.llm_update_text()` | Build text content prompt |
| `semantic_search` | `service.main.search()` | ChromaDB vector search |
| `dialectic_analyse` | `integrator.main.make_dialectics()` | ML dialectical analysis |
| `lint_filenames` | NEW | Bad char detection + autofix |
| `compare_contents` | NEW | Similarity + structural comparison |
| `search_concept` | NEW | Find concept across tree, group by branch, flag scattering |

### Execution Tools (Phase 6 — act on recommendations)

| Tool | Wraps | Purpose |
|------|-------|---------|
| `rename_topic` | NEW | Rename `.Topic.md` file inside directory (e.g. .Music.md → .Music-as-Discipline.md) |
| `move_topic` | NEW | Move entire triple directory to new tree location |
| `merge_topics` | NEW | Merge two topic dirs, preserve content, remove duplicate |

- All support `dry_run=True` for safety, `dry_run=False` with `git_auto_commit`
- Rename leaves antonym file untouched (own distinct title)
- Move fails if target exists — suggests merge instead
- Merge recursively copies missing children, appends antonym content, `shutil.rmtree` remove_path
- All registered in `run_agent.py`

### Verified
- Agent runs autonomous multi-step workflows (19 steps max tested)
- Each tool call logged with params, status, duration via visibility layer
- `search_concept` found "music" in 3 unrelated branches, agent proposed consolidation
- Linter correctly catches `,` in filenames, skips valid triple keys
- Compare tool detects structural asymmetry, content similarity
- Full pipeline: scan → search → compare → analyse → lint → report

### Usage
```bash
python3 -m lib.agent_tools.run_agent "scan topic 1/1 at depth 2 in /dialectics"
python3 -m lib.agent_tools.run_agent "search music in /dialectics and report scattering"
python3 -m lib.agent_tools.run_agent "compare 1/1/1 and 1/1/2 in /dialectics"
python3 -m lib.agent_tools.run_agent "lint filenames in /dialectics" --no-dry-run
```
