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

investigate this "elaborate a topic workflow" and make agent ops visible. 
