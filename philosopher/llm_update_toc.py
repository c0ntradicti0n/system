import os.path
import pathlib

import editdistance as editdistance
import regex
from helper import get_from_nested_dict, get_analogue_from_nested_dict, unique_by_func

from philosopher.analyser import analyse_toc
from philosopher.LLMFeature import create_path, features


def llm_update_toc(toc, kwargs, t):
    if os.path.exists(os.environ.get("MISSING_CHAPTERS", "missing.txt")):
        paths_to_fill = pathlib.Path(
            os.environ.get("MISSING_CHAPTERS", "missing.txt")
        ).read_text()
        paths_to_fill = [(p, "") for p in paths_to_fill.split("\n") if p][:5]
    else:
        paths_to_fill = [
            create_path(p) for p in analyse_toc(t, exclude=["111"])["min_depth_paths"]
        ][:3]
    paths_to_fill = unique_by_func(paths_to_fill, func=lambda p: p.replace("_", ""))

    themes = [
        (x, get_from_nested_dict(t, list(x), return_on_fail="<please invent>"))
        for x in paths_to_fill
    ]
    themes = [
        (k, (v if isinstance(v, str) else
        (", ".join(v.values()).replace(".md", "")) if v is not None else ""))
        for k, v in themes
    ]
    topics = [f"{k}, {v}," for k, v in themes]
    analogies = "\n".join(
        ''.join(l) + f" {a}" for k, v in themes for l, a in get_analogue_from_nested_dict(t, list(k)) if a
    ).replace(".md", "")
    paths_to_fill = [p for p in paths_to_fill if not p.startswith("11")]

    instruction = (
        (
            """
You are extending a dialectical system, emulating Hegel's methodology, where concepts unfold within a 
fractal structure of triples. Each triple consists of:

 - Thesis (1)
 - Antithesis (2)
 - Synthesis (3)

Each triple is a dialectical unit, which can be further extended by adding a new triple to any of them.

Additionally we explain about each triple a antinomic mutual relation. Remember the Kantian antinomies, and 
other famous intricacies. 
This should point to  the evolution of the argumentation by identifying the conflict within the 
dialectic triple, thereby triggering the formation of the next triple. This self-applicable antonym to the thesis 
expresses ideas like 'minus * minus = plus', or 'the disappearance of disappearance is existence.') (_)

There is also this constraint: The fractal lives from semantic analogies, every subtopic from [1-3]*1 should mirror the topics in [1-3]*2 by ANALOGY (as far as possible) Please respect this, therefore you are given example other entries, that differ in editdistance 2. 

You'll work with a representation reflecting your current progress. Each unit is addressed via a key as "13221." 
for the thesis in "1322".
Your goal is to enhance and enrich the thematic structure of the fractal triples and also diving deeper by creating 
new triples.

Detect paths where it lacks cohesion or displays incorrect structure. Propose improvements by modifying or introducing 
new elements. Use exactly the following syntax:

Every nesting level should have a bare path for telling the theme and also a '_'-entry to mark dialectical conceptual 
movement, every path should match this regex: [1-3]_?

Examples:

"""
            + "\n\n".join([i for f in features if (i := f(**kwargs).instruction())])
            + """

Stick exactly to this syntax, don't exchange: '.', '_' with more meaningful information, put the meaning only into 
the strings.
Respond only with the keys like 31332 and succinct suggestions. Avoid any other explanatory phrase! Your proposals should 
be limited to 1-4 word titles, no sentences.
Thus your output should be a list with a syntax like this:

"""
            + "\n".join([e for f in features if (e := f(**kwargs).example())])
            + """

Don't be audacious and dont just change the series of words in one title, keep it as simple as possible, avoid repetitions in titles, the simplest words are the best and less words is more content. Be enormously precise with your answers and the keys, every wrong path causes chaos and will kill the whole project.
Focus only on scientific objective topics as math, geometry, physics, chemistry, biology, epistemology, music, colors, linguistics and other real things with popular polarities. Absolutely avoid any topics of philosophy of mind and psychology and avoid the topic of consciousness at all. Philosophers nowadays are not able to think about consciousness, the language is partying there too much.
Focus on a top-down approach to get to some more systematic dialectical structure; first all upper levels, then the lower levels.
Focus on completeness of the fractal, please fill up all incomplete triples, rather add new triples than improving existing ones.

So, please dump all your knowledge accurately about """
            + " and ".join(topics)
        )
    )

    toc_lines = []
    for l in toc.split("\n"):
        lp = regex.match(r"^\d*", l).group(0)
        for pf in paths_to_fill:
            c = max(len(pf), len(lp))
            depth_score = sum([
                (lp[i] == pf [i])*(c -i) for i in range(c) if i < len(lp) and i < len(pf)
            ])
            min_score = len(pf) *1.5

            if (depth_score > min_score or len(lp) < 2) and len(lp) < len(pf) + 2:
                toc_lines.append(l)
                break

    toc = "\n".join(toc_lines)

    prompt = (
        f"""
Here is a truncated version of the current dialectical system to the path, where you should operate on:

{toc}

So, please dump all you wisdom accurately about \n"""
        + "\n".join(topics)
        + ((
               """


Respect that it should work on analogies resembling the following topics""")
           + f"""

{analogies}
    """
           if analogies else ""
           )
    )
    return instruction, prompt
