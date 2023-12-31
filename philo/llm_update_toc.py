import os.path
import pathlib

import regex
from compare import weighted_fuzzy_compare
from helper import (get_analogue_from_nested_dict, get_from_nested_dict,
                    unique_by_func)
from philosopher.analyser import analyse_toc
from philosopher.LLMFeature import create_path, features


def llm_update_toc(toc, kwargs, t, new_entries=1):
    if os.path.exists(os.environ.get("MISSING_CHAPTERS", "missing.txt")):
        paths_to_fill = pathlib.Path(
            os.environ.get("MISSING_CHAPTERS", "missing.txt")
        ).read_text()
        paths_to_fill = [p for p in paths_to_fill.split("\n") if p][:new_entries]
    else:
        paths_to_fill = [
            create_path(p)
            for p in analyse_toc(t, exclude=["111", "112"])["min_depth_paths"]
        ][:new_entries]
    paths_to_fill = unique_by_func(paths_to_fill, func=lambda p: p.replace("_", ""))

    themes = [
        (x, get_from_nested_dict(t, list(x), return_on_fail="<please invent>"))
        for x in paths_to_fill
    ]
    themes = [
        (
            k,
            (
                v
                if isinstance(v, str)
                else (", ".join(v.values()).replace(".md", ""))
                if v is not None
                else ""
            ),
        )
        for k, v in themes
    ]
    topics = [f"{k}, {v}," for k, v in themes if not k.endswith("_")]

    antonym_relations = "\n".join([f"{k}, {v}," for k, v in themes if k.endswith("_")])

    analogs = unique_by_func(
        [
            (l, a)
            for k, v in themes
            for l, a in get_analogue_from_nested_dict(t, list(k))
            if a
        ],
        lambda x: str(x[0]),
    )
    analogies = "\n".join("".join(l) + f" {a}" for l, a in analogs).replace(".md", "")

    instruction = (
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

There is also this constraint: 
The fractal lives from semantic analogies, every subtopic from [1-3]*1 should mirror the topics in [1-3]*2 by 
ANALOGY (as far as possible). Please respect this, therefore you are given example other entries, that might correspond.

Respect, that it moves from the most abstract to the most concrete concepts when moving deeper into the nesting 
and horizontally it moves by the defining other concepts by first treating the ingredients and then the complexities 
of those. Another Note on categorization: The fractal has to conquer the realm of overlapping categories. Feelings might be
 "happy" and "sad" as bigger categories and "joyful" and "depressed" might appear as sub-categories or 
 rather side-categories, this is out preferred approach.
 This means instead of 
 
11 happy
111 joyful
12 sad
121 depressed

we would like to organize it like this:

111 happy
112 sad
121 joyful
122 depressed

So the most basic category opens the field and gets mirrored by the other categories and dividing up all opposites 
inside a nesting.

Also respect, that we handle the concepts as "ideals" and daily politics and doubts are not the topic here, if something
needs to be changed, it should be changed, there must not be a hint, that something might not be right.

You'll work with a representation reflecting your current progress. Each unit is addressed via a key as "13221." 
for the thesis in "1322".
Your goal is enrich the thematic structure of the fractal triples by diving deeper by creating 
new triples.

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

Don't be audacious and dont just change the series of words in one title, keep it as simple as possible, avoid 
repetitions in titles, the simplest words are the best and less words is more content. Be enormously precise 
with your answers and the keys, every wrong path causes chaos and will kill the whole project.
Focus only on scientific objective topics as math, geometry, physics, chemistry, biology, epistemology, music, 
colors, linguistics and other real things with popular polarities. Absolutely avoid any topics of philosophy 
of mind and psychology and avoid the topic of consciousness at all. Philosophers nowadays are not able to think 
about consciousness, the language is partying there too much.
Focus on a top-down approach to get to some more systematic dialectical structure; first all upper levels, then the lower levels.
Focus on completeness of the fractal, please fill up all incomplete triples, rather add new triples than improving existing ones.


"""
        + (
            ""
            if not antonym_relations
            else f"""
Plase provide something that causes the conflict in
{antonym_relations}
"""
        )
        + """
And please dump all your knowledge accurately about """
        + " and ".join(topics)
    )

    toc_lines = filter_similar_paths(paths_to_fill, toc)

    toc = "\n".join(toc_lines)
    if not toc:
        raise Exception("No toc lines found")

    prompt = (
        f"""
Here is a truncated version of the current dialectical system to the path, where you should operate on:

{toc}


"""
        + (
            ""
            if not antonym_relations
            else f"""
Plase provide something that causes the conflict in
{antonym_relations}
"""
        )
        + """
And, please dump all you wisdom as a table of contents (only provide titles for chapters) by using our number format accurately and deeply nested about 
 \n"""
        + "\n".join(topics)
        + (
            (
                """


Respect that it should work on analogies resembling the following topics"""
            )
            + f"""

{analogies}
    """
            if analogies
            else ""
        )
        + """\nand don't output any other commentary, the code will extract the titles from your output.
  
Can you please stay on the ground and first develop the topics with quite normal knowledge? As motion is a kind combination of space and time.
Another good hint to the structure, that MUST be observed, is: 1 - is the general, 2 - is the particular, 3 - is the individual as their combination.
"""
    )
    return instruction, prompt


def filter_similar_paths(paths_to_fill, toc, target_count=100, precision=0.05):
    lower_bound = 0.0
    upper_bound = 1.0
    threshold = 0.6  # starting threshold

    while True:
        toc_lines = []

        for l in toc.split("\n"):
            lp = regex.match(r"^\d*", l).group(0)
            if len(lp) < 3:
                if l not in toc_lines:
                    toc_lines.append(l)
            for pf in paths_to_fill:
                if weighted_fuzzy_compare(lp, pf, threshold)[0]:
                    if l not in toc_lines:
                        toc_lines.append(l)
                    break

        count = len(toc_lines)

        if (
            abs(count - target_count) <= precision * target_count
        ):  # Stop if close enough
            return toc_lines

        if count > target_count:
            # Too many lines. Increase threshold.
            new_threshold = threshold + (upper_bound - threshold) / 3
            lower_bound = threshold
        else:
            # Too few lines. Decrease threshold.
            new_threshold = threshold - (threshold - lower_bound) / 3
            upper_bound = threshold

        if (
            abs(new_threshold - threshold) < 0.0000001
        ):  # Prevent infinite loop by having a minimum difference
            return toc_lines

        threshold = new_threshold
