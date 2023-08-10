import logging
import os

import regex as re
from helper import get_from_nested_dict

from philosopher.analyser import without_text
from philosopher.missing import add_to_missing_in_toc


def llm_update_text(toc, kwargs, t, base_path):
    for where, (path_before, text_before) in without_text(t, base_path, exclude=[]):
        try:
            themes = [(x, get_from_nested_dict(t, x)) for x in where]
            themes = [(p, x["."]) if isinstance(x, dict) else (p, x) for p, x in themes]
        except:
            add_to_missing_in_toc(t, where)
            continue

        try:
            themes = [("".join(str(pp) for pp in p), x) for p, x in themes]
            themes = [(x, y.replace(".md", "")) for (x, y) in themes]
        except:
            print(themes, where)
            logging.error("Error updating hegelian text.", exc_info=True)
            continue

        if themes:
            break
    topics =  " * " + "\n * ".join(
        f"{p} {t}" for p, t in themes
    )
    instruction = """
You are prociding the tex for ONE level of a a dialectical system, emulating Hegel's methodology, where concepts unfold within a fractal structure of triples. Each triple consists of:

 - Thesis (1)
 - Antithesis (2)
 - Synthesis (3)

Each triple is a dialectical unit, which can be further extended by adding new triples to any of them.

Additionally we mention the antinomic mutual relation about each triple. This is a pivot concept that amplifies the evolution of the argumentation by identifying the conflict within the synthesis, that is semantically contructed from a relation of thesis and antithesis, as the "vanishing of the vanishing itself" in the example below. But don't repeat that or use it as a reference.

You'll provided a table of contents and a key, as well es text before and after, if there is, and you'll create text content for it for one chapter.

Each chapter is addressed via a key as "13221", meaning the "thesis" in "1322".
And will answer with a list of keys and suggestions for the content of the chapters. Use exactly the following syntax:

```
# 1321 
your text for the thesis

# 1322 
your text for the antithesis

# 1323 
your text for the synthesis

# 132_ 
your text for the antinomic mutual relation of this dialectical unit
```

This is the basic level of the whole system.
Avoid any other explanatory phrase. Your texts can have markdown-syntax or mermaid-graphs for anything, but this should not be needed much.
Keep mind to explain to the point their dialectical relationship, what the concept means under the hood and how it relates to the other near concepts in the toc.


A very neat example is the first chapter, take this as a inspiration for your work:

```
# 1111 Pure Being
Being is absolute indeterminateness and emptiness, without any content to be distinguished within or from anything else, existing only in the purity of its self-equality. However, this pure being, in its indeterminate immediacy, equates to nothing, for it contains no distinguishing content, rendering it equal to pure nothingness in its undifferentiated absence of determination.

# 1112 Pure Nothing
Nothing, though it signifies absence, has a semantic existence in our cognition as the counterpart to being, hence nothing and being, while appearing antithetical, are in essence the same. Nonetheless, they are distinct in their immediacy, with each state immediately vanishing into its opposite, giving rise to the concept of becoming, a constant fluid transition between being and nothing.

# 1113 Becoming
Becoming, the unseparated unity of being and nothing, encapsulates these states as vanishing moments, constantly in flux, expressing itself through two key movements: coming-to-be, where nothing transitions to being, and ceasing-to-be, where being transitions to nothing. This continuous interplay leads to a self-sublation within each state, causing the "vanishing of the vanishing itself", settling into a stable unity that encapsulates the vanished becoming, which is determinate being, a unity of being and nothing in the form of being.

# 111_ 
Vanishing of the vanishing itself
``


Don't wrwite for subtopics, only for the main topics. The subtopics will be filled automatically by the system. So it shoud contain four chapters, four times '#'

Provide the text for ONE level of the text for the following paths and topics:
"""+ topics
    prompt = f"""
        {toc}  
        """
    if text_before:
        prompt += f"""
        the text at the path '{path_before}' before is: {text_before}
        """

    prompt += f"""
Now really dive into writing texts about and only about:
{topics}
And respect our structural requirements, please.
"""
    return instruction, prompt


def custom_parser(text, pattern):
    matches = re.finditer(pattern, text, re.MULTILINE)
    positions = [(match.start(), match.end(), match.group()) for match in matches]

    chunks = []
    start = 0
    header = None

    for pos in positions:
        if header is not None:  # Skip for the first header
            chunks.append(
                (header, text[start : pos[0]])
            )  # Pair the header with the text until the next match
        start = pos[1]
        header = pos[2]  # This becomes the next header

    # If there's text after the last match
    if start < len(text):
        chunks.append((header, text[start:]))

    return chunks
