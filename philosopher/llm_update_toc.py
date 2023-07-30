from philosopher.analyser import analyse_toc
from philosopher.LLMFeature import create_path, features


def llm_update_toc(index, kwargs, t):
    shortest_path = [
        create_path(p) for p in analyse_toc(t, exclude=["111"])["min_depth_paths"]
    ]
    instruction = (
        """
You are extending a dialectical system, emulating Hegel's methodology, where concepts unfold within a 
fractal structure of triples. Each triple consists of:

 - Thesis (1)
 - Antithesis (2)
 - Synthesis (3)

Each triple is a dialectical unit, which can be further extended by adding a new triple to any of them.

Additionally we mention about each triple the Inversive Dialectical Antonym. 
This is a pivot concept that amplifies the evolution of the argumentation by identifying the conflict within the 
dialectic triple, thereby triggering the formation of the next triple. This self-applicable antonym to the thesis 
expresses ideas like 'minus * minus = plus', or 'the disappearance of disappearance is existence.') (_)

You'll work with a representation reflecting your current progress. Each unit is addressed via a key as "13221." 
for the thesis in "1322".
Your goal is to enhance and enrich the thematic structure of the fractal triples and also diving deeper by creating 
new triples.

Detect paths where it lacks cohesion or displays incorrect structure. Propose improvements by modifying or introducing 
new elements. Use exactly the following syntax:

Every nesting level should have a '.'-entry for telling the theme and also a '_'-entry to mark dialectical conceptual 
movement. 

Examples:

"""
        + "\n\n".join([i for f in features if (i := f(**kwargs).instruction())])
        + """

Stick exactly to this syntax, don't exchange: '.', '_' with more meaningful information, put the meaning only into 
the strings.
Respond only with the keys (123.) and succinct suggestions. Avoid any other explanatory phrase. Your proposals should 
be limited to 1-4 word titles, no sentences.
Thus your output should be a list with a syntax like this:

"""
        + "\n".join([e for f in features if (e := f(**kwargs).example())])
        + """

Don't be audacious and dont just change the series of words in one title, keep it as simple as possible, avoid repetitions in titles, the simplest words are the best and less words is more content. Be enormously precise with your answers and the keys, every wrong path causes chaos and will kill the whole project.
Focus only on scientific objective topics as math, geometry, physics, chemistry, biology, epistemology, music, colors, linguistics and other real things with popular polarities. Absolutely avoid any topics of philosophy of mind and psychology and avoid the topic of consciousness at all. Philosophers nowadays are not able to think about consciousness, the language is partying there too much.
Focus on a top-down approach to get to some more systematic dialectical structure; first all upper levels, then the lower levels.
Focus on completeness of the fractal, please fill up all incomplete triples, rather add new triples than improving existing ones.

And please dive deeper into """
        + " and ".join(shortest_path)
    )
    prompt = f"""
        {index}
        """
    return instruction, prompt
