import json
import os
import pprint
import re

import config
import guidance
import openai
import simple_cache as simple_cache
from dotenv import load_dotenv
from helper import (OutputLevel, extract, get_prefix,
                    nested_dict_to_filesystem, post_process_tree,
                    sanitize_nested_dict, tree, update_nested_dict)
from regex import regex

from philosopher.analyser import analyse
from philosopher.LLMFeature import create_path, features
from philosopher.yaml import (object_to_yaml_str, rec_sort,
                              single_key_completion)
from server.os_tools import git_auto_commit

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = ""


if os.environ.get("OPENAI_API_KEY"):
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")

    @simple_cache.cache_it(filename=os.getcwd() + "/dialectic_triangle.cache", ttl=120)
    def llm(instruction, text):
        return openai.ChatCompletion.create(
            model=os.environ.get("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
        )

else:
    print(os.environ.get("MODEL", "openlm-research/open_llama_3b_600bt_preview"))
    guidance.llm = guidance.llms.Transformers(
        os.environ.get("MODEL", "openlm-research/open_llama_3b_600bt_preview"),
        trust_remote_code=True,
    )
    # "openlm-research/open_llama_3b_600bt_preview")


def dialectic_triangle(
    base_path,
    location,
    task="index",
    info_radius=2,
    preset_output_level=None,
    **kwargs,
):
    location_path = "/".join(location)

    t = tree(
        basepath=base_path,
        startpath="",
        format="json",
        sparse=True,
        info_radius=info_radius,
        location=location_path,
        pre_set_output_level=preset_output_level,
        exclude=[".git", ".git.md", ".idea"],
        prefix_items=True,
    )
    pprint.pprint(t)
    index = post_process_tree(single_key_completion(object_to_yaml_str(rec_sort(t))))

    if task == "index":
        shortest_path = [
            create_path(p) for p in analyse(t, exclude=["111"])["min_depth_paths"]
        ]

        instruction = (
            """
You are extending a dialectical system, emulating Hegel's methodology, where concepts unfold within a fractal structure of triples. Each triple consists of:

 - Thesis (1)
 - Antithesis (2)
 - Synthesis (3)
 
Each triple is a dialectical unit, which can be further extended by adding new triples to the synthesis.
Their content has to be addressed by a topic given as '.'
 
Additionally we mention about each triple the Inversive Dialectical Antonym (this is a pivot concept that amplifies the evolution of the argumentation by identifying the conflict within the dialectic triple, thereby triggering the formation of the next triple. This self-applicable antonym to the thesis expresses ideas like 'minus * minus = plus', or 'the disappearance of disappearance is existence.') (_)

You'll work with a representation reflecting your current progress. Each unit is addressed via a key as "13221." for the thesis in "1322".
Your goal is to enhance and enrich the thematic structure of the fractal triples and also diving deeper by creating new triples.

Detect paths where it lacks cohesion or displays incorrect structure. Propose improvements by modifying or introducing new elements. Use exactly the following syntax:

Every nesting level should have a '.'-entry for telling the theme and also a '_'-entry to mark dialectical conceptual movement. 

Examples:

"""
            + "\n\n".join([i for f in features if (i := f(**kwargs).instruction())])
            + """
   
Stick exactly to this syntax, don't exchange: '.', '_' with more meaningful information, put the meaning only into the strings.
Respond only with the keys (123.) and succinct suggestions. Avoid any other explanatory phrase. Your proposals should be limited to 1-4 word titles, no sentences.
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
    else:
        raise NotImplementedError(f"{task} not implemented")

    print(prompt)
    print(location)
    print(f"{len(prompt)=} {len(index)=} {len(location)=}")

    lllm_output = ".cache/lllm.output.txt"
    lllm_input = ".cache/lllm.input.txt"

    if not os.path.exists(lllm_output):
        api_result = llm(instruction=instruction.strip(), text=prompt)
        output = api_result["choices"][0]["message"]["content"]
        with open(lllm_input, "w") as f:
            f.write(instruction.strip() + "\n" + prompt + "\n")

        with open(lllm_output, "w") as f:
            f.write(output + "\n")
    else:
        with open(lllm_output) as f:
            output = f.read()

    xpaths = output.split("\n")
    xpaths = [x for x in xpaths if x]
    xpaths = [regex.sub(r"^\d\.", "", x) for x in xpaths]
    print(xpaths)

    if task == "index":
        paths = []
        while not paths:
            for path in xpaths:
                if not path:
                    continue
                m = extract(path)

                if m and m["key"] != "not matched":
                    paths.append(m)
                else:
                    print(f"Could not extract {path}")

            if not paths:
                xpaths = input("No paths found. Try web interface and paste.").split(
                    "\n"
                )

        assert paths, "No paths found. Try web interface and paste."

        for m in paths:
            keys = (
                m["key"]
                .replace("['", "")
                .replace("']", "|")
                .replace("$.", "")
                .replace("topic", ".")
                .replace("antonym", "_")
                .replace("thesis", "1")
                .replace("antithesis", "2")
                .replace("synthesis", "3")
                .replace("/", "-")
            )
            keys = [k for k in keys]
            last_key = keys[-1]
            prefix = get_prefix(last_key)

            if (
                m["key"].startswith("$.['1']['1']")
                or m["key"].startswith("$.['1']['2']")
                or m["key"].startswith("$.['2']['1']")
                or m["key"].startswith("$.['2']['2']")
            ):
                continue

            v = m["value"].replace("/", "-")
            if "{" in v and "}" in v:
                vv = m["value"]
                vv = regex.sub(r"(\d):", '"\\1":', vv)
                v = json.loads(vv)
                if not keys[-1] == ".":
                    keys.append(".")

            update_nested_dict(
                t,
                keys,
                v,
                prefix=prefix,
                change_keys=True,
                sanitize_key=lambda x: x.replace("..", ".").replace("__", "_")
                if isinstance(x, str)
                else x,
            )

        t = sanitize_nested_dict(
            t,
            sanitize_key=lambda x: x.replace("..", ".")
            .replace("__", "_")
            .replace("/", "-")
            if isinstance(x, str)
            else x,
            healthy_last_key=lambda x: re.match(r"^([1-3]|\.|_)", x)
            if isinstance(x, str)
            else x,
            healthy_normal_key=lambda x: re.match(r"^[1-3]$", x) and len(x) == 1
            if isinstance(x, str)
            else x,
            last_key_to_normal_key=lambda x: re.match(r"^([1-3]|\.|_)", x).group(0)
            if isinstance(x, str)
            else x,
        )

        created = nested_dict_to_filesystem(f"{base_path}".strip(), t)
        print(f"Created {created} files.")


if __name__ == "__main__":
    with git_auto_commit(
        config.system_path, commit_message_prefix="Automated Commit"
    ) as ctx:
        print(
            dialectic_triangle(
                base_path=config.system_path,
                location="",
                improve=True,
                info_radius=100000,
                preset_output_level=OutputLevel.FILENAMES,
                block=True,
                shift=True,
                topic=True,
                antithesis=True,
                thesis=True,
                synthesis=True,
                inversion_antonym=True,
            )
        )
