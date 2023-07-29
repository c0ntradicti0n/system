import json
import logging
import os
import pathlib
import pprint
import re

import guidance
import openai
import simple_cache as simple_cache
from regex import regex

import config
from helper import (
    o,
    remove_last_line_from_string,
    tree,
    OutputLevel,
    extract,
    sanitize,
    update_with_jsonpath,
    nested_dict_to_filesystem,
    update_nested_dict,
    get_prefix,
)
from philosopher.LLMFeature import features, create_path
from philosopher.analyser import analyse
from server.os_tools import git_auto_commit
from philosopher.yaml import (
    object_to_yaml_str,
    yaml_string_to_object,
    yaml_file_to_object,
    rec_sort,
    single_key_completion,
)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import os
from dotenv import load_dotenv

load_dotenv()
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

explanation_of_method = """We take a note for the general topic.
1. Thesis
2. Antithesis
3. Synthesis
Also we want to write down a central notion of the developing argumentation, that marks the conflict of the dialectic triple to develop the next one.
It has to be turning point empowering some notion like 'minus * minus = plus', or 'disappearing of disappearing leaves back existence'.
This must be like an antonym of the thesis, that can be applied to itself. We left out some values to be more parsimonious.
"""


def sanitize_nested_dict(
    d,
    sanitize_key=lambda x: x,
    sanitize_value=lambda x: x,
    healthy_last_key=lambda x: True,
    healthy_normal_key=lambda x: True,
    last_key_to_normal_key=lambda x: x,
):
    if isinstance(d, dict):
        r = {}
        for k, v in d.items():
            new_key = sanitize_key(k)
            if isinstance(d[k], dict) and healthy_last_key(k):
                new_key = last_key_to_normal_key(new_key)
            if (
                not v
                or isinstance(d[k], str)
                and healthy_last_key(k)
                or healthy_normal_key(k)
                or isinstance(d[k], dict)
                and healthy_last_key(k)
            ):
                r[new_key] = sanitize_nested_dict(
                    d[k],
                    sanitize_key=sanitize_key,
                    sanitize_value=sanitize_value,
                    healthy_last_key=healthy_last_key,
                    healthy_normal_key=healthy_normal_key,
                    last_key_to_normal_key=last_key_to_normal_key,
                )
            else:
                print("dropped key " + k)
                continue
        return r
    elif isinstance(d, list):
        return [
            sanitize_nested_dict(
                e, sanitize_key=sanitize_key, sanitize_value=sanitize_value
            )
            for e in d
        ]
    elif isinstance(d, str):
        return sanitize_value(d)


def dialectic_triangle(
    base_path,
    location,
    dive_deeper=False,
    improve=False,
    info_radius=2,
    preset_output_level=None,
    **kwargs,
):
    location_path = "/".join(location)
    path = base_path + "/".join(location_path)

    if not improve:
        if dive_deeper:
            last_location = location[-1]
            location = location[:-1]

        touch(base_path + "/" + location_path, is_dir=True)
        dir_contents = os.listdir(base_path + f"/{location_path}")
        if (dir_contents and not dive_deeper) and not improve:
            logging.error("target location not empty")
            return
        if dir_contents and dive_deeper:
            files = [c for c in dir_contents if c.endswith(".md")]
            antonyms = [c for c in dir_contents if c.startswith("_")]
            topic = [c for c in dir_contents if c.startswith(".")]
            file_to_shift = [f for f in files if f[0] == last_location]
            assert file_to_shift.__len__() == 1, "file_to_shift must be unique"

            if file_to_shift:
                logging.info(f"shifting file to child for {topic}: {file_to_shift[0]}")
                match = regex.match("(?P<num>\d)-(?P<topic>.*)\.md", file_to_shift[0])
                new_topic = match.group("topic")
                touch(
                    base_path
                    + "/"
                    + location_path
                    + "/"
                    + last_location
                    + "/."
                    + new_topic,
                    is_dir=True,
                )

                os.system("rm " + f"{base_path}/{location_path}/{file_to_shift[0]}")
                location_path = location_path + "/" + last_location

        new_path = base_path + "/".join(location_path)

        generate_prompting_file(
            location_path, f"topic", prefix=".", prompt_content="generate topic"
        )
        generate_prompting_file(
            location_path,
            prefix="1-",
            prompt_content=f"generate thesis title",
            content=f"content of thesis",
        )
        generate_prompting_file(
            location_path,
            prefix="2-",
            prompt_content=f"generate antithesis title",
            content=f"content of antithesis",
        )
        generate_prompting_file(
            location_path,
            prefix="3-",
            prompt_content=f"generate synthesis title",
            content=f"content of synthesis",
        )
        generate_prompting_file(
            location_path, f"inversive dialectical antonym", prefix="_"
        )
    t = tree(
        basepath=base_path,
        startpath="",
        format="json",
        sparse=True,
        info_radius=info_radius,
        location=location_path,
        pre_set_output_level=preset_output_level,
        exclude=[".git", ".git.md", ".idea"],
        prefix_items=improve,
    )
    pprint.pprint(t)
    index = post_process_tree(single_key_completion(object_to_yaml_str(rec_sort(t))))

    location_xpath = location_path.replace("/", ".")

    if improve:
        shortest_path = [
            create_path(p) for p in analyse(t, exclude=["111"])["min_depth_paths"]
        ]

        prompt_preface = (
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

Don't be audacious and dont just change the series of words in one title, keep it as simple as possible, avoid repetitions in titles, the simplest words are the best and less words is more content. Be enourmously precise with your answers and the keys, every wrong path causes chaos and will kill the whole project.
Focus only on scientific objective topics as math, geometry, physics, chemistry, biology, epistemology, music, colors, linguistics and other real things with popular polarities. Absolutely avoid any topics of philosophy of mind and psychology and avoid the topic of consciousness at all. Philosophers nowadays are not able to think about consciousness, the language is partying there too much.
Focus on a top-down approach to get to some more systematic dialectical structure; first all upper levels, then the lower levels.
Focus on completeness of the fractal, please fill up all incomplete triples, rather add new triples than improving existing ones.

And please dive deeper into """
            + " and ".join(shortest_path)
        )
        provide_chapter = False
    elif dive_deeper:
        theme = post_process_tree(file_to_shift[0])
        provide_chapter = True
        prompt_preface = f"""
You are rewriting Hegels System. It's underlying logic is more semantic and develops concepts in a fractal of triples following
the method of dialectics. 

We want to dive deeper into the topic {location[-1]}. Please help us to find the next triple. It should present the
topic, thesis (1) and antithesis (2) and synthesis (3) and provide information about the dialectic movement by the antonymical expression. As an
inversive antonym as "minus * minus = plus", or "disappearing of disappearing leaves back existence".

{explanation_of_method}
"""
    else:
        provide_chapter = True
        prompt_preface = f"""
You are rewriting Hegels System. It's underlying logic is more semantic and develops concepts in a fractal of triples.
We try to redo this from scratch finding analogies. Please help us to find the next triple. It should follow the
previous triples and bring three new concepts and explanations into place. It should follow the dialectic method of Hegel.

{explanation_of_method}
"""

    if provide_chapter:
        prompt_preface = f"""

To avoid to much output, just answer with json paths and values in this form:
Only answer for values that are enclosed in double curly braces '{{{{'

$.{location_xpath}.topic "{{topic}}"
$.{location_xpath}.1-title "{{title thesis}}"
$.{location_xpath}.1-text "{{content thesis}}"
$.{location_xpath}.2-title "{{title antithesis}}"
$.{location_xpath}.2-text "{{content antithesis}}"
$.{location_xpath}.3-title "{{title synthesis}}"
$.{location_xpath}.3-text "{{content synthesis}}"
$.{location_xpath}.antonym "{{_inverse antonym}}"
$.{location_xpath}.antonym_explanation "{{_explanation of inverse antonym}}"

The place to think about is: {location_xpath} and the topic should be {theme}

"""
    else:
        prompt = f"""
{index}
"""

    print(prompt)
    print(location)
    print(f"{len(prompt)=} {len(index)=} {len(location)=}")

    lllm_output = ".cache/lllm.output.txt"
    lllm_input = ".cache/lllm.input.txt"

    if not os.path.exists(lllm_output):
        api_result = llm(instruction=prompt_preface.strip(), text=prompt)
        output = api_result["choices"][0]["message"]["content"]
        with open(lllm_input, "w") as f:
            f.write(prompt_preface.strip() + "\n" + prompt + "\n")

        with open(lllm_output, "w") as f:
            f.write(output + "\n")
    else:
        with open(lllm_output) as f:
            output = f.read()

    xpaths = output.split("\n")
    xpaths = [x for x in xpaths if x]
    xpaths = [regex.sub(r"^\d\.", "", x) for x in xpaths]
    print(xpaths)

    if improve:
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
    else:
        result = {
            "topic": extract(xpaths[0])["value"],
            "title1": sanitize(extract(xpaths[1])["value"], ["(^| )?Thesis( |$)?"]),
            "explanation1": extract(xpaths[2])["value"],
            "title2": extract(xpaths[3])["value"],
            "explanation2": extract(xpaths[4])["value"],
            "title3": extract(xpaths[5])["value"],
            "explanation3": extract(xpaths[6])["value"],
            "inversion-antonym": extract(xpaths[7])["value"],
            "explanation4": extract(xpaths[8])["value"],
        }

        if not dive_deeper:
            dump(f"{new_path}/1-{result['topic'].strip()}.md", "")

        dump(f"{new_path}/1-{result['title1'].strip()}.md", result["explanation1"])
        dump(f"{new_path}/2-{result['title2'].strip()}.md", result["explanation2"])
        dump(f"{new_path}/3-{result['title3'].strip()}.md", result["explanation3"])
        dump(
            f"{new_path}/_{result['inversion-antonym'].strip()}.md",
            result["explanation4"],
        )  # result['explanation4'])

        if dive_deeper:
            os.system("rm " + f"{path}/" + file_to_shift[0].replace(" ", r"\ ") + "")
            os.system("find " + new_path + r" -regex '.*\{.*'  -delete ")

        return result


def generate_prompting_file(
    base_path, location, prompt_content, is_dir=False, prefix="", content=None
):
    AI_COMMAND = ""
    touch(
        rf"{base_path}/{location}/{prefix}{AI_COMMAND}{{{{{prompt_content}}}}}",
        is_dir=is_dir,
    )
    if content:
        with open(
            rf"{base_path}/{location}/{prefix}{AI_COMMAND}{{{{{prompt_content}}}}}",
            "w",
        ) as f:
            f.write(f"{AI_COMMAND}{{{{{content}}}}}")


def touch(path, is_dir=False):
    if is_dir:
        os.makedirs(path, exist_ok=True)
    else:
        pathlib.Path(path).touch()


def dump(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def post_process_tree(tree):
    tree = (
        tree.replace(".md", "")
        .replace("- _", "- self-antonym: ")
        .replace("- .", "- topic: ")
        .replace(r"/", "")
    )
    tree = regex.sub("- (\d+)-?", r"\1. ", tree)
    return tree


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
