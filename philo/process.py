import glob
import json
import os
from pprint import pprint
import regex as re

from openai import OpenAI

from lib import config
from lib.os_tools import git_auto_commit
from lib.t import catchtime

client = OpenAI()
import regex
from llm_update_text import custom_parser, llm_update_text
from llm_update_toc import llm_update_toc
from yaml_tools import object_to_yaml_str, rec_sort, single_key_completion

from lib.helper import (OutputLevel, extract, get_prefix, post_process_tree,
                        sanitize_nested_dict, tree, update_nested_dict, nested_dict_to_filesystem)

models = {
    "toc": "gpt-4-1106-preview",
    "text": "gpt-4-1106-preview",
}


def llm(instruction, text, model=os.environ.get("OPENAI_MODEL")):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ],
    )


def generate_prompt(
    base_path,
    location,
    task="index",
    info_radius=2,
    preset_output_level=None,
):
    location_path = "/".join(location)
    if task == "toc":
        kwargs = dict(  # llm features
            block=True,
            shift=True,
            topic=True,
            antithesis=True,
            thesis=True,
            synthesis=True,
            inversion_antonym=True,
            # llm model
            model="gpt-4-1106-preview",
        )
    elif task == "text":
        kwargs = dict(  # llm features
            # llm features
            block=True,
            shift=False,
            topic=True,
            antithesis=True,
            thesis=True,
            synthesis=True,
            inversion_antonym=True,
            # llm model
            model="gpt-4-1106-preview",
        )
    with catchtime("tree"):
        t = tree(
            basepath=base_path,
            startpath="",
            sparse=True,
            info_radius=info_radius,
            location=location_path,
            pre_set_output_level=preset_output_level,
            exclude=(".git", ".git.md", ".idea"),
            prefix_items=True,
            depth=100 if task == "text" else 7,
        )
    # pprint.pprint(t)
    with catchtime("post_process_tree"):
        index = post_process_tree(single_key_completion(object_to_yaml_str(rec_sort(t))))

    with catchtime("taks"):

        if task == "toc":
            instruction, prompt = llm_update_toc(
                index, kwargs, t, path_to_fill=location_path
            )
        elif task == "text":
            instruction, prompt = llm_update_text(
                index, kwargs, t, base_path=os.path.join(config.system_path, location_path)
            )
        else:
            raise NotImplementedError(f"{task} not implemented")

    # print(prompt)
    # print(location)
    # print(f"{len(prompt)=} {len(index)=} {len(location)=}")

    return instruction, prompt


def post_process_model_output(output, task, t, _path):
    if task == "toc":
        xpaths = output.split("\n")
        xpaths = [x for x in xpaths if x]
        print(xpaths)

        paths = []
        while not paths:
            for path in xpaths:
                path = regex.sub(rf"^{_path}", '', path)

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
            keys = m["key"]
            keys = [k for k in keys]
            last_key = keys[-1]
            prefix = get_prefix(last_key)

            if m["key"].startswith("111"):
                continue

            v = m["value"].replace("/", "-")
            if "{" in v and "}" in v:
                vv = m["value"]
                vv = regex.sub(r"(\d):", '"\\1":', vv)
                vv = vv.replace('""', "_")
                v = json.loads(vv)
                if not keys[-1] == ".":
                    keys.append(".")
            else:
                v = v.replace('"', "").replace("'", "").replace(":", "")

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
            healthy_last_key=lambda x: regex.match(r"^([1-3]|\.|_)", x)
            if isinstance(x, str)
            else x,
            healthy_normal_key=lambda x: regex.match(r"^[1-3]$", x) and len(x) == 1
            if isinstance(x, str)
            else x,
            last_key_to_normal_key=lambda x: regex.match(r"^([1-3]|\.|_)", x).group(0)
            if isinstance(x, str)
            else x,
        )
        return t
    elif task == "text":
        # The findall function of the re module is used to get all matching patterns.
        split_strings = custom_parser(output, "# (\d|_)+ .*$")
        return split_strings


import os

from joblib import Memory

# Set up a cache directory
cachedir = "./cache-openai"

if not os.path.exists(cachedir):
    os.makedirs(cachedir, exist_ok=True)

memory = Memory(cachedir, verbose=0)


@memory.cache
def process_user_input(hint, instruction, prompt, task):
    api_result = llm(
        instruction=instruction.strip(),
        text=prompt + "\n\nThink also of " + hint,
        model=models[task],
    )
    output = api_result.choices[0].message.content

    print(output)
    return output

def commit(data, task, path, token
           ):
    with git_auto_commit(
            config.system_path, commit_message_prefix=f"Automated {task} Commit {path} from {token}"
    ):
        print(data)

        if task == "toc":
            fs_path = "/".join(path)
            base_path = os.path.join(config.system_path, fs_path)
            created = nested_dict_to_filesystem(f"{base_path}".strip(), data)
            print(f"TOC created {created} files.")

        elif task == "text":
            # The findall function of the re module is used to get all matching patterns.
            for header, content in data:
                try:
                    path = re.match("# (\d*(\d|_)) .+$", header).group(1)
                except AttributeError:
                    print(f"Could not parse {header}")
                    continue
                if path[-1].isdigit():
                    file_pattern = base_path + "/" + "/".join(path) + "/.*.md"
                else:
                    file_pattern = base_path + "/" + "/".join(path) + "*.md"
                files = glob.glob(file_pattern)
                assert files[0]
                with open(files[0], "w") as f:
                    f.write(content)




if __name__ == "__main__":
    location = os.environ.get("LOCATION", "2333")
    location_path = "/".join(location)
    task = os.environ.get("TASK", "toc")
    preset_output_level = os.environ.get("PRESET_OUTPUT_LEVEL", OutputLevel.FILENAMES)

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

    pprint(t)

    base_path = os.environ.get("SYSTEM", "/home/runner/work/llm/llm")
    info_radius = int(os.environ.get("INFO_RADIUS", 2))

    instruction, prompt = generate_prompt(
        base_path=base_path,
        location=location,
        task=task,
        info_radius=info_radius,
        preset_output_level=preset_output_level,
    )

    print(prompt)

    hint = "The first sentence of the first paragraph of the first chapter of the first book of the first volume of the first edition of the first translation of the first work of the first philosopher."

    output = process_user_input(hint, instruction, prompt, task)

    print(output)

    t = post_process_model_output(output, task)

    print(t)

    print(object_to_yaml_str(t))
