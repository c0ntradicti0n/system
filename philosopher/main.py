import glob
import json
import os
import pprint
import re

import config
import openai
import simple_cache as simple_cache
from dotenv import load_dotenv
from helper import (OutputLevel, extract, get_prefix,
                    nested_dict_to_filesystem, post_process_tree,
                    sanitize_nested_dict, tree, update_nested_dict)
from regex import regex

from philosopher.llm_update_text import custom_parser, llm_update_text
from philosopher.llm_update_toc import llm_update_toc
from philosopher.yaml import (object_to_yaml_str, rec_sort,
                              single_key_completion)
from server.os_tools import git_auto_commit

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = ""


if os.environ.get("OPENAI_API_KEY"):
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")

    @simple_cache.cache_it(filename=os.getcwd() + "/dialectic_triangle.cache", ttl=120)
    def llm(instruction, text, model=os.environ.get("OPENAI_MODEL")):
        return openai.ChatCompletion.create(
            model=model,
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
        depth=3 if task == "text" else 7,
    )
    pprint.pprint(t)
    index = post_process_tree(single_key_completion(object_to_yaml_str(rec_sort(t))))

    if task == "index":
        instruction, prompt = llm_update_toc(index, kwargs, t)
    elif task == "text":
        instruction, prompt = llm_update_text(index, kwargs, t, base_path)
    else:
        raise NotImplementedError(f"{task} not implemented")

    print(prompt)
    print(location)
    print(f"{len(prompt)=} {len(index)=} {len(location)=}")

    task_cache = f".cache_{task}/"
    if not os.path.exists(task_cache):
        os.makedirs(task_cache, exist_ok=True)
    lllm_output = task_cache + "response.txt"
    lllm_input = task_cache + "prompt.txt"

    os.makedirs(".cache_text/", exist_ok=True)
    with open(".cache_text/response.txt", "w") as f:
        f.write("")
    pass

    if not os.path.exists(lllm_output):
        with open(lllm_input, "w") as f:
            f.write(instruction.strip() + "\n" + prompt + "\n")

        api_result = llm(
            instruction=instruction.strip(), text=prompt, model=kwargs["model"]
        )
        output = api_result["choices"][0]["message"]["content"]

        with open(lllm_output, "w") as f:
            f.write(output + "\n")
    else:
        with open(lllm_output) as f:
            output = f.read()

    if task == "index":
        xpaths = output.split("\n")
        xpaths = [x for x in xpaths if x]
        xpaths = [regex.sub(r"^\d\.", "", x) for x in xpaths]
        print(xpaths)

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
        print(f"TOC created {created} files.")

    elif task == "text":
        # The findall function of the re module is used to get all matching patterns.
        split_strings = custom_parser(output, "# (\d|_)+ .*$")
        for header, content in split_strings:
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

        print(split_strings)


if __name__ == "__main__":
    for i in range(3):
        with git_auto_commit(
            config.system_path, commit_message_prefix="Automated TEXT Commit"
        ) as ctx:
            print(
                dialectic_triangle(
                    base_path=config.system_path,
                    location="",
                    task="text",
                    info_radius=100000,
                    preset_output_level=OutputLevel.FILENAMES,
                    # llm features
                    block=True,
                    shift=True,
                    topic=True,
                    antithesis=True,
                    thesis=True,
                    synthesis=True,
                    inversion_antonym=True,
                    # llm model
                    model="gpt-3.5-turbo",
                )
            )
        os.system("rm -rf .cache_text/")

    with git_auto_commit(
        config.system_path, commit_message_prefix="Automated TOC Commit"
    ) as ctx:
        print(
            dialectic_triangle(
                base_path=config.system_path,
                location="",
                task="index",
                info_radius=100000,
                preset_output_level=OutputLevel.FILENAMES,
                # llm features
                block=True,
                shift=True,
                topic=True,
                antithesis=True,
                thesis=True,
                synthesis=True,
                inversion_antonym=True,
                # llm model
                model="gpt-4",
            )
        )
