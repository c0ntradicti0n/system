import logging
import os
import pathlib
import re
import shlex
import subprocess
from enum import Enum
from functools import total_ordering

from cache import file_based_cache
from jsonpath_ng import parse
from Levenshtein import distance
from regex import regex


def o(cmd, user=None, cwd="./", err_out=True):
    try:
        logging.info(f"calling {user=}: " + cmd)

        output, error = subprocess.Popen(
            [shlex.quote(c) for c in shlex.split(cmd)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            user=user,
        ).communicate()
        if error and err_out:
            raise (Exception(error.decode()))
    except PermissionError:
        logging.error("can only run in docker")
    except Exception as e:
        logging.error(e, exc_info=True)
    return output.decode() if output else error.decode()


def sort_key(name):
    # Prioritize hidden files and directories by negating the result of str.startswith()
    return not name.startswith("."), name.startswith("_"), name.lower()


def update_nested_dict(
    nested_dict, keys, value, prefix="", change_keys=False, sanitize_key=lambda x: x
):
    keys = digitize(keys)

    current_dict = nested_dict
    new = False

    for key in keys[:-1]:
        try:
            if key in current_dict and current_dict[key] is not None:
                if isinstance(current_dict[key], str):
                    current_dict[key] = {".": current_dict[key]}
                    new = True
                    current_dict = current_dict[key]

                else:
                    current_dict = current_dict[key]
            else:
                current_dict[sanitize_key(key)] = {}
                current_dict = current_dict[key]
        except TypeError:
            raise

    if change_keys:
        key_to_remove = None
        try:
            for k in current_dict.keys():
                if k.startswith(keys[-1][0]):
                    key_to_remove = k
                    break
        except:
            pass

        if key_to_remove is not None and not new:
            new_key = prefix
            v = current_dict[key_to_remove]
            del current_dict[key_to_remove]

    if isinstance(value, dict):
        if "." in value:
            current_dict["."] = value["."]
        if "_" in value:
            current_dict["_"] = value["_"]
        if "1" in value:
            if not isinstance(current_dict.get(1), dict):
                current_dict[1] = {}
            current_dict[1]["."] = value["1"]
        if "2" in value:
            if not isinstance(current_dict.get(2), dict):
                current_dict[2] = {}
            current_dict[2]["."] = value["2"]
        if "3" in value:
            if not isinstance(current_dict.get(3), dict):
                current_dict[3] = {}
            current_dict[3]["."] = value["3"]
    else:
        current_dict[keys[-1]] = value


def digitize(keys):
    return [
        k if isinstance(k, int) else (k if not k.isdigit() else int(k))
        for k in keys
        if k
    ]


def add_to_nested_dict(nested_dict, keys, value):
    keys = digitize(keys)
    for key in keys[:-1]:
        nested_dict = nested_dict.setdefault(key, {})
    nested_dict[keys[-1]] = value


def exists_in_nested_dict(nested_dict, keys):
    keys = digitize(keys)

    for key in keys[:-1]:
        nested_dict = nested_dict.setdefault(key, {})
    return keys[-1] in nested_dict


def get_from_nested_dict(nested_dict, keys, return_on_fail=None):
    try:
        keys = digitize(keys)
        old_dict = nested_dict

        for key in keys[:-1]:
            if key == "..":
                nested_dict = old_dict

            old_dict = nested_dict
            nested_dict = nested_dict.setdefault(key, {})
        return nested_dict[keys[-1]]
    except KeyError:

        if return_on_fail is not None:
            return return_on_fail
        else:
            raise


def round_school(x):
    i, f = divmod(x, 1)
    return int(i + ((f >= 0.5) if (x > 0) else (f > 0.5)))


@total_ordering
class OutputLevel(Enum):
    FILE = 1
    FILENAMES = 2
    DIRECTORY = 3
    TOPIC = 4
    NONE = 5

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value


# @file_based_cache
def tree(
    startpath,
    basepath,
    indent=0,
    output=None,
    format="string",
    keys=[],
    sparse=False,
    info_radius=100,
    location="",
    pre_set_output_level=OutputLevel.TOPIC,
    exclude=[],
    prefix_items=False,
    depth=None,
):
    if not pre_set_output_level:
        d = distance(startpath, location, weights=(1000, 1000, 100000))
        visibility = round_school(max(d + info_radius, 0) / 2000)
        match visibility:
            case x if x in {
                0,
                1,
                2,
            }:
                output_level = OutputLevel.FILE
            case x if x in {3, 4, 5}:
                output_level = OutputLevel.FILENAMES
            case x if x in {6, 7}:
                output_level = OutputLevel.DIRECTORY
            case x if x in {8, 9}:
                output_level = OutputLevel.TOPIC
            case _:
                output_level = OutputLevel.TOPIC
    else:
        output_level = pre_set_output_level

    path = os.path.join(basepath, startpath)

    if output is None:
        if format == "string":
            output = []
        elif format == "json":
            output = dict()

    if format == "string":
        if indent > 0:
            output.append("{}- {}/".format("   " * indent, os.path.basename(path)))
    elif format == "json":
        pass

    indent += 1

    subdirs = [
        name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
    ]
    files = [
        name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))
    ]

    for name in sorted(subdirs, key=sort_key):
        if name in exclude:
            continue
        if format == "string":
            result = tree(
                os.path.join(path, name),
                indent,
                format=format,
                sparse=sparse,
                exclude=exclude,
            )
            if output_level <= OutputLevel.DIRECTORY:
                output.append(result)
        elif format == "json":
            if depth == None or depth >= 0:
                tree(
                    startpath=os.path.join(startpath, name),
                    indent=indent,
                    format="json",
                    keys=keys + [name],
                    output=output,
                    sparse=sparse,
                    location=location,
                    info_radius=info_radius,
                    basepath=basepath,
                    pre_set_output_level=pre_set_output_level,
                    exclude=exclude,
                    prefix_items=prefix_items,
                    depth=depth - 1 if depth is not None else None,
                )
                if not exists_in_nested_dict(output, keys + [name]):
                    if output_level <= OutputLevel.DIRECTORY:
                        add_to_nested_dict(output, keys + [name], None)
                    elif output_level <= OutputLevel.TOPIC and name.startswith("."):
                        add_to_nested_dict(output, keys + [name], None)

    for name in sorted(files, key=sort_key):
        if name in exclude:
            continue
        try:
            content = pathlib.Path(os.path.join(path, name)).read_text("utf-8")
            if sparse:
                content = content[:100]
        except Exception as e:
            raise Exception("error reading file", os.path.join(path, name)) from e

        if format == "json":
            k = name

            if output_level <= OutputLevel.FILE:
                add = True
                v = content

            elif output_level <= OutputLevel.FILENAMES:
                v = None
                add = True

            elif output_level <= OutputLevel.TOPIC and name.startswith("."):
                v = None
                add = True

            else:
                add = False

            if prefix_items and add and not v:
                try:
                    k = regex.match(r"(\d+-|\.|_)", name).group(0)
                except Exception as e:
                    raise Exception("error matching prefix", name) from e
                v = name[len(k) :]
                k = k[:1]

            if add:
                add_to_nested_dict(output, keys + [k], v)

    if format == "json":
        return output


def remove_last_line_from_string(s):
    return s[: s.rfind("\n")]


def update_with_jsonpath(json_obj, jsonpath_expr, value):
    expr = parse(jsonpath_expr)
    matches = [match for match in expr.find(json_obj)]

    if matches:
        # Get the last path part
        last_path_part = matches[0].path.fields[-1]
        # Get the parent of the match
        parent_obj = matches[0].context.value
        # Update the value
        parent_obj[last_path_part] = value
    else:
        print(f"No matches for '{jsonpath_expr}' in JSON object.")
        return None

    return json_obj


def nested_dict_to_filesystem(path, tree, prefix_items=False, creations=None):
    try:
        if creations is None:
            creations = []
        for key, value in tree.items():
            if isinstance(value, dict):
                nested_dict_to_filesystem(
                    os.path.join(path, str(key).strip()),
                    value,
                    prefix_items=prefix_items,
                    creations=creations,
                )
            else:
                os.makedirs(path, exist_ok=True)
                new_file = os.path.join(path, str(key).strip())

                if new_file[-1].isdigit():
                    new_file = new_file + "/."

                if value is None:
                    logging.error(f"Value is None for {str(tree)}")
                    continue
                new_file = new_file + value

                if not new_file.endswith(".md"):
                    new_file += ".md"

                new_file = new_file.replace('"', "")

                if not os.path.exists(new_file):
                    prefix = get_prefix(key)

                    os.system(f"rm -rf {path}/{prefix}*.md")
                    os.system(f"rm -rf {path}/{prefix}*")
                    os.makedirs(os.path.dirname(new_file), exist_ok=True)
                    with open(new_file, "w") as f:
                        f.write(value if value else "")
                    creations.append(new_file)
        return creations
    except Exception as e:
        logging.error(f"Error creating file {new_file}: {e}", exc_info=True)


def get_prefix(last_key):
    if isinstance(last_key, int):
        last_key = str(last_key)
    if last_key.startswith("_"):
        return "_"
    elif last_key.startswith("."):
        return "."
    else:
        return last_key[:2]


def extract(x):
    try:
        m = regex.match(r"(?P<key>^[\d\._]+) (?P<value>.+)\"?$", x).groupdict()
        return m
    except:
        print(f"not matched: {x}")
        return {"key": "not matched", "value": "not matched"}


def sanitize(x, stuff=[]):
    for s in stuff:
        x = regex.sub(s, "", x)
    return x


def nested_str_dict(d):
    d_to_update = []
    for k, v in d.items():
        if isinstance(v, dict):
            nested_str_dict(v)
        if not isinstance(k, str):
            d_to_update.append((k, v))
    for k, v in d_to_update:
        d[str(k)] = v
        del d[k]

    return d


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
    tree = regex.sub(r".+ null(\n|$)", r"", tree)
    tree = regex.sub(r"((\d*)\.)", r"\2", tree)
    return tree


if __name__ == "__main__":
    t = tree(
        basepath="../product/", startpath="", format="json", sparse=True, location="1/1"
    )
    update_with_jsonpath(t, "$.1.1.3.topic", "Multiple Existence")

    print(tree("../product/", format="json", sparse=True, location="1/1"))
