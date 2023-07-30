# https://stackoverflow.com/questions/47614862/best-way-to-use-ruamel-yaml-to-dump-yaml-to-string-not-to-stream
from io import StringIO
from pathlib import Path

import ruamel.yaml

# setup loader (basically options)
yaml = ruamel.yaml.YAML()
yaml.version = (1, 2)
yaml.indent(mapping=1, sequence=1, offset=0)
yaml.allow_duplicate_keys = True
yaml.explicit_start = False


# show null
def my_represent_none(self, data):
    return self.represent_scalar("tag:yaml.org,2002:null", "null")


yaml.representer.add_representer(type(None), my_represent_none)


# o->s
def object_to_yaml_str(obj, options=None):
    if options == None:
        options = {}
    string_stream = StringIO()
    yaml.dump(obj, string_stream, **options)
    output_str = string_stream.getvalue()
    string_stream.close()
    return output_str


# s->o
def yaml_string_to_object(string, options=None):
    if options == None:
        options = {}
    return yaml.load(string, **options)


# f->o
def yaml_file_to_object(file_path, options=None):
    if options == None:
        options = {}
    as_path_object = Path(file_path)
    return yaml.load(as_path_object, **options)


def sort_by_digit_or_dot_or_underscore(s):
    if isinstance(s, int) or s.isdigit():
        return int(s), int(s)
    else:
        return 0, s


# https://stackoverflow.com/questions/40226610/ruamel-yaml-equivalent-of-sort-keys
def rec_sort(d):
    try:
        if isinstance(d, ruamel.yaml.CommentedMap):
            return d.sort()
    except AttributeError:
        pass
    if isinstance(d, dict):
        # could use dict in newer python versions
        res = ruamel.yaml.CommentedMap()
        for k in sorted(d.keys(), key=sort_by_digit_or_dot_or_underscore):
            res[k] = rec_sort(d[k])
        return res
    if isinstance(d, list):
        for idx, elem in enumerate(d):
            d[idx] = rec_sort(elem)
    return d


def single_key_completion(yaml_str):
    new_lines = []
    keys = ""
    for line in yaml_str.split("\n"):
        if ":" in line:
            sep = line.index(":")
            active_key = line[sep - 1]
            keys = keys[: sep - 1] + active_key

            rest = line[sep + 1 :]
            if not rest.strip() == "":
                if active_key.isdigit():
                    keys += "."

                content = keys + line[sep + 1 :]

                new_lines.append(content)

    return "\n".join(new_lines)
