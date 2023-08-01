import os

from helper import get_from_nested_dict


def add_line_if_not_exists(filename, line_to_add):
    if not os.path.exists(filename):
        with open(filename, "w")  as f:
            f.write("")

    with open(filename, 'r+') as f:
        lines = f.read().splitlines()
        if line_to_add not in lines:
            f.write('\n' + line_to_add)

def remove_line_if_exists(filename, line_to_remove):
    with open(filename, 'r') as f:
        lines = f.readlines()
    with open(filename, 'w') as f:
        for line in lines:
            if line.strip("\n") != line_to_remove:
                f.write(line)

def add_to_missing_in_toc(t, where):
    for x in where:
        try:
            get_from_nested_dict(t, x)
        except:
            add_line_if_not_exists(os.environ.get("MISSING_CHAPTERS", "missing.txt"), "".join( str(xx) for xx in x))
