import os
import re


def list_files_with_regex(directory, pattern):
    # List all files in the directory
    all_files = os.listdir(directory)

    # Filter files based on the regex pattern and extract matched groups
    matched_groups = []
    for f in all_files:
        match = re.match(pattern, f)
        if match:
            matched_groups.append(match.groupdict())
    print(f"{matched_groups=}")
    return matched_groups


if __name__ == "__main__":
    print(list_files_with_regex("../integrator/states/", "(?P<hash>.*)\-text.pkl"))
