import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

from lib.helper import OutputLevel, tree

from enum import Enum

from lib.json import encode, decode


class ConceptPosition(Enum):
    SUMMARIZING_CONCEPT = 0
    THESIS = 1
    ANTITHESIS = 2
    SYNTHESIS = 3
    MORE_ABSTRACT = 4
    MORE_CONCRETE_COMPLEX = 5
    SUBSUMED_INTO_THESIS = 6
    SUBSUMED_INTO_ANTITHESIS = 7
    SUBSUMED_INTO_SYNTHESIS = 8

    def __int__(self):
        return self.value

int(ConceptPosition.SUMMARIZING_CONCEPT)  # 0
def random_123_string(max_length, start_string=None, shorter = False, longer = False, return_extension=False):
    """
    Generate a random string of "1", "2", and "3" within a given length.
    If start_string is provided, modify it based on a probability.

    :param max_length: Maximum length of the random string.
    :param start_string: Optional starting string to modify.
    :param p: Probability of extending the string. With 1-p chance, the string will be stripped.
    :return: Modified or generated string.
    """

    # Function to generate a random string of "1", "2", and "3"
    def generate_random_string(length):
        return ''.join(random.choice(['1', '2', '3']) for _ in range(length))

    if start_string:
        if not (shorter ^ longer):
            raise ValueError("Must set either longer or shorter for the random string gen with a given string")
        # Determine the action: True to extend, False to strip
        modify_length = random.randint(1, max_length)

        if longer:
            # Extend the string
            extension = generate_random_string(modify_length)
            result_string = start_string + extension
        elif shorter:
            # Strip the string, ensuring it does not become shorter than 1
            strip_length = min(modify_length, len(start_string))
            result_string = start_string[:-strip_length] if strip_length < len(start_string) else start_string[0]
    else:
        # Generate a new random string
        random_length = random.randint(1, max_length)
        result_string = generate_random_string(random_length)

    if return_extension:
        return result_string, extension
    return result_string


def find_matching_entries(nested_dict):
    # Stack item format: (current_dict, path_to_current_dict)
    stack = [(nested_dict, [])]
    while stack:
        current_dict, path = stack.pop()
        # Check if current_dict is a valid candidate (having keys 1, 2, 3)
        if all(key in current_dict for key in [1, 2, 3]):
            # Retrieve the parent dictionary to check for the "." key
            if path:
                parent_path = path[:-1]
                parent_dict = nested_dict
                for step in parent_path:
                    parent_dict = parent_dict[step]
                if '.' in parent_dict:
                    # Found a match, return the relevant entries
                    return {key: current_dict[key] for key in [1, 2, 3, '.']}
        # Push children onto the stack
        for key, value in current_dict.items():
            if isinstance(value, dict):
                stack.append((value, path + [key]))
    # No matching entry found
    raise KeyError("No matching entry found.")


def make_sample(which_kind, which):
    path = random_123_string(10)
    try:
        file_dict = get_from_triangle(path)

    except FileNotFoundError:
        return None

    try:
        matching_entries = find_matching_entries(file_dict)
        a = matching_entries[1]["."]
        b = matching_entries[2]["."]
        c = matching_entries[3]["."]
        d = matching_entries["."]

    except KeyError:
        return None

    except TypeError:
        return None

    try:
        # 1 - use higher
        # 2 - use same
        # 3 - use deeper
        if which_kind == 1:
            # higher
            path2 = random_123_string(10)
            if path2 in path:
                return None
            if path in path2:
                return None
            file_dict = get_from_triangle(path2)
            matching_entries = find_matching_entries(file_dict)

            X = matching_entries["."]

            if (X in [a, b, c, d]):
                return None

            if path > path2:
                kind = ConceptPosition.MORE_ABSTRACT
            else:
                kind = ConceptPosition.MORE_CONCRETE_COMPLEX
        if which_kind == 2 or which_kind == 4:
            path2, extension = random_123_string(10, start_string=path, longer=True, return_extension=True)
            file_dict = get_from_triangle(path2)
            matching_entries = find_matching_entries(file_dict)

            if which == 1:
                X = a
                a = matching_entries["."]
                kind = ConceptPosition.THESIS
            if which == 2:
                X = b
                b = matching_entries["."]
                kind = ConceptPosition.ANTITHESIS
            if which == 3:
                X = c
                c = matching_entries["."]
                kind = ConceptPosition.SYNTHESIS
            if which == 4:
                X = d
                d = matching_entries["."]
                kind = ConceptPosition.SUMMARIZING_CONCEPT
        if which_kind == 3 or which_kind == 5:
            path2, extension = random_123_string(10, start_string=path, longer=True, return_extension=True)

            file_dict = get_from_triangle(path2)

            matching_entries = find_matching_entries(file_dict)
            X = matching_entries["."]
            if extension.startswith("1"):
                kind = ConceptPosition.SUBSUMED_INTO_THESIS
            if extension.startswith("2"):
                kind = ConceptPosition.SUBSUMED_INTO_ANTITHESIS
            if extension.startswith("3"):
                kind = ConceptPosition.SUBSUMED_INTO_SYNTHESIS
    except KeyError:
        return None
    except FileNotFoundError:
        return None

    assert all(
        isinstance(_, str) for _ in (a, b, c, d, X)), f"{a=}, {b=}, {c=}, {d=}, {X=}, {kind=} {which_kind=} {which=}"

    return a, b, c, d, X, kind
def sync_gen(config):
    which_kind = random.choice([1, 2, 3, 4, 5])
    which = random.choice([1, 2, 3, 4])

    while True:
        try:
            a, b, c, d, X, kind = make_sample(which_kind, which)
        except TypeError:
            continue
        yield tuple( (_, kind ) for _ in (a,b,c,d, X))
        print ("YIELD")
        # change next option to chooese
        which_kind = random.choice([1,2,3, 4, 5])
        which = random.choice([1, 2, 3, 4])


def parallel_gen(config, max_workers=160):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # A dictionary to map futures to their corresponding which_kind and which values
        future_to_params = {}

        which_kind_options = [1, 2, 3, 4, 5]
        which_options = [1, 2, 3, 4]

        # Initial submission of tasks
        for _ in range(max_workers):
            which_kind = random.choice(which_kind_options)
            which = random.choice(which_options)
            future = executor.submit(make_sample, which_kind, which)
            future_to_params[future] = (which_kind, which)

        while True:
            # Wait for at least one future to complete
            done, _ = as_completed(future_to_params.keys(), timeout=None), None

            for future in done:
                which_kind, which = future_to_params[future]  # Get the params used for this future

                try:
                    a, b, c, d, X, kind = future.result()
                    yield tuple((_, kind) for _ in (a, b, c, d, X))
                    #print("YIELD")
                except TypeError as e:
                    #print(f"Error with {which_kind}, {which}: {e}")
                    pass

                # Remove the completed future and its params
                del future_to_params[future]

                # Immediately submit a new task to keep the executor busy
                new_which_kind = random.choice(which_kind_options)
                new_which = random.choice(which_options)
                new_future = executor.submit(make_sample, new_which_kind, new_which)
                future_to_params[new_future] = (new_which_kind, new_which)

            # Break out of the loop if we somehow run out of futures, should not happen
            if not future_to_params:
                break
def get_from_triangle(path):
    file_dict = tree(
        basepath=config.system_path,
        startpath="/".join(path),
        format="json",
        keys=path.split("/"),
        info_radius=0,
        exclude=[".git", ".git.md", ".gitignore", ".DS_Store", ".idea"],
        pre_set_output_level=OutputLevel.FILENAMES,
        prefix_items=True,
        depth=1,
    )
    return file_dict

def store_and_yield_gen(original_gen, filepath):
    """
    A generator that stores tuples returned by another generator in a file and yields them.

    :param original_gen: The original generator yielding tuples.
    :param filepath: Path to the file where tuples will be stored.
    """
    for item in original_gen:
        # Convert the tuple to a JSON string for storage
        # Note: This requires the items in the tuple to be serializable by json.dumps.
        with open(filepath, 'a') as f:
            s = encode(item, ConceptPosition)
            f.write(s)
            f.write('\n')  # Ensure each item is on a new line for readability
        yield item


def yield_from_file(config, filepath="___.txt", cycle=True):
    """
    A generator that reads tuples from a file (stored in JSON format) and yields them.
    Optionally cycles through the data again if the end of the file is reached.

    :param filepath: Path to the file containing the stored tuples.
    :param cycle: If True, continue to yield data in a cycle after reaching the end of the file.
    """
    while True:  # Loop to allow for cycling
        with open(filepath, 'r') as file:
            lines = file.readlines()
            if not lines:  # If the file is empty, stop or continue based on the cycle flag
                if cycle:
                    continue  # Restart the loop to cycle through the file again
                else:
                    break  # Exit the loop and stop yielding if not cycling
            for line in lines:
                # Deserialize the JSON string back into a tuple
                item = decode(line, ConceptPosition)


                # Assuming the item was originally a tuple of tuples
                a, b, c, d, X = item  # Unpack the tuple

                yield (
                    ("thesis: " + a[0], a[1] ),
                    ("antithesis: " + b[0], b[1] ),
                    ("synthesis: " + c[0], c[1] ),
                    ("summary topic: " + d[0], d[1] ),
                    ("new: " + X[0], X[1] ),
                )
        if not cycle:
            break  # Exit the while loop if cycling is not enabled


gen = yield_from_file

if __name__ == "__main__":
    print(random_123_string(10))  # Generates a new string
    print(random_123_string(10, start_string="123", longer=True))  # Modifies the given

    # Example usage with the provided dictionary structure
    nested_dict = {
        2: {
            3: {
                1: {
                    2: {
                        1: {
                            1: {'.': 'Formulation.md'},
                            2: {'.': 'Experimentation.md'},
                            3: {'.': 'Analysis.md'},
                            '.': 'Scientific Method.md',
                            '_': 'Hypothesis-Proof.md'
                        },
                        2: {
                            1: {'.': 'Old Models.md'},
                            2: {'.': 'Breakthroughs.md'},
                            3: {'.': 'New Models.md'},
                            '.': 'Paradigm Shifts.md',
                            '_': 'Static-Dynamic.md'
                        },
                        3: {
                            1: {'.': 'Tools.md', '_': 'Analog-Digital.md'},
                            2: {'.': 'Units.md', '_': 'Metric-Imperial.md'},
                            3: {'.': 'Standards.md', '_': 'Precision-Accuracy.md'},
                            '.': 'Measurement.md',
                            '_': 'Qualitative-Quantitative.md'
                        },
                        '.': 'Science of Science.md',
                        '_': 'Methodology-Epistemology.md'
                    }
                }
            }
        }
    }

    try:
        matching_entries = find_matching_entries(nested_dict)
        print("Found matching entries:", matching_entries)
    except ValueError as e:
        print(e)


    g = parallel_gen(None)
    print (next(g))
    print (next(g))
    print (next(g))
    print (next(g))
    print (next(g))
    print (next(g))
    print (next(g))
    print (next(g))

    for _ in range(10_000_000):
        print(next(store_and_yield_gen(g, "../___.txt")))