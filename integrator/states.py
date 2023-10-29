import os
import pickle
from shutil import rmtree

from integrator.reader import parse_text
from integrator.tree import Tree
from lib.ls import list_files_with_regex

# Sample in-memory state


class States:
    def __init__(self):
        pass

    @classmethod
    def path(cls, hash_id):
        return os.path.join("states/", f"{hash_id}.pkl")

    @staticmethod
    def get_all():
        matched_files = list_files_with_regex("states/", "(?P<hash>.*)\-text.pkl")

        results = []
        for matched_file in matched_files:
            hash_value = matched_file["hash"]
            meta_filename = os.path.join("states", f"{hash_value}-meta.pkl")

            if os.path.exists(meta_filename):
                with open(meta_filename, "rb") as meta_file:
                    metadata = pickle.load(meta_file)
            else:
                metadata = None

            matched_file["meta"] = metadata

            results.append(matched_file)
        return results

    def __getitem__(self, hash_id):
        if hash_id.endswith("-text"):
            with open(self.path(hash_id), "rb") as f:
                text = pickle.load(f)
            print(f"loaded text {hash_id}")
            return text
        elif hash_id.endswith("-meta"):
            with open(self.path(hash_id), "rb") as f:
                meta = pickle.load(f)
            print(f"loaded meta {hash_id}")
            return meta
        else:
            if os.path.exists(self.path(hash_id + "-text")):
                tree, i = Tree.load_state(hash_id)
                if not i:
                    with open(self.path(hash_id + "-text"), "rb") as f:
                        text = pickle.load(f)
                    inputs = parse_text(text)
                    if not inputs:
                        print(f"invalid text for {hash_id}")
                        return None, -1
                    print(f"loaded tree from text {hash_id}")

                    tree, i = Tree(list(inputs.items())), 0
                else:
                    print(f"loaded {i=} {hash_id} {tree=} ")
                print(f"loaded tree {hash_id}")

                return tree, i
            print(f"loaded nothing {hash_id}")

            return None, None

    def __setitem__(self, hash_id, state):
        if hash_id.endswith("-text"):
            with open(self.path(hash_id), "wb") as f:
                pickle.dump(state, f)
        elif hash_id.endswith("-meta"):
            with open(self.path(hash_id), "wb") as f:
                pickle.dump(state, f)
        else:
            tree, i = state
            tree.save_state(i, hash_id)
            print(f"saved {i=} {hash_id} {tree=} ")

    def __delitem__(self, key):
        rmtree(self.path(key), ignore_errors=True)
        try:
            os.unlink(self.path(key + "-text"))
        except FileNotFoundError:
            pass
        try:
            os.unlink(self.path(key + "-meta"))
        except FileNotFoundError:
            pass


states = States()
