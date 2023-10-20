import os
import pickle

from lib.ls import list_files_with_regex
from reader import parse_text
from tree import Tree


# Sample in-memory state


class States:
    def __init__(self):
        self.states = {}

    @classmethod
    def path(cls, hash_id):
        return os.path.join("states/", f"{hash_id}.pkl")

    def get(self, hash_id, default=None):
        return self.states.get(hash_id, default)

    @staticmethod
    def get_all():
        return list_files_with_regex("states/", "(?P<hash>.*)\-text.pkl")

    def __getitem__(self, hash_id):
        if hash_id in self.states:
            return self.states[hash_id]
        else:
            if hash_id.endswith("-text"):
                with open(self.path(hash_id), "rb") as f:
                    text = pickle.load(f)
                self.states[hash_id] = text
                print(f"loaded text {hash_id}")
                return text
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
                    self.states[hash_id] = tree, i
                    print(f"loaded tree {hash_id}")

                    return tree, i
                print(f"loaded nothing {hash_id}")

                return None, None

    def set(self, hash_id, state):
        self.states[hash_id] = state

    def __setitem__(self, hash_id, state):
        if hash_id.endswith("-text"):
            with open(self.path(hash_id), "wb") as f:
                pickle.dump(state, f)
        self.states[hash_id] = state


states = States()