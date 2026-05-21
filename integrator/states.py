import logging
import os
import pickle
import shutil

import regex as re

from reader import parse_text
from tree import Tree
from lib.ls import list_files_with_regex

# State files live on the shared PVC so all pods (integrator, worker) can access them.
# Built-in default states are baked into the Docker image at /app/states/ and are
# copied here on first init if the directory is empty.
STATE_DIR = "/system/states/"
BUILTIN_DIR = "/app/states/"


class States:
    def __init__(self):
        os.makedirs(STATE_DIR, exist_ok=True)
        self._seed_defaults()

    @classmethod
    def _seed_defaults(cls):
        """Copy built-in states from Docker image into shared PVC if empty."""
        if not os.path.isdir(BUILTIN_DIR):
            return
        existing = {f for f in os.listdir(STATE_DIR) if f.endswith(".pkl")}
        for fname in os.listdir(BUILTIN_DIR):
            if fname.endswith(".pkl") and fname not in existing:
                shutil.copy2(os.path.join(BUILTIN_DIR, fname), STATE_DIR)

    @classmethod
    def path(cls, hash_id):
        return os.path.join(STATE_DIR, f"{hash_id}.pkl")

    @staticmethod
    def get_all():
        matched_files = list_files_with_regex(STATE_DIR, r"(?P<hash>.*)\-text\.pkl")
        results = []
        for mf in matched_files:
            hash_value = mf["hash"]
            meta_path = os.path.join(STATE_DIR, f"{hash_value}-meta.pkl")
            metadata = None
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    metadata = pickle.load(f)
            results.append({"hash": hash_value, "meta": metadata, **mf})
        return results

    def __getitem__(self, hash_id):
        p = self.path(hash_id)
        if hash_id.endswith("-params"):
            return pickle.load(open(p, "rb")) if os.path.exists(p) else {}
        if hash_id.endswith("-text"):
            return pickle.load(open(p, "rb"))
        if hash_id.endswith("-meta"):
            return pickle.load(open(p, "rb")) if os.path.exists(p) else ""
        # Main state: tree + iteration
        text_path = self.path(hash_id + "-text")
        if os.path.exists(text_path):
            try:
                tree, i = Tree.load_state(hash_id)
            except Exception:
                logging.error(f"Corrupt file loading {hash_id}", exc_info=True)
                i = None
            if not i:
                text = pickle.load(open(text_path, "rb"))
                inputs = parse_text(text)
                if not inputs:
                    print(f"invalid text for {hash_id}")
                    return None, -1
                if isinstance(inputs, list):
                    tree, i = Tree(list(enumerate(inputs))), 0
                else:
                    tree, i = Tree(list(inputs.items())), 0
            return tree, i
        return None, None

    def __setitem__(self, hash_id, state):
        if hash_id.endswith(("-params", "-text", "-meta")):
            with open(self.path(hash_id), "wb") as f:
                pickle.dump(state, f)
            os.chmod(self.path(hash_id), 0o777)
            return
        tree, i = state
        tree.save_state(i, hash_id)
        print(f"saved {i=} {hash_id} {tree=}")
        # keep only the 3 most recent checkpoints
        state_dir = os.path.join(STATE_DIR, hash_id)
        if os.path.isdir(state_dir):
            files = []
            for fname in os.listdir(state_dir):
                m = re.match(r"tree_state_(?P<i>\d+)\.pkl", fname)
                if m:
                    files.append((int(m.group("i")), fname))
            for _, fname in sorted(files, reverse=True)[3:]:
                os.unlink(os.path.join(state_dir, fname))

    def __delitem__(self, key):
        shutil.rmtree(os.path.join(STATE_DIR, key), ignore_errors=True)
        for suffix in ("-text", "-meta"):
            try:
                os.unlink(self.path(key + suffix))
            except FileNotFoundError:
                pass

    def reset(self, hash_id):
        shutil.rmtree(os.path.join(STATE_DIR, hash_id), ignore_errors=True)


states = States()

if __name__ == "__main__":
    v, i = states["1cb10b667a508d3585b760a6ef9e8567b38af5421469a767678007a8a525d911"]
    states["1cb10b667a508d3585b760a6ef9e8567b38af5421469a767678007a8a525d911"] = v, i
