import json
import os
import pickle
from hashlib import sha256
from pprint import pprint

import jsonpatch
from flask import Flask
from flask_socketio import SocketIO, emit
from main import update_triangle_graph
from reader import parse_text
from tree import Tree
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=None)

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)


# Sample in-memory state
class States:
    def __init__(self):
        self.states = {}

    @classmethod
    def path(cls, hash_id):
        return os.path.join("states/", f"{hash_id}.pkl")

    def get(self, hash_id, default=None):
        return self.states.get(hash_id, default)

    def __getitem__(self, hash_id):
        if hash_id in self.states:
            return self.states[hash_id]
        else:
            if hash_id.endswith("-text"):
                with open(self.path(hash_id), "rb") as f:
                    text = pickle.load(f)
                self.states[hash_id] = text
                print(f"loaded text {hash}")
                return text
            else:
                if os.path.exists(self.path(hash_id + "-text")):
                    tree, i = Tree.load_state(hash_id)
                    if not i:
                        with open(self.path(hash_id + "-text"), "rb") as f:
                            text = pickle.load(f)
                        inputs = parse_text(text)
                        if not inputs:
                            socketio.emit("error", "text is invalid")

                            print(f"invalid text for {hash}")

                            return None, -1
                        print(f"loaded tree from text {hash}")

                        tree, i = Tree(list(inputs.items())), 0
                    else:
                        print(f"loaded {i=} {hash_id} {tree=} ")
                    self.states[hash_id] = tree, i
                    print(f"loaded tree {hash}")

                    return tree, i
                print(f"loaded nothing {hash}")

                return None, None

    def set(self, hash_id, state):
        self.states[hash_id] = state

    def __setitem__(self, hash_id, state):
        if hash_id.endswith("-text"):
            with open(self.path(hash_id), "wb") as f:
                pickle.dump(state, f)
        self.states[hash_id] = state


states = States()


@socketio.on("update_state")
def handle_update(hash):
    print(f"update_state {hash}")
    old_state, i = states[hash]

    old_graph = Tree.max_score_triangle_subgraph(
        old_state.graph, return_start_node=True
    )
    new_graph = Tree.max_score_triangle_subgraph(
        update_triangle_graph(old_state, i, hash), return_start_node=True
    )
    print(f"{old_graph=}")
    print(f"{new_graph=}")

    new_state, i = (
        old_state,
        i + 1,
    )
    states[hash] = new_state, i

    patch = jsonpatch.make_patch(
        Tree.serialize_graph_to_structure(*old_graph),
        Tree.serialize_graph_to_structure(*new_graph),
    )
    serialized_patch = json.loads(patch.to_string())

    print(f"{serialized_patch=}")
    socketio.emit("state_patch", serialized_patch)


@socketio.on("set_init_state")
def handle_set_state(hash):
    old_state, i = states[hash]

    print(f"handle_set_state {hash}" + str(old_state))
    active_version = Tree.serialize_graph_to_structure(
        *Tree.max_score_triangle_subgraph(old_state.graph, return_start_node=True)
    )
    pprint(f"{active_version=}")
    socketio.emit("set_state", active_version)


@socketio.on("set_init_text")
def handle_set_text(text):
    print(f"handle_set_text '{text[:10]}...'")

    if not text.strip():
        return

    # Convert the received text into a pickled object
    pickled_obj = pickle.dumps(text)
    hash = sha256(pickled_obj).hexdigest()
    states[hash + "-text"] = text
    # Emit the hash back to the client
    emit("set_hash", hash)
    emit("initial_state")


@socketio.on("set_init_hash")
def handle_set_hash(hash_id):
    print(f"set_init_hash {hash_id}")
    if not hash_id:
        return
    print(f"set_init_hash {hash_id}")
    # Return the state associated with the hash
    text = states[hash_id + "-text"]
    emit("set_text", text)
    emit("initial_state")


if __name__ == "__main__":
    print("RUNNING FROM SCRATCH")
    socketio.run(app, host="0.0.0.0", port=5000)
