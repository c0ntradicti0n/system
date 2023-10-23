from gevent import monkey

monkey.patch_all()
import gevent
from states import states
from functools import wraps
import json
import pickle
from hashlib import sha256

import jsonpatch
from flask import Flask, copy_current_request_context
from flask_socketio import SocketIO, emit
from main import update_triangle_graph
from tree import Tree
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)


def socket_event(event_name, emit_event_name=None):
    def decorator(f):
        @socketio.on(event_name)
        @wraps(f)
        def wrapper(*args, **kwargs):
            @copy_current_request_context
            def subfunction(*args, **kwargs):
                result = f(*args, **kwargs)
                if emit_event_name:
                    socketio.emit(emit_event_name, result)

            gevent.spawn(subfunction, *args, **kwargs)

        return wrapper

    return decorator


@socket_event("update_state", "state_patch")
def handle_update(hash_id):
    print(f"update_state {hash_id}")
    old_state, i = states[hash_id]

    old_graph = Tree.max_score_triangle_subgraph(
        old_state.graph, return_start_node=True
    )

    new_graph = Tree.max_score_triangle_subgraph(
        update_triangle_graph(old_state, i, hash_id), return_start_node=True
    )

    new_state, i = (
        old_state,
        i + 1,
    )
    states[hash_id] = new_state, i

    try:
        patch = jsonpatch.make_patch(
            Tree.serialize_graph_to_structure(*old_graph),
            Tree.serialize_graph_to_structure(*new_graph),
        )
        serialized_patch = json.loads(patch.to_string())
    except:
        print(f"error making patch {old_graph=} {new_graph=}")
        serialized_patch = []

    return serialized_patch


@socket_event("set_initial_mods", "set_mods")
def handle_set_user_mods():
    print (f"handle_set_user_mods")
    return states.get_all()

@socket_event("set_init_state", "set_state")
def handle_set_state(hash_id):
    print(f"handle_set_state {hash_id}")

    old_state, i = states[hash_id]

    active_version = Tree.serialize_graph_to_structure(
        *Tree.max_score_triangle_subgraph(old_state.graph, return_start_node=True)
    )
    return active_version


@socket_event("set_init_text", "set_hash")
def handle_set_text(text):
    print(f"handle_set_text '{text[:10]}...'")

    if not text.strip():
        return

    # Convert the received text into a pickled object
    pickled_obj = pickle.dumps(text)
    hash_id = sha256(pickled_obj).hexdigest()
    states[hash_id + "-text"] = text
    return hash_id

@socket_event("get_meta", "set_meta")
def handle_get_meta(hash_id):
    print(f"handle_get_meta {hash_id=} ")

    if not hash_id.strip():
        return

    meta = states[hash_id + "-meta"]
    print (f"handle_get_meta {meta=}")
    return meta

@socket_event("set_init_meta")
def handle_set_meta(hash_id, meta):
    print(f"handle_set_meta {hash_id=} '{meta[:10]}...'")

    if not meta.strip():
        return

    states[hash_id + "-meta"] = meta
    return hash_id


@socket_event("set_init_hash", "set_text")
def handle_set_hash(hash_id):
    print(f"set_init_hash {hash_id}")
    if not hash_id:
        return
    print(f"set_init_hash {hash_id}")

    text = states[hash_id + "-text"]
    return text[:1000]


if __name__ == "__main__":
    print("RUNNING FROM SCRATCH")
    socketio.run(app, host="0.0.0.0", port=5000)
