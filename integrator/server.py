import gc
import re
import time

import requests
from gevent import monkey

monkey.patch_all()
import pickle
from functools import wraps
from hashlib import sha256

import gevent
from flask import Flask, copy_current_request_context
from flask_socketio import SocketIO
from states import states
from werkzeug.middleware.proxy_fix import ProxyFix

from integrator.tree import Tree
from integrator.serialize import serialize_graph_to_structure

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")
import atexit
import sys


class ExitHooks(object):
    def __init__(self):
        self.exit_code = None
        self.exception = None

    def hook(self):
        self._orig_exit = sys.exit
        self._orig_exc_handler = self.exc_handler
        sys.exit = self.exit
        sys.excepthook = self.exc_handler

    def exit(self, code=0):
        self.exit_code = code
        self._orig_exit(code)

    def exc_handler(self, exc_type, exc, *args):
        self.exception = exc
        self._orig_exc_handler(self, exc_type, exc, *args)


def exit_handler():
    if hooks.exit_code is not None:
        print("death by sys.exit(%d)" % hooks.exit_code)
    elif hooks.exception is not None:
        print("death by exception: %s" % hooks.exception)
    else:
        print("natural death")


hooks = ExitHooks()
hooks.hook()
atexit.register(exit_handler)
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


@socket_event("update_state", "set_task_id")
def update_state(hash_id):
    if not hash_id:
        print(f"handle_update hash id null!!! {hash_id=}")
        return
    gc.collect()

    # trigger celery
    print(f"handle_update {hash_id}")
    result = requests.post(
        "http://queue:5000/threerarchy", json={"hash": hash_id}
    ).json()
    print(f"result {result}")
    return result["task_id"]


@socket_event("patch_poll", "patch_receive")
def handle_trigger_celery(task_id):
    try:
        result = requests.get(f"http://queue:5000/task_result/{task_id}").json()
    except Exception as e:
        print(f"error getting task result {task_id=}" + str(e))
        return []
    print(f"handle_trigger_celery {result=}")
    return result


@socket_event("get_mods", "set_mods")
def get_mods():
    print(f"handle_set_user_mods")
    return states.get_all()


@socket_event("delete_mod", "refresh")
def delete_mod(hash):
    if not hash:
        print(f"handle_delete_mod hash id null!!! {hash=}")
        return
    assert re.match(r"[a-f0-9]{64}", hash)
    print(f"delete_mod", hash)
    del states[hash]
    return hash


@socket_event("reset_mod", "refresh")
def reset_mod(hash):
    if not hash:
        print(f"handle_reset_mod hash id null!!! {hash=}")
        return
    assert re.match(r"[a-f0-9]{64}", hash)
    print(f"reset_mod", hash)
    states.reset(hash)
    return hash


@socket_event("get_state", "set_state")
def handle_set_state(hash_id):
    print(f"handle_set_state {hash_id}")

    old_state, i = states[hash_id]

    active_version = serialize_graph_to_structure(
        *old_state.max_score_triangle_subgraph(old_state.graph, return_start_node=True)
    )
    return active_version


@socket_event("get_params", "set_params")
def get_params(hash_id):
    print(f"get_params {hash_id}")
    if not hash_id.strip():
        return None
    try:
        params = states[hash_id + "-params"]
        return params

    except Exception as e:
        print(f"error getting params {hash_id=}" + str(e))
        return None


@socket_event("save_params", "set_params")
def save_params(params, hash_id):
    print(f"save_params {params} {hash_id}")

    states[hash_id + "-params"] = params
    new_params = states[hash_id + "-params"]

    handle_set_state(hash_id)

    print(f"new params {new_params}")

    return new_params


@socket_event("save_text", "set_hash")
def get_text(text):
    print(f"save_text '{text[:10]}...'")

    if not text.strip():
        return

    # Convert the received text into a pickled object
    pickled_obj = pickle.dumps(text)
    hash_id = sha256(pickled_obj).hexdigest()
    states[hash_id + "-text"] = text
    return hash_id


@socket_event("get_text", "set_hash")
def get_text(text):
    print(f"get_text '{text[:10]}...'")

    if not text.strip():
        return

    # Convert the received text into a pickled object
    pickled_obj = pickle.dumps(text)
    hash_id = sha256(pickled_obj).hexdigest()
    states[hash_id + "-text"] = text
    return hash_id


@socket_event("get_meta", "set_meta")
def get_meta(hash_id):
    print(f"get_meta {hash_id=} ")

    if not hash_id.strip():
        return

    meta = states[hash_id + "-meta"]
    print(f"get_meta {meta=}")
    return meta


@socket_event("save_meta")
def save_meta(hash_id, meta):
    print(f"save_meta {hash_id=} '{meta[:10]}...'")

    if not meta.strip():
        return

    states[hash_id + "-meta"] = meta
    return hash_id


@socket_event("get_hash", "set_text")
def get_hash(hash_id):
    print(f"get_hash {hash_id}")
    if not hash_id:
        return
    print(f"get_hash {hash_id}")

    text = states[hash_id + "-text"]
    return text[:1000]


if __name__ == "__main__":
    print("RUNNING FROM PYTHON MAIN")
    socketio.run(app, host="0.0.0.0", port=5000)
