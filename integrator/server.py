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
def handle_update(hash_id):
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


@socket_event("set_initial_mods", "set_mods")
def handle_set_user_mods():
    print(f"handle_set_user_mods")
    return states.get_all()


@socket_event("delete_mod", "refresh")
def handle_delete_mod(hash):
    if not hash:
        print(f"handle_delete_mod hash id null!!! {hash=}")
        return
    assert re.match(r"[a-f0-9]{64}", hash)
    print(f"delete_mod", hash)
    del states[hash]
    return hash


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
    print(f"handle_get_meta {meta=}")
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
    print("RUNNING FROM PYTHON MAIN")
    socketio.run(app, host="0.0.0.0", port=5000)
