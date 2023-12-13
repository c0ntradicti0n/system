from gevent import monkey

monkey.patch_all()
import logging
import pickle
import re
import time
from hashlib import sha256

import requests
from flask import Flask
from flask_socketio import SocketIO, join_room, leave_room
from states import states
from werkzeug.middleware.proxy_fix import ProxyFix

from integrator.serialize import serialize_graph_to_structure
from lib.t import catchtime

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

results = {}

room_memberships = {}


@socketio.on("join")
def on_join(hash_id):
    join_room(hash_id)
    if not hash_id or not hash_id.strip():
        mods = states.get_all()
        socketio.emit("set_mods", mods, room=hash_id)
        return

    # Add the user to the room_memberships
    if hash_id not in room_memberships:
        room_memberships[hash_id] = set()
    room_memberships[hash_id].add(hash_id)

    t, i = states[hash_id]
    progress = t.progress()

    state = get_new_best_subgraph(progress, t)

    socketio.emit("set_state", state, room=hash_id)

    meta = states[hash_id + "-meta"]
    text = states[hash_id + "-text"]
    params = states[hash_id + "-params"]
    socketio.emit("set_meta", meta, room=hash_id)
    socketio.emit("set_text", text, room=hash_id)
    socketio.emit("set_params", params, room=hash_id)
    socketio.emit("set_progress", progress, room=hash_id)


def get_new_best_subgraph(progress, t):
    if "syn_1_syn_2" in progress:
        with catchtime("serialize"):
            state = serialize_graph_to_structure(
                *t.max_score_triangle_subgraph(t.graph, return_start_node=True)
            )
    else:
        state = {}
    return state


@socketio.on("leave")
def on_leave(hash_id):
    leave_room(hash_id)

    print(f"leave {hash_id=}")

    # Remove the user from the room_memberships
    if hash_id in room_memberships:
        room_memberships[hash_id].discard(hash_id)
        if not room_memberships[hash_id]:
            del room_memberships[hash_id]  # Remove the room if it's empty


tasks = {}
running_serializations = {}


def background_thread():
    while True:
        try:
            active_rooms = list(room_memberships.keys())
            for hash_id in active_rooms:
                current_state, i = states[hash_id]
                if current_state.finished():
                    continue

                if not hash_id in tasks or not tasks[hash_id]:
                    try:
                        tasks[hash_id] = requests.post(
                            "http://queue:5000/threerarchy", json={"hash": hash_id}
                        ).json()
                    except Exception as e:
                        logging.error("Error starting task", exc_info=True)

                try:
                    result = requests.get(
                        f"http://queue:5000/task_result/{tasks[hash_id]}"
                    ).json()

                except Exception as e:
                    logging.error("Error getting task result", exc_info=True)
                    print(
                        requests.get(f"http://queue:5000/task_result/{tasks[hash_id]}")
                    )
                    continue

                if result["status"] == "SUCCESS":
                    print(f"task {hash_id} ({tasks[hash_id]} finished")

                    tasks[hash_id] = requests.post(
                        "http://queue:5000/threerarchy", json={"hash": hash_id}
                    ).json()

                    print(f"new result {tasks[hash_id]}: {result}")

                    # Emit the updated result to the room
                    socketio.emit(
                        "set_state",
                        result["result"]["state"],
                        room=hash_id,
                    )
                    socketio.emit(
                        "set_i",
                        i,
                        room=hash_id,
                    )
                    socketio.emit(
                        "set_progress",
                        result["result"]["progress"],
                        room=hash_id,
                    )
                    socketio.emit(
                        "set_status",
                        result["result"]["status"],
                        room=hash_id,
                    )

                else:
                    print(
                        f"task {hash_id} ({tasks[hash_id]} still running {result['status']}"
                    )

        except Exception as e:
            logging.error("Error in background thread", exc_info=True)

        time.sleep(0.5)
        # print(f"tick {active_rooms=}")


@socketio.on("delete_mod")
def delete_mod(hash_id):
    if not hash_id:
        print(f"handle_delete_mod hash id null!!! {hash_id=}")
        return
    assert re.match(r"[a-f0-9]{64}", hash_id)
    print(f"delete_mod", hash_id)
    del states[hash_id]
    return states.get_all()


@socketio.on("reset")
def reset_mod(hash_id):
    if not hash_id:
        print(f"handle_reset_mod hash id null!!! {hash_id=}")
        return
    assert re.match(r"[a-f0-9]{64}", hash_id)
    print(f"reset_mod", hash_id)
    states.reset(hash_id)
    return hash_id


@socketio.on("get_state")
def get_state(hash_id):
    print(f"get_state {hash_id}")

    old_state, i = states[hash_id]
    with catchtime("serialize"):
        serialized = serialize_graph_to_structure(
            *old_state.max_score_triangle_subgraph(
                old_state.graph, return_start_node=True
            )
        )

    return serialized


@socketio.on("get_params")
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


@socketio.on("save_params")
def save_params(params, hash_id):
    print(f"save_params {params} {hash_id}")

    states[hash_id + "-params"] = params
    new_params = states[hash_id + "-params"]

    t, i = states[hash_id]
    progress = t.progress()

    state = get_new_best_subgraph(progress, t)
    socketio.emit("set_state", state, room=hash_id)
    socketio.emit("set_params", new_params, room=hash_id)


@socketio.on("save_text_meta")
def get_text_meta(text, meta):
    print(f"save_text_meta '{text[:10]}...' '{str(meta)}'")

    if not text.strip() or not meta.strip():
        return

    # Convert the received text into a pickled object
    pickled_obj = pickle.dumps(text)
    hash_id = sha256(pickled_obj).hexdigest()
    states[hash_id + "-text"] = text
    states[hash_id + "-meta"] = meta

    mods = states.get_all()
    socketio.emit("set_mods", mods)

    return hash_id


@socketio.on("get_text")
def get_text(hash_id):
    if not hash_id:
        return
    print(f"get_text {hash_id}")

    text = states[hash_id + "-text"]
    return text[:1000]


@socketio.on("get_full_text")
def get_full_text(hash_id):
    print(f"get_full_text {hash_id}")

    if not hash_id:
        return
    print(f"get_full_text {hash_id}")

    text = states[hash_id + "-text"]
    return text


@socketio.on("get_meta", "set_meta")
def get_meta(hash_id):
    print(f"get_meta {hash_id=} ")

    if not hash_id.strip():
        return

    meta = states[hash_id + "-meta"]
    print(f"get_meta {meta=}")
    return meta


socketio.start_background_task(background_thread)

if __name__ == "__main__":
    print("RUNNING FROM PYTHON MAIN")

    socketio.run(app, host="0.0.0.0", port=5000)
