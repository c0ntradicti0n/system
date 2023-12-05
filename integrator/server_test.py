import time

import jsonpatch
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from gevent import monkey
from werkzeug.middleware.proxy_fix import ProxyFix

from lib.fatten_dict import fatten_dict

monkey.patch_all()

app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    port="5000",
    async_mode="gevent",
    # debug=True,
    # engineio_logger=True,
    # logger=True,
)

CORS(app, supports_credentials=True)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)


results = {}

room_memberships = {}


@socketio.on("join")
def on_join(hash_id):
    join_room(hash_id)

    # Add the user to the room_memberships
    if hash_id not in room_memberships:
        room_memberships[hash_id] = set()
    room_memberships[hash_id].add(hash_id)


@socketio.on("leave")
def on_leave(hash_id):
    leave_room(hash_id)

    # Remove the user from the room_memberships
    if hash_id in room_memberships:
        room_memberships[hash_id].discard(hash_id)
        if not room_memberships[hash_id]:
            del room_memberships[hash_id]  # Remove the room if it's empty


@socketio.on("set_state")
def set_state(hash_id, *args, **kwargs):
    res = results.get(hash_id, None)
    return {"status": "ok", "result": res}


def background_thread():
    while True:
        for hash_id in room_memberships.keys():
            if hash_id not in results:
                results[hash_id] = fatten_dict({})
            result = results[hash_id]

            new_result = fatten_dict(result)

            # generate a patch
            patch = jsonpatch.make_patch(result, new_result).patch

            # apply patch to result
            results[hash_id] = jsonpatch.apply_patch(result, patch)

            # Emit the updated result to the room
            socketio.emit("patch", {"status": "ok", "patch": patch}, room=hash_id)

        time.sleep(0.1)  # Adjust the sleep time as needed
        # print(room_memberships)


@socketio.on("connect")
def on_connect():
    print("Client connected")


@socketio.on("disconnect")
def on_disconnect():
    print("Client disconnected")


if __name__ == "__main__":
    socketio.start_background_task(background_thread)
    socketio.run(app, host="0.0.0.0", port=5000)
