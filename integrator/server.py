import random
import time

import jsonpatch
from flask import Flask, jsonify
from flask_socketio import SocketIO, emit, send
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
socketio = SocketIO(
    app, cors_allowed_origins="*", async_mode=None, logger=True, engineio_logger=True
)

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Sample in-memory state
states = {"hash123": {"data": "initial data"}}


@app.route("/state/<hash_id>")
def get_state(hash_id):
    return jsonify(states.get(hash_id, {}))


@socketio.on("connect")
def handle_connect():
    # Send initial state to the client after connection
    socketio.emit("initial_state", states["hash123"])


@socketio.on("update_state")
def handle_update():
    # Update the state and send patches to clients
    old_state = states["hash123"]

    update_state_continuously()
    new_state = states["hash123"]

    patch = jsonpatch.make_patch(old_state, new_state)
    serialized_patch = patch.to_string()

    states["hash123"] = new_state
    socketio.emit("state_patch", serialized_patch)


def update_state_continuously():
    random_data = {
        "data": f"random data {random.randint(1, 1000)}",
        "nested": {"key": f"value {random.randint(1, 1000)}"},
    }
    states["hash123"] = random_data

    time.sleep(2.5)
    print("Updated state")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
