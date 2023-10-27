import logging
import os

import redis
from celery.result import AsyncResult
from flask import Flask, jsonify, request
from tasks import threerarchy

app = Flask(__name__)
redis_client = redis.StrictRedis(host="redis", port=6379, db=1)


@app.route("/threerarchy", methods=["POST"])
def enqueue_update_task():
    data = request.get_json()
    hash_id = data.get("hash")

    # Define the file path based on the hash_id
    file_path = f"./threerarchy_{hash_id}.txt"

    try:
        # If the file exists, read the task_id from it
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_task = f.read().strip()

            if existing_task:
                result = AsyncResult(existing_task, app=threerarchy.app)
                if result.status not in ["SUCCESS", "FAILURE", "REVOKED"]:
                    # If the task is still running (or pending, or retrying), return its task_id
                    return jsonify({"task_id": existing_task}), 202
    except:
        logging.error("Error checking for existing task", exc_info=True)

    # If no existing task or if the existing task has completed/failed, start a new task
    task = threerarchy.delay(hash_id)

    # Save the new task_id to the file
    with open(file_path, 'w') as f:
        f.write(task.id)

    return jsonify({"task_id": task.id}), 202

@app.route("/task_status/<task_id>", methods=["GET"])
def get_task_status(task_id):
    task_result = AsyncResult(task_id, app=threerarchy.app)
    return jsonify({"task_id": task_id, "status": task_result.status})


@app.route("/task_result/<task_id>", methods=["GET"])
def get_task_result(task_id):
    task_result = AsyncResult(task_id, app=threerarchy.app)
    try:
        if task_result.ready():
            return jsonify(
                {
                    "task_id": task_id,
                    "status": task_result.status,
                    "result": task_result.result,
                }
            )
        else:
            return (
                jsonify(
                    {
                        "task_id": task_id,
                        "status": task_result.status,
                        "result": "Not ready",
                    }
                ),
                202,
            )
    except Exception as e:
        print(f"Error getting task result {task_id=} {task_result=}")
        return (
            jsonify(
                {
                    "task_id": task_id,
                    "status": task_result.status,
                    "result": "Not ready",
                }
            ),
            202,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
