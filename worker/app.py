import logging
import os
import time

import redis
from celery.result import AsyncResult
from flask import Flask, jsonify, request
from tasks import threerarchy

app = Flask(__name__)
redis_client = redis.StrictRedis(host="redis", port=6379, db=1)
not_flushed = True
while not_flushed:
    try:
        redis_client.flushdb()
        not_flushed = False

    except:
        print("redis not ready")
        time.sleep(1)
os.system("rm -rf /tmp/threerarchy")
os.mkdir("/tmp/threerarchy")


@app.route("/threerarchy", methods=["POST"])
def enqueue_update_task():
    data = request.get_json()
    hash_id = data.get("hash")
    task = threerarchy.delay(hash_id)
    return jsonify(task.id), 202


@app.route("/task_status/<task_id>", methods=["GET"])
def get_task_status(task_id):
    task_result = AsyncResult(task_id, app=threerarchy.app)
    return jsonify({"task_id": task_id, "status": task_result.status})


@app.route("/task_result/<task_id>", methods=["GET"])
def get_task_result(task_id):
    task_result = AsyncResult(task_id, app=threerarchy.app)
    if task_result.status == "FAILURE":
        task_result.forget()
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
