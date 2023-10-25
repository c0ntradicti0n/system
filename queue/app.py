from celery.result import AsyncResult
from flask import Flask, jsonify, request
from tasks import threerarchy

app = Flask(__name__)

@app.route('/threerarchy', methods=['POST'])
def enqueue_update_task():
    data = request.get_json()
    x = data.get('hash')
    task = threerarchy.delay(x)
    return jsonify({'task_id': task.id}), 202

@app.route('/task_status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task_result = AsyncResult(task_id, app=threerarchy.app)
    return jsonify({'task_id': task_id, 'status': task_result.status})

@app.route('/task_result/<task_id>', methods=['GET'])
def get_task_result(task_id):
    task_result = AsyncResult(task_id, app=threerarchy.app)
    if task_result.ready():
        return jsonify({'task_id': task_id, 'status': task_result.status, 'result': task_result.result})
    else:
        return jsonify({'task_id': task_id, 'status': task_result.status, 'result': 'Not ready'}), 202

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
