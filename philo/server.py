import logging
import os

from flask import Flask, request
from flask_cors import CORS
from flask_restx import Api, Namespace, Resource, reqparse

from lib.t import catchtime
from process import (generate_prompt, post_process_model_output,
                     process_user_input, commit)

from lib import config
from lib.helper import OutputLevel, tree

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
CORS(app)
api = Api(app)
tokens = dict([t.split(":") for t in os.environ.get("TOKENS", "").split(",")])
print(tokens)
philosopher = Namespace("philosopher", description="Philosopher API")


@api.route("/api/philo/init")
class Philosopher(Resource):
    # Define request parser for query parameters
    parser = reqparse.RequestParser()
    parser.add_argument("location", type=str, required=True, help="Location parameter")
    parser.add_argument("task", type=str, required=True, help="Task parameter")

    def post(self):
        location = request.json.get("path")
        task = request.json.get("task")
        location_path = "/".join(location)

        print(location, task)

        prompt, instruction = generate_prompt(
            base_path=config.system_path,
            task=task,
            location=location,
            info_radius=3,
            preset_output_level=OutputLevel.FILENAMES,
        )
        t = tree(
            basepath=config.system_path,
            startpath=location_path,
            sparse=True,
            info_radius=3,
            location="",
            pre_set_output_level=OutputLevel.FILENAMES,
            exclude=[".git", ".git.md", ".idea"],
            prefix_items=True,
            depth=100 if task == "text" else 2,
        )

        # Now you can use 'location' and 'task' in your logic
        # Example response using 'location' and 'task' values
        response = {
            "message": prompt + "\n\n" + instruction,
            "prompt": prompt.strip(),
            "instruction": instruction,
            "data": t,
        }
        return response


@api.route("/api/philo/commit")
class PhilosopherReply(Resource):
    def post(self):
        try:
            # Get user input from the request
            data = request.json.get("data")
            token = request.json.get("token")
            path = request.json.get("path")
            task = request.json.get("task")

            if not (tokens["*"] == token or tokens[path] == token):
                raise Exception(f"Invalid token provided for {path} {token}")

            # Pass user input to the provided code for processing
            # Modify this part to integrate with your provided code
            commit(data, task, path, token)

        except Exception as e:
            logging.error(e, exc_info=True)
            return {"error": str(e)}


@api.route("/api/philo/reply")
class PhilosopherSubmit(Resource):
    def post(self):
        try:
            # Get user input from the request
            hint = request.json.get("message")
            prompt = request.json.get("prompt")
            task = request.json.get("task")
            token = request.json.get("token")
            path = request.json.get("path")
            location_path = "/".join(path)
            instruction = request.json.get("instruction")
            print(f"token: {token}, path: {path}, tokens: {tokens}")
            if not (tokens["*"] == token or tokens[path] == token):
                raise Exception(f"Invalid token provided for {path}")

            t = tree(
                basepath=config.system_path,
                startpath=location_path,
                sparse=True,
                info_radius=3,
                location="",
                pre_set_output_level=OutputLevel.FILENAMES,
                exclude=[".git", ".git.md", ".idea"],
                prefix_items=True,
                depth=100 if task == "text" else 2,
            )
            # Pass user input to the provided code for processing
            # Modify this part to integrate with your provided code
            result = process_user_input(
                hint=hint, prompt=prompt, instruction=instruction, task=task
            )
            data = post_process_model_output(result, task, t, path)
            return {"reply": result, "data": data}
        except Exception as e:
            logging.error(e, exc_info=True)
            return {"error": str(e)}


if __name__ == "__main__":

    # Get user input from the request
    location = "1313"
    task = "toc"
    location_path = "/".join(location)

    print(location, task)
    with catchtime("tree"):
        t = tree(
            basepath=config.system_path,
            startpath=location_path,
            sparse=True,
            info_radius=3,
            location="",
            pre_set_output_level=OutputLevel.FILENAMES,
            exclude=(".git", ".git.md", ".idea"),
            prefix_items=True,
            depth=100 if task == "text" else 2,
        )
    with catchtime("generate_prompt"):
        prompt, instruction = generate_prompt(
            base_path=config.system_path,
            task=task,
            location=location,
            info_radius=3,
            preset_output_level=OutputLevel.FILENAMES,
        )


    # Now you can use 'location' and 'task' in your logic
    # Example response using 'location' and 'task' values
    response = {
        "message": prompt + "\n\n" + instruction,
        "prompt": prompt.strip(),
        "instruction": instruction,
        "data": t,
    }


    app.run(debug=True, host="0.0.0.0", port=5000)
