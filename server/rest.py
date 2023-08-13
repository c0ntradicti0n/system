import logging
import os
from pprint import pprint

import coloredlogs
import config
from flask import Flask, jsonify
from flask_cors import CORS
from flask_restx import Api, Namespace, Resource
from helper import OutputLevel, get_from_nested_dict, nested_str_dict, tree

logging.basicConfig(level=logging.DEBUG)
coloredlogs.install(level="DEBUG")

app = Flask(__name__)
CORS(app)
api = Api(app)

toc = Namespace("toc", description="Logic Fractal TOC operations")
text = Namespace("text", description="Logic Fractal text operations")

api.add_namespace(toc, path="/api/toc")
api.add_namespace(text, path="/api/text")


@toc.route("/", defaults={"path": ""})
@toc.route("/<path:path>")
class LogicFractal(Resource):
    @toc.doc("Load Table of contents")
    def get(self, path):
        try:
            file_dict = tree(
                basepath=config.system_path,
                startpath=path,
                format="json",
                keys=path.split("/"),
                info_radius=1,
                exclude=[".git", ".git.md", ".gitignore", ".DS_Store", ".idea"],
                pre_set_output_level=OutputLevel.FILENAMES,
                prefix_items=True,
                depth=os.environ.get("DEPTH", 4)
            )
            return jsonify(nested_str_dict(file_dict))
        except FileNotFoundError:
            logging.error(f"File not found: {path}", exc_info=True)
            return {"error": "File not found"}, 404


@text.route("/", defaults={"path": ""})
@text.route("/<path:path>")
class LogicFractalText(Resource):
    @text.doc("Get text for things")
    def get(self, path):
        try:
            file_dict = tree(
                basepath=config.system_path,
                startpath=path,
                format="json",
                keys=path.split("/"),
                info_radius=1,
                exclude=[".git", ".git.md", ".gitignore", ".DS_Store", ".idea"],
                pre_set_output_level=OutputLevel.FILE,
                depth=0,
                prefix_items=True,
            )

            if path.__len__():
                file_dict = get_from_nested_dict(file_dict, path.split("/"))
            #print(f"{file_dict=}")
            try:
                file_dict = {
                    str(k)[0]: (v.strip() if v else "")
                    if not isinstance(v, dict)
                    else list(v.values())[0]
                    for k, v in file_dict.items()
                }
            except Exception as e:
                raise e
            #pprint(f"{file_dict=}")
            return jsonify(nested_str_dict(file_dict))
        except FileNotFoundError:
            logging.error(f"File not found: {path}", exc_info=True)
            return {"error": "File not found"}, 404


if __name__ == "__main__":
    with app.app_context():
        result = LogicFractalText().get("")
        print(result)
        result = LogicFractalText().get("1")
        print(result)
    with app.app_context():
        print(LogicFractal().get(""))

    app.run(debug=os.environ.get("DEBUG", False))
