import logging
import os

import coloredlogs
import config
from flask import Flask, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource
from helper import OutputLevel, nested_str_dict, tree

logging.basicConfig(level=logging.DEBUG)
coloredlogs.install(level="DEBUG")

app = Flask(__name__)
CORS(app)
api = Api(app)
ns = api.namespace("api", description="Logic fractal operations")


@ns.route("/", defaults={"path": ""})
@ns.route("/<path:path>")
class LogicFractal(Resource):
    @ns.doc("Load triangle")
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
            )
            return jsonify(nested_str_dict(file_dict))
        except FileNotFoundError:
            logging.error(f"File not found: {path}", exc_info=True)
            return {"error": "File not found"}, 404


if __name__ == "__main__":
    with app.app_context():
        print(LogicFractal().get(""))

        print(LogicFractal().get("/system/1/3/1/2"))

    app.run(debug=os.environ.get("DEBUG", False))
