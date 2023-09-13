import json
import logging
import os
from pprint import pprint

import coloredlogs
import config
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_restx import Api, Namespace, Resource, reqparse
from helper import OutputLevel, get_from_nested_dict, nested_str_dict, tree

logging.basicConfig(level=logging.DEBUG)
coloredlogs.install(level="DEBUG")

app = Flask(__name__)
CORS(app)
api = Api(app)

toc = Namespace("toc", description="Logic Fractal TOC operations")
text = Namespace("text", description="Logic Fractal text operations")
search = Namespace("search", description="Search operations")

api.add_namespace(toc, path="/api/toc")
api.add_namespace(text, path="/api/text")
api.add_namespace(search, path="/api/search")

SECONDARY_BACKEND_URL = "http://service:5000/api/search"

parser = reqparse.RequestParser()
parser.add_argument('string', type=str, required=True, help='String to search', location='json')
parser.add_argument('filter_path', type=str, required=False, help='Filter path for the search', location='json')

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
                depth=os.environ.get("DEPTH", 4),
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
        if all(c in "123" for c in path):
            path = "/".join(path)
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
            # print(f"{file_dict=}")
            try:
                file_dict = {
                    str(k)[0]: (v.strip().strip("\n") if v else "")
                    if not isinstance(v, dict)
                    else list(v.values())[0]
                    for k, v in file_dict.items()
                }
            except Exception as e:
                raise e
            # pprint(f"{file_dict=}")
            return jsonify(nested_str_dict(file_dict))
        except FileNotFoundError:
            logging.error(f"File not found: {path}", exc_info=True)
            return {"error": "File not found"}, 404


@search.route("/")
class SearchProxy(Resource):
    @search.expect(parser)
    @search.doc("Proxy search to a secondary backend service using POST")
    def post(self):
        try:
            # Extract data from incoming POST request
            data = request.json
            string_to_search = data.get("string", "")
            args = parser.parse_args()
            string_to_search = args['string']
            filter_path = args['filter_path']

            if not string_to_search:
                return [], 200

            print(f"SEARCH STRING {string_to_search=}")
            print (f"FILTER PATH {filter_path=}")


            # Forward the request to the secondary backend service
            response = requests.post(f"{SECONDARY_BACKEND_URL}?filter_path={filter_path}", json=data)

            # Check if the request was successful
            response.raise_for_status()

            data = response.json()

            pprint(data)

            return data, 200

        except requests.RequestException as e:
            logging.error(f"Request to secondary backend failed: {e}", exc_info=True)
            return {"error": "Request to secondary backend failed"}, 500


if __name__ == "__main__":
    with app.app_context():
        result = LogicFractalText().get("")
        print(result)
        result = LogicFractalText().get("1")
        print(result)
    with app.app_context():
        print(LogicFractal().get(""))

    app.run(debug=os.environ.get("DEBUG", False))
