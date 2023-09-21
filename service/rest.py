import json
import logging
import os

import coloredlogs
import flask
from flask import Flask
from flask_cors import CORS
from flask_restx import Api, Namespace, Resource, reqparse
from helper import CustomEncoder
from main import search as semantic_search

logging.basicConfig(level=logging.DEBUG)
coloredlogs.install(level="DEBUG")

flask.json.encoder = CustomEncoder

app = Flask(__name__)

CORS(app)
api = Api(app)

search = Namespace("search", description="Search operations")

api.add_namespace(search, path="/api/search")

parser = reqparse.RequestParser()
parser.add_argument(
    "string", type=str, required=True, help="String to search", location="json"
)
parser.add_argument(
    "filter_path",
    type=str,
    required=False,
    help="Filter path for the search",
    location="json",
)


@search.route("/")
class SearchProxy(Resource):
    @search.expect(parser)
    @search.doc("Proxy search to a secondary backend service using POST")
    def post(self):
        try:
            args = parser.parse_args()
            string_to_search = args["string"]
            filter_path = args["filter_path"]

            if not string_to_search:
                return {"error": "Search string is missing"}, 400
            print(f"SEARCH STRING {string_to_search=}")
            print(f"FILTER PATH {filter_path=}")

            response = semantic_search(
                string_to_search, top_k=5, filter_path=filter_path
            )
            print(response)

            return app.response_class(
                response=json.dumps(response, cls=CustomEncoder),
                status=200,
                mimetype="application/json",
            )

        except requests.RequestException as e:
            logging.error(f"Request to secondary backend failed: {e}", exc_info=True)
            return {"error": "Request to secondary backend failed"}, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("DEBUG", False))
