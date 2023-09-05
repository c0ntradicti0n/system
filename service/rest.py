import json
import logging
import os
from pprint import pprint

import coloredlogs
import config
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_restx import Api, Namespace, Resource
from helper import (CustomEncoder, OutputLevel, get_from_nested_dict,
                    nested_str_dict, tree)
from main import main

logging.basicConfig(level=logging.DEBUG)
coloredlogs.install(level="DEBUG")

app = Flask(__name__)
CORS(app)
api = Api(app)

search = Namespace("search", description="Search operations")

api.add_namespace(search, path="/api/search")


@search.route("/")
class SearchProxy(Resource):
    @search.doc("Proxy search to a secondary backend service using POST")
    def post(self):
        try:
            # Extract data from incoming POST request
            data = request.json
            string_to_search = data.get("string", "")
            if not string_to_search:
                return json.dumps([]), 200
            print("SEARCH for ", string_to_search)

            # Forward the request to the secondary backend service
            response = main(string_to_search)

            return json.dumps(response, cls=CustomEncoder), 200

        except requests.RequestException as e:
            logging.error(f"Request to secondary backend failed: {e}", exc_info=True)
            return {"error": "Request to secondary backend failed"}, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("DEBUG", False))
