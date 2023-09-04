import json

from helper import CustomEncoder
from main import main

print(json.dumps(main("What is love?"), cls=CustomEncoder))
