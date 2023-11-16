import json

from main import main

from helper import CustomEncoder

print(json.dumps(main("What is love?"), cls=CustomEncoder))
print(json.dumps(main("Heart"), cls=CustomEncoder))
print(json.dumps(main("gods"), cls=CustomEncoder))
print(json.dumps(main("Friends"), cls=CustomEncoder))
print(json.dumps(main("electron"), cls=CustomEncoder))
print(json.dumps(main("good"), cls=CustomEncoder))
print(json.dumps(main("correct"), cls=CustomEncoder))
print(json.dumps(main("okay"), cls=CustomEncoder))
print(json.dumps(main("hello"), cls=CustomEncoder))
