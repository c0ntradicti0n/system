import os

try:
    system_path = os.environ["SYSTEM"]
except KeyError:
    system_path = 's.environ["SYSTEM"] not set'


try:
    system_path = os.environ["SYSTEM"]
except KeyError:
    system_path = 's.environ["SYSTEM"] not set'
embedding_dim = 1024
