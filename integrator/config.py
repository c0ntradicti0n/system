import os

MODEL_PATH = "model.pth"
OPTIMIZER_PATH = "optimizer.pth"
try:
    system_path = os.environ["SYSTEM"]
except KeyError:
    system_path = 's.environ["SYSTEM"] not set'
embedding_dim = 1024
batch_size = 64
n_samples = 4
n_classes = 5
