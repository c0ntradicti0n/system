from addict import Dict
from ruamel import yaml

with open("models.yml") as f:
    models = yaml.load(f)


def gen_config():
    import lib.default_config as config

    for key, model in models["models"].items():
        model = Dict(model)

        MODEL_DIR = f"models/{key}/"
        model.MODEL_DIR = MODEL_DIR

        MODEL_PATH = f"model.pth"
        model.MODEL_PATH = MODEL_PATH

        OPTIMIZER_PATH = "optimizer.pth"
        model.OPTIMIZER_PATH = OPTIMIZER_PATH

        model.system_path = config.system_path

        yield model
