import os
import re

from addict import Dict
from ruamel.yaml import YAML

yaml = YAML(typ="rt")


def get_models_root_dir():
    return os.environ.get("MODELS_CONFIG", "../classifier/models/")


def get_models_config_path():
    return os.path.join(get_models_root_dir(), "models.yml")


with open(get_models_config_path(), "r") as f:
    models = yaml.load(f)


def get_model_config(model_name):
    model = models["models"][model_name]
    model = configure_model(model_name, model)
    return model


def gen_config():
    import lib.default_config as config

    for key in models["models"]:
        model = models["models"][key]

        model = configure_model(key, model)

        model.system_path = config.system_path

        yield model


def configure_model(key, model):
    model = Dict(model)
    MODEL_DIR = f"/{key}/"
    model.MODEL_DIR = os.path.abspath(get_models_root_dir()+"/" +MODEL_DIR)
    MODEL_PATH = f"model.pth"
    model.MODEL_PATH = MODEL_PATH
    OPTIMIZER_PATH = "optimizer.pth"
    model.OPTIMIZER_PATH = OPTIMIZER_PATH
    return model

def print_dict_as_table(dict_data):
    max_key_length = max(len(str(key)) for key in dict_data.keys())
    max_val_length = max(len(str(value)) for value in dict_data.values())

    print(f"{'Key'.ljust(max_key_length)} | {'Value'.ljust(max_val_length)}")
    print("-" * (max_key_length + max_val_length + 3))

    for key, value in dict_data.items():
        print(f"{str(key).ljust(max_key_length)} | {str(value).ljust(max_val_length)}")

def print_model_config(config):
    print_dict_as_table(config)

def get_best_model(folder_path):
    # Regular expression to match the F-score in the filename
    pattern = re.compile(r"f1v=(\d+\.\d+)-model\.pth")

    # List all files in the directory
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Folder {folder_path} does not exist in {os.getcwd()}")
    # Extract F-scores and associate them with their filenames
    fscores = {}
    for file in files:
        match = pattern.search(file)
        if match:
            fscore = float(match.group(1))
            fscores[file] = fscore

    # Find the filename with the highest F-score
    best_model = max(fscores, key=fscores.get)

    return os.path.join(folder_path, best_model)


if __name__ == "__main__":
    folder_path = "../classifier/models/hierarchical/"
    best_model_path = get_best_model(folder_path)
    print(f"The best model is: {best_model_path}")
