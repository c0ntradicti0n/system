import os

import torch
from addict import Dict
from ruamel import yaml

from classifier.different_models import (configure_model, get_best_model,
                                         get_models_congig_path,
                                         get_models_root_dir)
from classifier.think import get_model, get_prediction
from lib.embedding import get_embeddings
from lib.t import catchtime


class Models:
    def __init__(self):
        self.models = {}

        with open(get_models_congig_path()) as f:
            self.model_configs = Dict(yaml.load(f, Loader=yaml.Loader))["models"]

    def __getitem__(self, item):
        self.active_model_name = item
        if item not in self.models:
            self.models[item] = self.load_model(item)
            return self
        else:
            return self

    def load_model(self, model_name):
        config = self.model_configs[model_name]
        if not config:
            raise ValueError(
                f"Model {model_name} not found in models.yml\nOptions are {list(self.model_configs.keys())}"
            )

        self.model_configs[model_name] = configure_model(model_name, config)
        model = self.load_torch_model(self.model_configs[model_name])
        return model

    def load_torch_model(self, config):
        model = get_model(config)
        model.load_state_dict(
            torch.load(get_best_model(os.path.join(config.MODEL_DIR)))
        )
        return model

    def encode(self, inputs, config):
        return torch.tensor(get_embeddings(inputs, config))

    @property
    def config(self):
        return self.model_configs[self.active_model_name]

    def predict(self, inputs):
        c = self.model_configs[self.active_model_name]
        embeddings = self.encode(inputs, c)
        for param in self.models[self.active_model_name].parameters():
            param.grad = None
        labels, certainty_scores = get_prediction(
            self.models[self.active_model_name],
            embeddings,
            c,
            compute_confidence=True,
        )
        if self.model_configs[self.active_model_name].result_add:
            labels = labels + self.model_configs[self.active_model_name].result_add
        return labels, certainty_scores


MODELS = Models()


if __name__ == "__main__":
    models = Models()
    with catchtime("loading and predicting"):
        labels, certainty_scores = models["hierarchical_2"].predict(
            ["class", "instance"]
        )
        print(labels, certainty_scores)
    with catchtime("loading and predicting"):
        labels, certainty_scores = models["hierarchical_2"].predict(["plant", "tree"])
        print(labels, certainty_scores)
    with catchtime("loading and predicting"):
        labels, certainty_scores = models["hierarchical_2"].predict(["yellow", "color"])
        print(labels, certainty_scores)
    with catchtime("loading and predicting"):
        labels, certainty_scores = models["tas_3_only"].predict(
            ["light", "dawn", "night"]
        )
        print(labels, certainty_scores)
    with catchtime("predicting"):
        labels, certainty_scores = models["tas_3_only"].predict(
            ["diplomat", "enemy", "friend"]
        )
        print(labels, certainty_scores)
    with catchtime("predicting"):
        labels, certainty_scores = models["tas_3_only"].predict(
            [
                "change it to have the same length",
                "insert a value in line and it will be more",
                "delete a char in a line to make it be less",
            ]
        )
        print(labels, certainty_scores)
