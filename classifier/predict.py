import os

import torch
from addict import Dict
from ruamel import yaml

from classifier.different_models import configure_model, get_best_model
from classifier.think import get_model, get_prediction
from lib.embedding import get_embeddings
from lib.t import catchtime


class Models:
    def __init__(self):
        self.models = {}

        with open("models.yml") as f:
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

    def predict(self, inputs):
        embeddings = self.encode(inputs, self.model_configs[self.active_model_name])
        labels, certainty_scores = get_prediction(
            self.models[self.active_model_name],
            embeddings,
            self.model_configs[self.active_model_name],
            compute_confidence=True,
        )
        return labels, certainty_scores


if __name__ == "__main__":
    models = Models()

    with catchtime("loading and predicting"):
        labels, certainty_scores = models["hierarchical"].predict(
            ["fuck", "yellow", "blue", "red", "colors"]
        )
        print(labels, certainty_scores)
    with catchtime("predicting"):
        labels, certainty_scores = models["hierarchical"].predict(
            ["fuck", "yellow", "blue", "red", "colors"]
        )
        print(labels, certainty_scores)
    with catchtime("predicting"):
        labels, certainty_scores = models["hierarchical"].predict(
            ["WAHT", "black", "white", "grey", "color"]
        )
        print(labels, certainty_scores)
