import logging
import os

import torch
from addict import Dict
from gevent.lock import BoundedSemaphore
from ruamel.yaml import YAML

from lib.shape import to_tensor, view_shape

yaml = YAML(typ="rt")

from torch.nn.functional import pairwise_distance

from classifier.model.different_models import (configure_model, get_best_model,
                                               get_models_config_path)
from classifier.result.think import get_model, get_prediction
from lib.embedding import get_embeddings
from lib.t import catchtime


class Models:
    def __init__(self):
        self.models = {}

        with open(get_models_config_path()) as f:
            self.model_configs = Dict(yaml.load(f))["models"]

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
        # try to load threshold
        try:
            with open(os.path.join(config.MODEL_DIR, "threshold.txt"), "r") as f:
                config.threshold = float(f.read())
                logging.info(f"Loaded threshold {config.threshold=} from file")
        except FileNotFoundError:
            pass

        return model

    def encode(self, inputs, config):
        return to_tensor(get_embeddings(inputs, config))

    @property
    def config(self):
        return self.model_configs[self.active_model_name]

    def embed(self, inputs):
        return self.models[self.active_model_name](inputs.unsqueeze(0))

    def predict(self, inputs):
        semaphore = BoundedSemaphore()
        with semaphore:
            config = self.model_configs[self.active_model_name]


            embeddings = self.encode(inputs, config)
            for param in self.models[self.active_model_name].parameters():
                param.grad = None

            if config.model == "siamese":
                try:
                    scores = []
                    for embedding_pair in embeddings:
                        anchor = embedding_pair[0].view((-1, config.embedding_dim))

                        tuned1 = self.models[self.active_model_name](anchor)
                        tuned2 = self.models[self.active_model_name](embedding_pair[1].view((-1, config.embedding_dim)))

                        scores.append(
                            pairwise_distance(tuned1, tuned2, p=2   ).view((-1,)).item(

                            ))


                    if config.threshold:
                        scores = [1- (abs(score)/config.threshold) for score in scores]



                    return view_shape(scores, (-1,))
                except:
                    raise

            elif config.model == "ntuple":
                labels, certainty_scores = get_prediction(
                    self.models[self.active_model_name],
                    embeddings,
                    config=config,
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
