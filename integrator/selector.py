import itertools
import torch

import os
import random

from helper import tree
from integrator.embedding import get_embedding


def tree_walker(basepath, yield_mode="valid"):
    """
    Generator function based on the tree function to yield either:
    - a full subfolder from the root directory (yield_mode="valid")
    - a random file from anywhere in the tree (yield_mode="random")
    """

    # Get JSON representation of the directory structure
    directory_structure = tree(startpath="", basepath=basepath, format="json")

    if yield_mode == "valid":
        # Yield each subfolder at the root
        for subfolder, contents in directory_structure.items():
            if isinstance(contents, dict):
                yield subfolder, contents

    elif yield_mode == "random":
        all_files = []

        def extract_files(d, current_path=[]):
            """
            Helper function to extract all files in the directory tree
            """
            for key, value in d.items():
                new_path = current_path + [key]
                if isinstance(value, dict):
                    extract_files(value, new_path)
                else:
                    all_files.append("/".join(new_path))

        extract_files(directory_structure)

        while True:
            # Randomly select a file from the list and yield
            yield random.choice(all_files)


class DataGenerator:
    def __init__(self):
        pass

    def adjust_data_distribution(self, fscore):
        # Adjust data distribution based on fscore
        pass

    def generate_data(self, n_samples=10):
        samples, labels = itertools.islice(tree_walker(os.environ["SYSTEM"], random.choice("valid", "random" )), n_samples)
        embeddings =  get_embedding(samples)
        return torch.stack(embeddings), torch.tensor(labels)