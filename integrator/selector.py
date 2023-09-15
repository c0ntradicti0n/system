import os
import random

import torch
from helper import OutputLevel, e, tree
from integrator import config

from integrator.embedding import get_embedding


def list_all_subdirs(dir_path, exclude):
    """Recursively lists all subdirectories."""
    subdirs = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for dirname in dirnames:
            if not any(e in dirname or e in dirpath for e in exclude):
                subdirs.append(os.path.join(dirpath, dirname))



    return subdirs


gold_samples = list_all_subdirs(
    os.environ["SYSTEM"],
    exclude=[".git", ".git.md", ".gitignore", ".DS_Store", ".idea"],
)


def random_subdir(dir_path):
    """Selects a random subdirectory."""

    if not gold_samples:
        raise ValueError("No subdirectories found!")

    return random.choice(gold_samples)




def tree_walker(yield_mode="valid", n_samples=10):
    samples = []
    while len(samples) < n_samples:
        random_dir = random.choice(gold_samples)
        d = tree(
            basepath="",
            startpath= random_dir,
            depth=1,
            format="json",
            info_radius=10,
            exclude=[".git", ".git.md", ".gitignore", ".DS_Store", ".idea"],
            pre_set_output_level=OutputLevel.FILENAMES,
            prefix_items=True,
        )

        if yield_mode == "valid":
            a, b, c = None, None, None
            with e:
                a = d[1]["."]
            with e:
                b = d[2]["."]
            with e:
                c = d[3]["."]

            if (
                all([a, b, c])
                and a not in samples
                and b not in samples
                and c not in samples
            ):
                samples.extend([(a, 1), (b, 2), (c, 3)])
                yield_mode = "random"
        elif yield_mode == "random":
            r = random.choice([1, 2, 3])
            c = None
            with e:
                c = d[r]["."]
            if not c:
                with e:
                    c = d["."]

            if c and not isinstance(c, str):
                raise ValueError(f"{c=}")
            if c:
                samples.append((c, 0))
    random.shuffle(samples)
    return samples


class DataGenerator:
    def __init__(self):
        pass

    def adjust_data_distribution(self, fscore):
        # Adjust data distribution based on fscore
        pass

    def generate_data(self, n_samples=config.n_samples):
        texts = []
        labels = []
        embeddings = []
        for _ in range(config.batch_size):
            sample, label = list(zip(*tree_walker(random.choice(["valid", "random"]), n_samples)))
            texts.append(sample)
            labels.append(label)

            embeddings.append(get_embedding(sample))

        return torch.tensor(embeddings), torch.tensor(labels, dtype=torch.long)
