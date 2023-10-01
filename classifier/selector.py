import contextlib
import os
import random
import timeit

import numpy as np
import torch

from lib.embedding import get_embeddings
from lib.helper import OutputLevel, e, tree


@contextlib.contextmanager
def time_it(name):
    start = timeit.default_timer()
    yield
    end = timeit.default_timer()
    print(f"{name} took {end - start} seconds")


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
        a, b, c, X = None, None, None, None

        random_dir = random.choice(gold_samples)
        d = tree(
            basepath="",
            startpath=random_dir,
            depth=1,
            format="json",
            info_radius=10,
            exclude=[".git", ".git.md", ".gitignore", ".DS_Store", ".idea"],
            pre_set_output_level=OutputLevel.FILENAMES,
            prefix_items=True,
        )
        if yield_mode == "labels":
            a, b, c = (
                "thesis - main topic",
                "antithesis - opposite",
                "synthesis - combination of thesis and antithesis",
            )
            samples.extend([(a, 1), (b, 2), (c, 3)])
            yield_mode = "random"
        elif yield_mode == "valid":
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
            with e:
                c = d[r]["."]
            if not c:
                with e:
                    c = d["."]

            if c and not isinstance(c, str):
                raise ValueError(f"{c=}")
            if c:
                samples.append((c, 0))

        elif yield_mode == "labels_hie":
            X, a, b, c = (
                "super - superordinated",
                "thesis - subordinated",
                "antithesis - subordinated",
                "synthesis - subordinated",
            )
            samples.extend([(a, 1), (b, 2), (c, 3), (X,  4)])
            yield_mode = "random"

        elif yield_mode == "valid_hie":
            with e:
                a = d[1]["."]
            with e:
                b = d[2]["."]
            with e:
                c = d[3]["."]
            with e:
                X = d["."]
            if (
                all([a, b, c, X])
                and a not in samples
                and b not in samples
                and c not in samples
                and X not in samples
            ):
                samples.extend([(a, 1), (b, 2), (c,3), (X, 4)])
                yield_mode = "random"
            else:
                gold_samples.pop(gold_samples.index(random_dir))

        elif yield_mode == "random_hie":
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
                samples.extend([(a, 1), (b,2), (c, 3)])
                yield_mode = "random"
            else:
                gold_samples.pop(gold_samples.index(random_dir))

        elif True:
            raise NotImplementedError(f"{yield_mode=} not implemented!")

    samples = [(text.replace(".md", "").strip(), label) for text, label in samples]
    random.shuffle(samples)
    samples.reverse()
    return samples


class DataGenerator:
    def __init__(self, config):
        self.config = config

    def adjust_data_distribution(self, fscore):
        # Adjust data distribution based on f score
        pass

    def generate_data(self):
        texts = []
        labels = []
        embeddings = []
        for _ in range(self.config.batch_size):
            sample, label = list(
                zip(
                    *tree_walker(
                        np.random.choice(self.config.classes, p=self.config.probs),
                        self.config.n_samples,
                    )
                )
            )
            texts.append(sample)
            labels.append(label)

            embeddings.append(get_embeddings(sample, self.config))

        return torch.tensor(embeddings), torch.tensor(labels, dtype=torch.long), texts
