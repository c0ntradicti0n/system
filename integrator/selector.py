import contextlib
import os
import random
import timeit

import torch
from helper import OutputLevel, e, tree

from integrator import config
from lib.embedding import get_embeddings


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

    samples = [(text.replace(".md", "").strip(), label) for text, label in samples]
    random.shuffle(samples)
    samples.reverse()
    return samples


class DataGenerator:
    def __init__(self):
        pass

    def adjust_data_distribution(self, fscore):
        # Adjust data distribution based on f score
        pass

    def generate_data(self, n_samples=config.n_samples):
        texts = []
        labels = []
        embeddings = []
        for _ in range(config.batch_size):
            sample, label = list(
                zip(
                    *tree_walker(random.choice(["labels", "valid"
                                                ]), n_samples)
                )  # , "random"
            )
            texts.append(sample)
            labels.append(label)

            embeddings.append(get_embeddings(sample))

        return torch.tensor(embeddings), torch.tensor(labels, dtype=torch.long), texts
