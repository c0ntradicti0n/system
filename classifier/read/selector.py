import contextlib
import os
import random
import timeit

import addict
import numpy as np
import torch

from lib.embedding import get_embeddings
from lib.helper import OutputLevel, e, tree
from lib.shape import to_tensor
from lib.wordnet import (yield_cycling_list, yield_extra_sample,
                         yield_random_ant_wordnet_sample,
                         yield_random_hie_wordnet_sample)


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


ant_wn_gen = None
hie_wn_gen = None
ant_pos_gen = None
ant_neg_gen = None

GENS = addict.Dict()


def tree_walker(yield_mode="valid", n_samples=10, config=None):
    samples = []
    global ant_wn_gen, hie_wn_gen, ant_pos_gen, ant_neg_gen

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
        elif yield_mode == "valid_ant":
            with e:
                a = d[1]["."]
            with e:
                b = d[2]["."]
            if all([a, b]) and a not in samples and b not in samples:
                samples.extend([(a, 1), (b, 2)])
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
            samples.extend([(a, 1), (b, 2), (c, 3), (X, 4)])
            yield_mode = "random"

        elif yield_mode == "valid_hie":
            t = random.choice([1, 2, 3])
            with e:
                a = d[t]["."]
            with e:
                X = d["."]
            if all([a, X]) and a not in samples and X not in samples:
                samples.extend([(X, 1), (a, 2)])
                yield_mode = "random"
            else:
                gold_samples.pop(gold_samples.index(random_dir))
        elif yield_mode == "random_hie":
            t = random.choice([1, 2, 3])

            with e:
                X = d["."]

            if all([X]) and X not in samples:
                samples.extend([(X, 0)])
                yield_mode = "random"
            else:
                gold_samples.pop(gold_samples.index(random_dir))
        elif yield_mode == "valid_hie_wordnet":
            if not hie_wn_gen:
                hie_wn_gen = yield_random_hie_wordnet_sample()
            hyper, hypo = next(hie_wn_gen)
            samples.extend([(hyper, 1), (hypo, 2)])
            yield_mode = "random"
        elif yield_mode == "valid_ant_wordnet":
            if not ant_wn_gen:
                ant_wn_gen = yield_random_ant_wordnet_sample()
            word, anto = next(ant_wn_gen)
            samples.extend([(word, 1), (anto, 2)])
            yield_mode = "random"
        elif yield_mode == "random_hie_wordnet":
            if not hie_wn_gen:
                hie_wn_gen = yield_random_hie_wordnet_sample()
            hyper1, _ = next(hie_wn_gen)
            _, hypo2 = next(hie_wn_gen)

            samples.extend([(hyper1, 0), (hypo2, 0)])
            yield_mode = "random"
        elif yield_mode == "valid_pos_samples":
            if not ant_pos_gen:
                ant_pos_gen = yield_extra_sample(config["valid_pos_samples"])
            word, anto = next(ant_pos_gen)
            samples.extend([(word, 1), (anto, 2)])
            yield_mode = "random"
        elif yield_mode == "list_neg_samples":
            if not ant_neg_gen:
                ant_neg_gen = yield_extra_sample(config["list_neg_samples"])
            word, anto = next(ant_neg_gen)
            samples.extend([(word, 0), (anto, 0)])
            yield_mode = "random"
        elif yield_mode== "from_file":
            if not GENS[config.from_file]:
                import importlib
                f = importlib.import_module(config.from_file)
                GENS[config.from_file] = yield_cycling_list(f.samples)
            row = next(GENS[config.from_file])
            samples.extend([(r, i+1) for i, r in enumerate(row)])
            yield_mode = "random"

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

    def generate_data(self, config, batch_size=None, no_labels=False):
        texts = []
        labels = []
        embeddings = []
        for _ in range(batch_size if batch_size else self.config.batch_size):
            kind = np.random.choice(self.config.classes, p=self.config.probs)
            sample, label = list(
                zip(*tree_walker(kind, self.config.n_samples, config=self.config))
            )
            texts.append(sample)
            labels.append(label)

            embeddings.append(get_embeddings(sample, self.config))
        if not no_labels:
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels =None
        if config.result_add:
            labels = labels - config.result_add
        return to_tensor(embeddings), labels, texts
