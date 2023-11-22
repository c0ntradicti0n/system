import ast
import csv
import random

from lib.interleave_generators import interleave_generators


def yield_random_hie_wordnet_sample():
    while True:
        with open("data/hyponyms.csv", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                x = ast.literal_eval(row[1])
                r = random.randint(0, len(x) - 1)

                yield row[0], x[r]



hyponyms = yield_random_hie_wordnet_sample()

extra = [
    ("linear operations", "addition"),
    ("multiplicative operations", "multiplication"),
    ("exponential and logarithmic functions", "exponent"),
]
samples = interleave_generators(extra, hyponyms)
