import ast
import csv
import json
import random

from nltk.corpus import wordnet as wn


def hypernym_generator():
    try:
        synsets = list(wn.all_synsets())
    except:
        import nltk

        nltk.download("wordnet")
        synsets = list(wn.all_synsets())

    while True:
        random_synset = random.choice(synsets)
        hypernyms = list(random_synset.hypernyms())
        if len(hypernyms) >= 4:
            yield random_synset, hypernyms


def write_to_csv(filename):
    try:
        synsets = list(wn.all_synsets())
    except:
        import nltk

        nltk.download("wordnet")
        synsets = list(wn.all_synsets())

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        added = []
        n = 0
        for synset in synsets:
            hypernyms = list(synset.hyponyms())
            if len(hypernyms) < 4:
                continue
            if synset.lemmas()[0].name() not in added:
                row = [
                    synset.lemmas()[0].name(),
                    [
                        lemma.name()
                        for hypernym in hypernyms
                        for lemma in hypernym.lemmas()
                    ],
                ]
                writer.writerow(row)
                added.append(synset.lemmas()[0].name())
                n += 1
            if n > 10000:
                break


def yield_random_wordnet_sample():
    with open("hypernyms.csv", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x = ast.literal_eval(row[1])
            r = random.randint(1, len(row) - 1)

            yield row[0], x[r]

if __name__ == "__main__":
    write_to_csv("hypernyms.csv")
