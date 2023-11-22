import ast
import csv
import itertools
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


def get_explanation(synset):
    explanation = " – " + " · ".join(list(synset.examples()))

    return explanation


def write_to_csv():
    try:
        synsets = list(wn.all_synsets())
    except:
        import nltk

        nltk.download("wordnet")
        synsets = list(wn.all_synsets())

    with open("antonyms.csv", "w", newline="") as ant_csvfile:
        antonym_writer = csv.writer(ant_csvfile)
        with open("hyponyms.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            hyp = []
            ant = []
            n = 0
            for synset in synsets:
                for lemma in synset.lemmas():
                    antonyms = list(lemma.antonyms())

                    antonym_row = [
                        "'" + synset.definition() + "'" + get_explanation(synset),
                        [
                            "'"
                            + antonym.synset().definition()
                            + "'"
                            + get_explanation(antonym.synset())
                            for antonym in antonyms
                        ],
                    ]

                    if len(antonym_row[1]) > 0:
                        antonym_writer.writerow(antonym_row)
                    antonym_row = [
                        lemma.name(),
                        [antonym.name() for antonym in antonyms],
                    ]
                    if len(antonym_row[1]) > 0:
                        antonym_writer.writerow(antonym_row)

                hyponyms = list(synset.hyponyms())
                if len(hyponyms) < 4:
                    continue
                if synset.lemmas()[0].name() not in hyp:
                    row = [
                        "'" + synset.definition() + "'" + get_explanation(synset),
                        [
                            "'" + hyponym.definition() + "'" + get_explanation(hyponym)
                            for hyponym in hyponyms
                        ],
                    ]
                    writer.writerow(row)

                    row = [
                        synset.definition(),
                        [hyponym.definition() for hyponym in hyponyms],
                    ]
                    writer.writerow(row)

                    row = [
                        synset.definition(),
                        [hyponym.definition() for hyponym in hyponyms],
                    ]
                    writer.writerow(row)

                    for lemma in synset.lemmas():
                        row = [
                            lemma.name(),
                            [
                                lemmah.name()
                                for hyponym in hyponyms
                                for lemmah in hyponym.lemmas()
                            ],
                        ]
                        writer.writerow(row)

                    hyp.append(synset.lemmas()[0].name())
                    n += 1
                if n > 10000:
                    break


def yield_extra_sample(filename):
    while True:
        with open(filename, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                x = ast.literal_eval(row[1])
                r = random.randint(0, len(x) - 1)

                yield row[0], x[r]


def yield_cycling_list(list):
    gen = itertools.cycle(list)
    yield from gen


def yield_random_hie_wordnet_sample():
    while True:
        with open("data/hyponyms.csv", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                x = ast.literal_eval(row[1])
                r = random.randint(0, len(x) - 1)

                yield row[0], x[r]


def yield_random_ant_wordnet_sample():
    while True:
        with open("data/antonyms.csv", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                x = ast.literal_eval(row[1])
                r = random.randint(0, len(x) - 1)

                if x:
                    try:
                        yield row[0], x[r]
                    except Exception as e:
                        raise


if __name__ == "__main__":
    write_to_csv()
