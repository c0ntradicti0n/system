import logging

from main import antagonizer, classifier, organizer

from helper import t

logging.basicConfig(level=logging.ERROR)


def test_antonym(pair):
    lsk = antagonizer(pair)
    if lsk and lsk[0] and lsk[0][0][0] == 2:
        return True
    return False


def test_classifier(tripple, expected):
    lsk = classifier(tripple)
    if lsk:
        result = compare_prediction_to_expected(expected, lsk, tripple)
        return result == expected
    logging.error(f"no result for {tripple=} {expected=}")
    return None


def compare_prediction_to_expected(expected, lsk, tripple):
    matches = {v: k for k, s, v in zip(*lsk[0])}
    result = [matches[i] for i in tripple]
    if result != expected:
        logging.error(f"{result=} {expected=} {tripple=}")
    return result


def test_organizer(tripple, expected):
    lsk = organizer(tripple)
    if lsk:
        result = compare_prediction_to_expected(expected, lsk, tripple)
        return result == expected
    logging.error(f"no result for {tripple=} {expected=}")
    return None


with t:
    assert test_antonym(["good", "bad"])
with t:
    assert test_antonym(["bad", "good"])
with t:
    assert test_antonym(["up", "down"])
with t:
    assert test_antonym(["down", "up"])
with t:
    assert test_antonym(["left", "right"])
with t:
    assert test_antonym(["right", "left"])
with t:
    assert test_antonym(["be", "not to be"])

with t:
    assert not test_antonym(["lazy", "fox"])
with t:
    assert not test_antonym(["a", "b"])

with t:
    assert test_antonym(["linear operations", "multiplicative operations"])
with t:
    assert test_classifier(
        [
            "linear operations",
            "multiplicative operations",
            "exponential and logarithmic functions",
        ],
        [1, 2, 3],
    )

with t:
    assert test_classifier(["good", "bad", "neutral"], [1, 2, 3])
with t:
    assert test_classifier(["black", "white", "grey"], [2, 1, 3])
with t:
    assert test_classifier(["plus", "minus", "plus minus zero"], [1, 2, 3])
with t:
    assert test_classifier(["up", "down", "straight"], [1, 2, 3])
with t:
    assert test_classifier(["friend", "enemy", "diplomat"], [1, 2, 3])
with t:
    assert test_classifier(["to be", "not to be", "become"], [1, 2, 3])

with t:
    assert test_organizer(["moral", "good"], [1, 2])
with t:
    assert test_organizer(["moral", "bad"], [1, 2])
with t:
    assert test_organizer(["moral", "neutral"], [1, 2])
with t:
    assert test_organizer(["color", "black"], [1, 2])
with t:
    assert test_organizer(["color", "white"], [1, 2])
with t:
    assert test_organizer(["color", "grey"], [1, 2])
with t:
    assert test_organizer(["operation", "plus"], [1, 2])
with t:
    assert test_organizer(["operation", "minus"], [1, 2])
with t:
    assert test_organizer(["operation", "neutral element 0"], [1, 2])
with t:
    assert test_organizer(["body", "hand"], [1, 2])
with t:
    assert test_organizer(["mathematics", "plus"], [1, 2])
with t:
    assert test_organizer(["mathematics", "multiplication"], [1, 2])
with t:
    assert test_organizer(["relation", "friend"], [1, 2])
with t:
    assert test_organizer(["relation", "enemy"], [1, 2])
with t:
    assert test_organizer(["relation", "diplomat"], [1, 2])
