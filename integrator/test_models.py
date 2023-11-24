import logging

from main import classifier, hierarchy, opposite

from lib.helper import t

logging.basicConfig(level=logging.ERROR)


def test_opposite(pair, expected):
    _, s = opposite(pair)[0]
    print(s)
    if (s > 0) == expected:
        print(f"A {'t' if expected else 'f'} GOOD {pair=}")
        return True
    else:
        print(f"A {'t' if expected else 'f'} BAD {pair=}")


def test_thesis_antithesis_synthesis(triple, expected):
    lsk = classifier(triple)
    if lsk:
        result = compare_prediction_to_expected(expected, lsk, triple)
        print(f"△ {'t' if expected else 'f'} GOOD {triple=}")
        return result == expected
    logging.error(f"no result for {triple=}")
    return None


def test_hierarchy(pair, expected):
    lsk = hierarchy(pair)
    if lsk:
        result = compare_prediction_to_expected(expected, lsk, pair)
        if result == expected:
            print(f"H {'t' if expected else 'f'} GOOD {pair=}")
        return result == expected
    else:
        print(f"H {'t' if expected else 'f'} BAD {pair=}")
        return False


def compare_prediction_to_expected(expected, lsk, triple):
    matches = {v: k for k, s, v in zip(*lsk[0])}
    result = [matches[i] for i in triple]
    if result != expected:
        logging.error(f"{result=} {triple=}")
    return result


with t:
    assert test_opposite(
        ["addition and subtraction", "multiplication and division"], True
    )
with t:
    assert test_opposite(["good", "bad"], True)
with t:
    assert test_opposite(["yes", "no"], True)
with t:
    assert test_opposite(["true", "false"], True)
with t:
    assert not test_opposite(["good", "good"], False)
with t:
    assert not test_opposite(["bad", "green"], False)

with t:
    assert test_opposite(["bad", "good"], True)
with t:
    assert test_opposite(["up", "down"], True)
with t:
    assert test_opposite(["down", "up"], True)
with t:
    assert test_opposite(["left", "right"], True)
with t:
    assert test_opposite(["right", "left"], True)
with t:
    assert test_opposite(["be", "not to be"], True)

with t:
    assert test_opposite(["lazy", "fox"], False)
with t:
    assert test_opposite(["a", "b"], False)
with t:
    assert test_opposite(["happy", "moon"], False)

with t:
    assert test_opposite(["subtraction", "addition"], True)
with t:
    assert test_opposite(["multiplication", "division"], True)
with t:
    assert test_opposite(["exponent", "root"], True)

with t:
    assert test_opposite(["linear operations", "multiplicative operations"], True)
with t:
    assert test_thesis_antithesis_synthesis(
        [
            "linear operations",
            "multiplicative operations",
            "exponential and logarithmic functions",
        ],
        [1, 2, 3],
    )
with t:
    assert test_thesis_antithesis_synthesis(
        [
            "addition and subtraction",
            "multiplication and division",
            "exponential and logarithm",
        ],
        [1, 2, 3],
    )

with t:
    assert test_thesis_antithesis_synthesis(
        [
            "plus and minus",
            "times and divide",
            "exponent and root",
        ],
        [1, 2, 3],
    )
with t:
    assert test_thesis_antithesis_synthesis(["good", "bad", "neutral"], [1, 2, 3])
with t:
    assert test_thesis_antithesis_synthesis(["black", "white", "grey"], [2, 1, 3])
with t:
    assert test_thesis_antithesis_synthesis(
        ["plus", "minus", "plus minus zero"], [1, 2, 3]
    )
with t:
    assert test_thesis_antithesis_synthesis(["up", "down", "straight"], [1, 2, 3])
with t:
    assert test_thesis_antithesis_synthesis(["friend", "enemy", "diplomat"], [1, 2, 3])
with t:
    assert test_thesis_antithesis_synthesis(["to be", "not to be", "become"], [1, 2, 3])

with t:
    assert test_hierarchy(["moral", "good"], [1, 2])
with t:
    assert test_hierarchy(["moral", "bad"], [1, 2])
with t:
    assert test_hierarchy(["moral", "neutral"], [1, 2])
with t:
    assert test_hierarchy(["color", "black"], [1, 2])
with t:
    assert test_hierarchy(["color", "white"], [1, 2])
with t:
    assert test_hierarchy(["color", "grey"], [1, 2])
with t:
    assert test_hierarchy(["operation", "plus"], [1, 2])
with t:
    assert test_hierarchy(["operation", "minus"], [1, 2])
with t:
    assert test_hierarchy(["operation", "neutral element 0"], [1, 2])
with t:
    assert test_hierarchy(["body", "hand"], [1, 2])
with t:
    assert test_hierarchy(["mathematics", "plus"], [1, 2])
with t:
    assert test_hierarchy(["mathematics", "multiplication"], [1, 2])
with t:
    assert test_hierarchy(["human", "friend"], [1, 2])
with t:
    assert test_hierarchy(["human", "enemy"], [1, 2])
with t:
    assert test_hierarchy(["human", "diplomat"], [1, 2])
