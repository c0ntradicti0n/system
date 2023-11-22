import logging

from main import classifier, hierarchy, opposite

from lib.helper import t

logging.basicConfig(level=logging.ERROR)


def test_opposite(pair, expected):
    (n1, n2), s = opposite(pair)[0]
    print(s)
    if (s > 0) == expected:
        print(f"A {'t' if expected else 'f'} GOOD {pair=}")
        return True
    else:
        print(f"A {'t' if expected else 'f'} BAD {pair=}")


def test_thesis_antithesis_synthesis(triple, expected):
    lsk = classifier(triple)
    lsk = [list(sorted(x[0])) for x in lsk]
    if lsk:
        result = compare_prediction_to_expected(expected, lsk, triple)
        return result == expected
    logging.error(f"no result for {triple=}")
    return None


def test_hierarchy(pair, expected):
    (n1, n2), s = hierarchy(pair)[0]
    if (s > 0) == expected:
        print(f"H {'t' if expected else 'f'} GOOD {pair=}")
        return True
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
    assert test_hierarchy(["moral", "good"], True)
with t:
    assert test_hierarchy(["moral", "bad"], True)
with t:
    assert test_hierarchy(["moral", "neutral"], True)
with t:
    assert test_hierarchy(["color", "black"], True)
with t:
    assert test_hierarchy(["color", "white"], True)
with t:
    assert test_hierarchy(["color", "grey"], True)
with t:
    assert test_hierarchy(["operation", "plus"], True)
with t:
    assert test_hierarchy(["operation", "minus"], True)
with t:
    assert test_hierarchy(["operation", "neutral element 0"], True)
with t:
    assert test_hierarchy(["body", "hand"], True)
with t:
    assert test_hierarchy(["mathematics", "plus"], True)
with t:
    assert test_hierarchy(["mathematics", "multiplication"], True)
with t:
    assert test_hierarchy(["relation", "friend"], True)
with t:
    assert test_hierarchy(["relation", "enemy"], True)
with t:
    assert test_hierarchy(["relation", "diplomat"], True)
