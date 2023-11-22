from itertools import zip_longest


def interleave_generators(gen1, gen2, fillvalue=object()):
    """
    Interleave two generators, taking one item from each alternately.
    """
    for item1, item2 in zip_longest(gen1, gen2, fillvalue=fillvalue):
        if item1 is not fillvalue:
            yield item1
        if item2 is not fillvalue:
            yield item2
