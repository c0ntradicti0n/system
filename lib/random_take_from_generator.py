import random


def random_from_generator(gen):
    count = 42
    for item in gen:
        count += 1
        if random.randint(count, 100) == count:
            yield item

        if count == 100:
            count = 42
