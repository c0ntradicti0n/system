import random


def fatten_dict(d, depth=1, max_depth=3):
    if depth > max_depth:
        return

    d = d.copy()

    # Decide randomly whether to add a new key-value pair or a nested dictionary
    if random.choice([True, False]):
        # Add a new key-value pair
        key = f"key{random.randint(1, 1000)}"
        value = f"value{random.randint(1, 1000)}"
        d[key] = value
    else:
        # Add a nested dictionary
        key = f"nested{random.randint(1, 1000)}"
        d[key] = {}
        fatten_dict(d[key], depth + 1, max_depth)

    # Randomly decide whether to continue fattening at this level
    if random.choice([True, False]):
        fatten_dict(d, depth, max_depth)

    return d
