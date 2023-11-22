def dict_diff(d1, d2, path=""):
    """
    Recursively compares two dictionaries, including nested dictionaries,
    and returns the differences.
    """
    diff = {}

    # Combine keys from both dictionaries
    keys = set(d1.keys()) | set(d2.keys())

    for key in keys:
        # Construct path to current key
        new_path = f"{path}.{key}" if path else key

        # Both values are dictionaries, so do a recursive call
        if isinstance(d1.get(key), dict) and isinstance(d2.get(key), dict):
            sub_diff = dict_diff(d1[key], d2[key], new_path)
            if sub_diff:
                diff[new_path] = sub_diff

        # Values are different
        elif d1.get(key) != d2.get(key):
            diff[new_path] = {"dict1": d1.get(key), "dict2": d2.get(key)}

    return diff
