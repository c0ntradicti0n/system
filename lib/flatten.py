def flatten_dict_values(d):
    def flatten(item):
        if isinstance(item, dict):
            for value in item.values():
                yield from flatten(value)
        else:
            yield item  #

    return list(flatten(d))


def flatten_and_format(d, separator=", "):
    """
    Flatten a nested dictionary and format its string values, including values in lists, tuples, and sets.
    Replace ".md" with an empty string and concatenate values with a specified separator.

    :param d: The dictionary to flatten and format.
    :param separator: The separator to use for concatenating string values.
    :return: A single string containing all the formatted values from the dictionary and its iterables.
    """
    def recursive_items(item):
        if isinstance(item, dict):
            for key, value in item.items():
                yield from recursive_items(value)
        elif isinstance(item, (list, tuple, set)):
            for element in item:
                yield from recursive_items(element)
        else:
            yield item

    def format_value(value):
        if isinstance(value, str):
            return value.replace(".md", "")
        else:
            return str(value)

    formatted_values = [format_value(value) for value in recursive_items(d)]
    return separator.join(formatted_values)


if __name__ == "__main__":
    # Example usage
    nested_dict = {
        "key1": "value1.md",
        "key2": {
            "nestedKey1": "nestedValue1.md",
            "nestedKey2": "nestedValue2.md"
        },
        "key3": ["listValue", "listValue2"],
        "key4": {
            "nestedKey3": {
                "nestedKey4": "nestedValue3.md"
            }
        }}


    print(flatten_and_format(nested_dict))