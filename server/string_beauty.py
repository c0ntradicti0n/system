from fuzzywuzzy import fuzz


def similarity(a, b):
    return fuzz.ratio(a, b) / 100


def remove_repeated_patterns(segment):
    length = len(segment)
    for i in range(1, length):
        pattern = segment[:i]
        if pattern * (length // i) == segment:
            return pattern
    return segment


def remove_duplicates(text, delimiters=[".", "!", "?"], threshold=0.7):
    segments = []
    last_index = 0
    for i, char in enumerate(text):
        if char in delimiters:
            segments.append(text[last_index : i + 1].strip())
            last_index = i + 1
    segments.append(text[last_index:].strip())

    segments = [remove_repeated_patterns(segment) for segment in segments]

    to_remove = set()
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if similarity(segments[i], segments[j]) > threshold:
                to_remove.add(segments[j])
                break  # break out once a segment is recognized as similar

    new_segments = []
    yet_removed = set()
    for segment in segments:
        if segment not in to_remove or segment in yet_removed:
            new_segments.append(segment)
        else:
            yet_removed.add(segment)

    return " ".join(new_segments)


# Tests
def test():
    test_cases = [
        ("ABCB CBCB. ABCB CBCB", "ABCB CBCB."),
        ("ABCB CBCB. ABCB CBCD", "ABCB CBCB."),
        ("This is a test. This is only a tst.", "This is a test. "),
        (
            "Hello world! Hola world! Bonjour le monde!",
            "Hello world! Bonjour le monde! ",
        ),
        ("Sample. Sample.", "Sample. "),
        ("This is similar. This is smilar.", "This is similar. "),
    ]

    for inp, expected in test_cases:
        output = remove_duplicates(inp)
        assert (
            output == expected
        ), f"For: '{inp}', Expected: '{expected}', but got: '{output}'"

    print("All tests passed!")


test()
