import math


def gaussian_weight(index, mean, std_dev):
    # Gaussian function to provide weights
    return (1 / (std_dev * math.sqrt(2 * math.pi))) * math.exp(
        -0.5 * ((index - mean) / std_dev) ** 2
    )


def weighted_fuzzy_compare(str1, str2, threshold=0.5):
    score = 0
    total_weight = 0

    if str2.startswith(str1[:-1]):
        return True, 0.9

    # Calculate mean and standard deviation for Gaussian weighting
    mean = len(str2) / 2
    std_dev = len(str2) / 4  # Roughly 68% of data within one std_dev

    for index, char2 in enumerate(str2[:-2]):
        weight = gaussian_weight(index, mean, std_dev)

        if index < len(str1):
            char1 = str1[index]
            score += weight if char1 == char2 else 0

        total_weight += weight

    # Normalize the score
    normalized_score = score / total_weight if total_weight != 0 else 0

    # Check for length difference penalty
    length_difference = len(str1) - len(str2[:-2])
    if length_difference > 0:
        normalized_score -= (
            length_difference * 0.1
        )  # Subtract a penalty for each extra character

    # Return if the normalized score is greater than the threshold
    return normalized_score > threshold, normalized_score


if __name__ == "__main__":
    # Test
    str1 = "131321"
    str2 = "132321"
    threshold = 0.5

    similar, score = weighted_fuzzy_compare(str1, str2, threshold)
    print(
        f"The strings are {'similar' if similar else 'not similar'}, with a score of {score:.2f}"
    )

    str1 = "132322"
    str2 = "132321"

    similar, score = weighted_fuzzy_compare(str1, str2, threshold)
    print(
        f"The strings are {'similar' if similar else 'not similar'}, with a score of {score:.2f}"
    )

    str1 = "13232212"
    str2 = "132321"

    similar, score = weighted_fuzzy_compare(str1, str2, threshold)
    print(
        f"The strings are {'similar' if similar else 'not similar'}, with a score of {score:.2f}"
    )

    str1 = "3221232323"
    str2 = "322"

    similar, score = weighted_fuzzy_compare(str1, str2, threshold)
    print(
        f"The strings are {'similar' if similar else 'not similar'}, with a score of {score:.2f}"
    )

    str1 = "32212"
    str2 = "322"

    similar, score = weighted_fuzzy_compare(str1, str2, threshold)
    print(
        f"The strings are {'similar' if similar else 'not similar'}, with a score of {score:.2f}"
    )

    str1 = "32123"
    str2 = "322"

    similar, score = weighted_fuzzy_compare(str1, str2, threshold)
    print(
        f"The strings are {'similar' if similar else 'not similar'}, with a score of {score:.2f}"
    )

    str1 = "3212"
    str2 = "313"

    similar, score = weighted_fuzzy_compare(str1, str2, threshold)
    print(
        f"The strings are {'similar' if similar else 'not similar'}, with a score of {score:.2f}"
    )

    str1 = "123321"
    str2 = "312123"

    similar, score = weighted_fuzzy_compare(str1, str2, threshold)
    print(
        f"The strings are {'similar' if similar else 'not similar'}, with a score of {score:.2f}"
    )

    str1 = "3122"
    str2 = "312111"

    similar, score = weighted_fuzzy_compare(str1, str2, threshold)
    print(
        f"The strings are {'similar' if similar else 'not similar'}, with a score of {score:.2f}"
    )
