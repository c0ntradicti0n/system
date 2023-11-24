def maxislice(generator, n):
    """Yield up to n items from the generator. Yields fewer if the generator is exhausted."""
    count = 0
    for item in generator:
        yield item
        count += 1
        if count >= n:
            break
