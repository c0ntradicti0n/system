import math
import pickle
from itertools import pairwise


class CustomCombinations:
    def __init__(self, iterable, r=None, on=None, on_indices=None):
        self.pool = tuple(iterable)
        self.n = len(self.pool)
        self.r = self.n if r is None else r

        if self.r > self.n or self.r < 0:
            self.exhausted = True
        else:
            self.indices = list(range(self.r))
            self.exhausted = False

        self.yielded_count = 0
        self.on = on
        self.on_indices = on_indices

    def __iter__(self):
        return self

    def __next__(self):
        while not self.exhausted:
            result = tuple(self.pool[i] for i in self.indices)
            if self.on is not None:
                valid = True
                for a, b in pairwise(result + result[:1]):
                    if self.on_indices is not None:
                        i_a, i_b = self.on_indices[a], self.on_indices[b]
                    else:
                        i_a, i_b = a, b
                    if not self.on[i_a, i_b]:
                        self._advance_generator()
                        self.yielded_count += 1
                        valid = False

                if not valid:
                    continue

            self.yielded_count += 1
            self._advance_generator()
            return result
        raise StopIteration

    def _advance_generator(self):
        for i in reversed(range(self.r)):
            if self.indices[i] != i + self.n - self.r:
                break
        else:
            self.exhausted = True
            return

        self.indices[i] += 1
        for j in range(i + 1, self.r):
            self.indices[j] = self.indices[j - 1] + 1

    def remaining_combinations(self):
        total_combinations = math.comb(self.n, self.r)
        return total_combinations - self.yielded_count

    def is_exhausted(self):
        return self.exhausted

    def get_percentage(self):
        return round(self.yielded_count / math.comb(self.n, self.r), 2)

    def __getstate__(self):
        return (
            self.pool,
            self.r,
            self.indices,
            self.exhausted,
            self.yielded_count,
            self.on,
            self.on_indices,
        )

    def __setstate__(self, state):
        (
            self.pool,
            self.r,
            self.indices,
            self.exhausted,
            self.yielded_count,
            self.on,
            self.on_indices,
        ) = state
        self.n = len(self.pool)


if __name__ == "__main__":

    def test(elements, r):
        combinations_iterator = CustomCombinations(elements, r)

        for _ in range(3):
            print(next(combinations_iterator))

        with open(str(elements.__hash__) + "iterator_state.pkl", "wb") as f:
            pickle.dump(combinations_iterator, f)

        print("---- Later or in another environment ----")
        with open(str(elements.__hash__) + "iterator_state.pkl", "rb") as f:
            restored_iterator = pickle.load(f)

        for combination in restored_iterator:
            print(combination)

        assert restored_iterator.is_exhausted()

    data = [1, 2, 3, 4, 5]
    test(data, 3)
    data = ["a", "b", "c", "d", "e"]
    test(data, 3)
    data = ["ab", "bc", "cd", "de", "ef"]
    test(data, 3)
    data = ["ab", "bc", "cd", "de", "ef"]
    test(data, 2)
