from time import perf_counter


class catchtime:
    def __init__(self, task):
        self.task = task

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f"Time {self.task}: {self.time:.3f} seconds"
        print(self.readout)
