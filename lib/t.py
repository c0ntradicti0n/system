from time import perf_counter

indent = 0


class catchtime:
    def __init__(self, task):
        self.task = task

    def __enter__(self):
        global indent
        self.start = perf_counter()
        indent += 1
        return self

    def __exit__(self, type, value, traceback):
        global indent
        self.time = perf_counter() - self.start
        self.readout = f"{self.time:.3f}s" + "  " * indent + f" {self.task}"
        print(self.readout)
        indent -= 1


class indented:
    def __init__(self, task):
        self.task = task

    def __enter__(self):
        global indent
        indent += 1
        return self

    def __exit__(self, type, value, traceback):
        global indent
        self.readout = f"      " + "  " * indent + f" {self.task}"
        print(self.readout)
        indent -= 1
