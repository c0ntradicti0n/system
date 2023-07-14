from random import randint


def create_random_path():
    no_len = randint(1, 6)
    path_i = [randint(1, 3) for i in range(no_len)]
    path = f"""{''.join([f"{i}" for i in path_i])}"""
    return path


def create_path(keys):
    path = f"""{''.join([f"{i}" for i in keys])}"""
    return path


class LLMFeature:
    def __init__(self, name, pattern_instruction, example):
        self.name = name.replace(" ", "_")
        self._instruction = pattern_instruction.strip()
        self._example = example.strip()

    def __call__(self, **kwargs):
        if self.name in kwargs:
            self.on = kwargs[self.name]
        else:
            raise ValueError(f"Feature {self.name} not found in Feature config")
        return self

    def instruction(self):
        if self.on:
            return self._instruction
        else:
            return ""

    def example(self):
        if self.on:
            return self._example
        else:
            return ""


class Block(LLMFeature):
    def __init__(self, **kwargs):
        super().__init__(
            "block",
            pattern_instruction=""" - add multiple things like (but with other contents)
31 {1: "Force", 2: "Motion", 3: "Energy", "_": "Inertia-Dynamics", ".": "Physics"}
32 {1: "Cell", 2: "Organism", 3: "Ecosystem", "_": "Individual-Community", ".": "Biology"}
31 {1: "Atom", 2: "Molecule", 3: "Compound", "_": "Element-Compound", ".": "Chemistry"}
 """,
            example="""
31 {1: "Force", 2: "Motion", 3: "Energy", "_": "Inertia-Dynamics", ".": "Physics"}
""",
            **kwargs,
        )


class Shift(LLMFeature):
    def __init__(self, **kwargs):
        path1 = create_random_path()
        path2 = create_random_path()
        super().__init__(
            "shift",
            """
 - shift the position of a thing:
312 <-> 213
""",
            example=f"""
{path1} <-> {path2}
""",
            **kwargs,
        )


class Replace(LLMFeature):
    def __init__(self, what, sign, example, **kwargs):
        path = create_random_path()
        super().__init__(
            what,
            f"""
- suggest alternative {what}
{path}{sign} new {what}
""",
            example=f"""
{path} "{example}"
""",
            **kwargs,
        )


features = [
    Replace("topic", ".", "Colors"),
    Replace("inversion antonym", "_", "Wave-Particle"),
    Replace("thesis", "1", "Light"),
    Replace("antithesis", "2", "Darkness"),
    Replace("synthesis", "3", "Color"),
    Block(),
    Shift(),
]
