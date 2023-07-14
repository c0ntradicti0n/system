from collections import Counter, defaultdict


def analyse(tree, exclude):
    stack = [((), tree)]
    len_counter = defaultdict(list)
    while stack:
        path, current = stack.pop()
        for k, v in current.items():
            if isinstance(v, dict):
                stack.append((path + (k,), v))
            else:
                keys = "".join([str(p) for p in path])
                if any(keys.startswith(e) for e in exclude):
                    continue
                if any(isinstance(vv, dict) for vv in current.values()):
                    continue
                len_counter[len(path)].append(keys)

    len_counter = {k: list(sorted(set(v))) for k, v in len_counter.items() if k}

    return {
        "min_depth_paths": len_counter[min(len_counter)],
        "max_depth_paths": len_counter[max(len_counter)],
        "len_counter": len_counter,
    }


if __name__ == "__main__":
    tree = {
        "1": {
            "1": {
                "1": {
                    ".": "Ontology.md",
                    "1": "Being.md",
                    "2": "Nothing.md",
                    "3": "Becoming.md",
                    "_": "Disappeareance-of-Disappearing.md",
                },
                "2": {
                    "1": {
                        ".": "Qualitative Determination.md",
                        "1": "Determinate Being (Dasein).md",
                        "2": "Negation.md",
                        "3": "Something.md",
                        "_": "Negation-of-Negation.md",
                    },
                    "2": {
                        ".": "Relativity & Comparison.md",
                        "1": "Otherness.md",
                        "2": "Alteration.md",
                        "3": "Limit.md",
                        "_": "Transcendence-of-Limit.md",
                    },
                    "3": "Nothingness.md",
                    ".": "Quality.md",
                },
                "3": {
                    "1": {
                        ".": "Wholeness.md",
                        "1": "Unity.md",
                        "2": "Plurality.md",
                        "3": "Totality.md",
                        "_": "Part-of-Whole.md",
                    },
                    "2": {
                        "1": {
                            "1": {".": "Equation.md"},
                            "2": {"_": "Minus-times-Minus is Plus.md"},
                            "3": {
                                ".": "Exponential and Root.md",
                                "1": "Power.md",
                                "2": "Root.md",
                                "3": "Exponential.md",
                                "_": "0 to power 1 and 1 to power 0.md",
                            },
                            ".": "Calculus.md",
                        },
                        "2": {
                            "1": {
                                "1": {
                                    "1": {".": "Addition.md"},
                                    "2": {".": "Substraction.md"},
                                    "3": {
                                        ".": "a+b=a--b.md",
                                        "_": "Loss of Loss is status quo.md",
                                    },
                                    ".": "Line Calculation.md",
                                },
                                "2": {
                                    "1": {".": "Multiplication.md"},
                                    "2": {".": "Division.md"},
                                    "3": {
                                        ".": "Neutral Element 1.md",
                                        "_": "Minus times Minus is Plus.md",
                                    },
                                    ".": "Dot Calculation.md",
                                },
                                "3": {
                                    "1": {".": "Exponentiation.md"},
                                    "2": {".": "Root Extraction.md"},
                                    "3": {".": "Logarithm.md", "_": "0^1 and 1^0.md"},
                                    ".": "Exponentiation.md",
                                },
                            },
                            "2": {
                                "1": {".": "Integration.md"},
                                "2": {".": "Differentiation.md"},
                                "3": {
                                    "1": {".": "integration of e^x = e^x.md"},
                                    "2": {".": "d:dx ln(x)=1:x.md"},
                                    "3": {
                                        ".": "e for exponents, roots and logarithms.md .md"
                                    },
                                    ".": "Neutral element e.md",
                                },
                            },
                            ".": "Calculate.md",
                        },
                        "3": {".": "Analysis.md"},
                        ".": "Algebra.md",
                    },
                    "3": {".": "Mathematics.md", "_": "Quantitative-qualitative.md"},
                    ".": "Quantity.md",
                },
                ".": "Categories.md",
            },
            "2": {
                ".": "Ontology.md",
                "_": "Minus-is-plus.md",
                "1": {
                    "_": "Absence-Presence",
                    ".": "Ontology",
                    "1": {
                        "1": "Being",
                        "2": "Becoming",
                        "3": "Substance",
                        "_": "Change-Static",
                        ".": "Metaphysics",
                    },
                    "2": {
                        "1": "Time",
                        "2": "Space",
                        "3": "Event",
                        "_": "Continuum-Instants",
                        ".": "Phenomenology",
                    },
                    "3": {
                        "1": "Potential",
                        "2": "Actual",
                        "3": "Virtual",
                        "_": "Manifestation-Hidden",
                        ".": "Modal Logic",
                    },
                },
                "2": {
                    "_": "Continuous-Discrete",
                    ".": "Mathematics",
                    "1": {
                        "1": "Number",
                        "2": "Set",
                        "3": "Function",
                        "_": "Constant-Variation",
                        ".": "Arithmetic",
                    },
                    "2": {
                        "1": "Dimensions",
                        "2": "Transformations",
                        "3": "Topology",
                        "_": "Linear-Nonlinear",
                        ".": "Geometry",
                    },
                    "3": {
                        "1": "Scale",
                        "2": "Ratio",
                        "3": "Magnitude",
                        "_": "Relative-Absolute",
                        ".": "Algebra",
                    },
                },
                "3": {
                    "_": "Whole-Parts",
                    ".": "Structuralism",
                    "1": {
                        "1": "Element",
                        "2": "Compound",
                        "3": "Mixture",
                        "_": "Homogenous-Heterogenous",
                        ".": "Chemistry",
                    },
                    "2": {
                        "1": "Organism",
                        "2": "Cell",
                        "3": "DNA",
                        "_": "Complex-Simple",
                        ".": "Biology",
                    },
                    "3": {
                        "1": "Solar system",
                        "2": "Galaxy",
                        "3": "Universe",
                        "_": "Finite-Infinite",
                        ".": "Cosmology",
                    },
                },
            },
            "3": {
                "1": {
                    "1": "Fundamental forces.md",
                    "2": "Fundamental particles.md",
                    "3": "Fields.md",
                    ".": "Quantum field theory.md",
                    "_": "Force-Field.md",
                },
                "2": {
                    "1": "Operations.md",
                    "2": "Variables.md",
                    "3": "Functions.md",
                    ".": "Linear Algebra.md",
                    "_": "Opal-Function.md",
                },
                "3": "Gravity.md",
                ".": "Physical Cosmology.md",
                "_": "Density-Vacuum.md",
            },
        },
        "2": {
            "1": {
                "3": "Ethics.md",
                ".": "Explicit Aspects.md",
                "1": "Morality.md",
                "2": "Amorality.md",
                "_": "Ethic-Amoral.md",
            },
            "2": {
                ".": "Aesthetics.md",
                "1": "Beauty.md",
                "2": "Ugliness.md",
                "3": "Sublime.md",
                "_": "Ugly-Beauty.md",
            },
            "3": {
                ".": "Epistemology.md",
                "1": "Knowledge.md",
                "2": "Ignorance.md",
                "3": "Wisdom.md",
                "_": "Wisdom of Ignorance.md",
            },
        },
        "3": {
            "1": {
                ".": "Physics of light.md",
                "1": "Light.md",
                "2": "Darkness.md",
                "3": "Colors.md",
            },
            "2": {
                "1": {
                    ".": "Epistemology.md",
                    "1": "Subjectivity.md",
                    "2": "Objectivity.md",
                    "3": "Inter-subjectivity.md",
                    "_": "Subject-object-switch.md",
                },
                "2": {
                    ".": "Methodology.md",
                    "1": "Empiricism.md",
                    "2": "Rationalism.md",
                    "3": "Critical theory.md",
                    "_": "Fact-Theory.md",
                },
                "3": {
                    ".": "Information theory.md",
                    "1": "Analogue.md",
                    "2": "Digital.md",
                    "3": "Quantum computing.md",
                    "_": "Binary-Superposition.md",
                },
            },
            "3": {
                "1": {
                    ".": "Computational science.md",
                    "1": "Quantum computing.md",
                    "2": "Classical computing.md",
                    "3": "Post-quantum cryptography.md",
                    "_": "Superposition-Determinism.md",
                },
                "2": {
                    ".": "Musical acoustics.md",
                    "1": "Melodic interval.md",
                    "2": "Harmonic interval.md",
                    "3": "Compound interval.md",
                    "_": "Consonance-Dissonance.md",
                },
                "3": {
                    ".": "Physics of sound.md",
                    "1": "Sound.md",
                    "2": "Silence.md",
                    "3": "Melody.md",
                    "_": "Absence-Presence.md",
                },
            },
        },
        ".": "System.md",
    }

    result = analyse(tree, exclude=["111"])
    assert result["min_depth_paths"] == ["11", "12", "13", "21", "22", "23", "31"]
    assert result["max_depth_paths"] == [
        "11322111",
        "11322112",
        "11322113",
        "11322121",
        "11322122",
        "11322123",
        "11322131",
        "11322132",
        "11322133",
        "11322231",
        "11322232",
        "11322233",
    ]
    assert min(result["len_counter"]) == 2
    assert max(result["len_counter"]) == 8
