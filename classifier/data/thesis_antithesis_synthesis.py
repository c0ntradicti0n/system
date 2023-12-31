import ast
import csv
import random

from lib.interleave_generators import interleave_generators

extra = [
    ("plus and minus", "multiplication and division", "exponent and root"),
    ("addition and subtraction", "multiplication and division", "exponential and logarithm"),
    ("linear operations", "multiplicative operations", "exponential and logarithmic functions"),
    ("to be", "not to be", "become"),
    ("friend", "enemy", "diplomat"),
    ("up", "down", "straight"),
    ("plus", "minus", "plus minus zero"),
    ("black", "white", "grey"),
    ("able", "unable", "ability"),
    ("bad", "good", "neutral"),
    ("true", "false", "probable"),
    ("beautiful", "ugly", "okay"),
    ("warm", "cold", "neutral"),
    ("gas", "solid", "liquid"),
    ("plant", "fungus", "animal"),
    ("bacteria", "virus", "parasite"),
    ("white blood cell", "red blood cell", "platelet"),
    ("neuron", "axon", "dendrite"),
    ("nucleus", "mitochondria", "ribosome"),
    ("proton", "electron", "neutron"),
    ("atom nucleus", "electron cloud", "metal"),
    ("time", "space", "motion"),
    ("point", "line", "plane"),
    ("triangle", "circle", "polygon"),
    ("square", "cube", "hypercube"),
    ("sphere", "cylinder", "cone"),
    ("rectangle", "parallelogram", "trapezoid"),
    ("thesis", "antithesis", "synthesis"),
    ("general", "specific", "particular"),
    ]
samples = (extra)
