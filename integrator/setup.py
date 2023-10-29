#!/usr/bin/env python
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize(
        "tree.py", compiler_directives={"language_level": "3"}, annotate=True
    )
)
