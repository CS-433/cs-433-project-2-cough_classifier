import sys
from setuptools import setup, find_packages

import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name='cc',
    version='1.0.0',
    packages=['src', 'src.models', 'src.utils'],
    python_requires='>=3.6',
)
