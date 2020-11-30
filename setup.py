import sys
from setuptools import setup, find_packages

import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name='cc',
    version='1.0.0',
    packages=['src', 'src.models', 'src.utils', 'crnn_audio', 'crnn_audio.eval', 'crnn_audio.train', 'crnn_audio.net',
              'crnn_audio.utils', 'crnn_audio.data'],
    python_requires='>=3.6',
)
