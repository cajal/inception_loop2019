#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='inception_loop',
    version='0.0.1',
    description='Inception loop code for Walker et al. 2019 Nature Neuro',
    author='Edgar. Y. Walker, Fabian H. Sinz',
    author_email='eywalker@bcm.edu',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'scipy>=1.2', 'tqdm', 'gitpython', 'scikit-image',
                      'datajoint', 'h5py', ],
)
