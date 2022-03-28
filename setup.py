#!/usr/bin/env python3
from setuptools import setup

setup(
    name='lanenet',
    version='0.1',
    description='LaneNet',
    author='MaybeShewill-CV',
    maintainer='Chiao Hsieh',
    maintainer_email='chsieh16@illinois.edu',
    license='Apache-2.0',
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'opencv_contrib_python',
        'scikit_learn==0.24.1',
        'tensorflow>=2.2',
        'tensorflow_gpu>=2.2',
        'PyYaml',
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache License 2.0',
        'Programming Language :: Python :: 3.8',
    ]
)
