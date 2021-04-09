# -*- coding: utf-8 -*-
"""Setup file for torchrunner PyTorch deep learning wrapper.
"""

from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="torchrunner",
    version="0.1.0",
    description="PyTorch deep learning wrapper",
    license="GPLv3",
    long_description=long_description,
    author="John Lloyd",
    author_email="jlloyd237@gmail.com",
    url="https://github.com/jlloyd237/torchrunner/",
    packages=["torchrunner"],
    install_requires=["numpy", "scipy", "torch", "PIL", "opencv-python"]
)
