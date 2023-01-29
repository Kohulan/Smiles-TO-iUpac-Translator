#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="STOUT-pypi",
    version="2.0.2",
    author="Kohulan Rajan",
    author_email="kohulan.rajan@uni-jena.de",
    maintainer="Kohulan Rajan",
    maintainer_email="kohulan.rajan@uni-jena.de",
    description="STOUT V2.0 - Smiles TO iUpac Translator Version 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kohulan/STOUT-2",
    packages=setuptools.find_packages(),
    license="MIT",
    install_requires=[
        "tensorflow==2.10.0",
        "pystow",
        "unicodedata2",
        "rdkit-pypi",
        "jpype1",
    ],
    package_data={"STOUT": ["repack/*.*", "trainer/*.*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
)
