#!/usr/bin/env python

import setuptools
import platform

if (
    platform.processor() == "arm" or platform.processor() == "i386"
) and platform.system() == "Darwin":
    tensorflow_os = "tensorflow-macos==2.15.0"
else:
    tensorflow_os = "tensorflow==2.15.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="STOUT-pypi",
    version="2.1.0",
    author="Kohulan Rajan",
    author_email="kohulan.rajan@uni-jena.de",
    maintainer="Kohulan Rajan",
    maintainer_email="kohulan.rajan@uni-jena.de",
    description="STOUT V2.0 - Smiles TO iUpac Translator Version 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kohulan/Smiles-TO-iUpac-Translator",
    packages=setuptools.find_packages(),
    license="MIT",
    install_requires=[
        tensorflow_os,
        "pystow",
        "unicodedata2",
        "jpype1",
        "rdkit",
    ],
    package_data={"STOUT": ["repack/*.*", "trainer/*.*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
