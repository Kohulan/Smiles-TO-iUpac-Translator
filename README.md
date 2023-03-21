[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/MIt)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)](https://github.com/Kohulan/Smiles-TO-iUpac-Translator/graphs/commit-activity)
![Workflow](https://github.com/Kohulan/Smiles-TO-iUpac-Translator/actions/workflows/Check_errors.yml/badge.svg)
[![GitHub issues](https://img.shields.io/github/issues/Kohulan/Smiles-TO-iUpac-Translator.svg)](https://GitHub.com/Kohulan/Smiles-TO-iUpac-Translator/issues/)
[![GitHub contributors](https://img.shields.io/github/contributors/Kohulan/Smiles-TO-iUpac-Translator.svg)](https://GitHub.com/Kohulan/Smiles-TO-iUpac-Translator/graphs/contributors/)
[![tensorflow](https://img.shields.io/badge/TensorFlow-2.10.1-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org)
[![GitHub release](https://img.shields.io/github/release/Kohulan/Smiles-TO-iUpac-Translator.svg)](https://GitHub.com/Kohulan/Smiles-TO-iUpac-Translator/releases/)
[![PyPI version fury.io](https://badge.fury.io/py/STOUT-pypi.svg)](https://pypi.org/project/STOUT-pypi/)
![versions](https://img.shields.io/pypi/pyversions/STOUT-pypi.svg)

![GitHub Logo](https://github.com/Kohulan/Smiles-TO-iUpac-Translator/blob/stout-1/important_assets/STOUT.png?raw=true)

# STOUT V2.0 - Smiles TO iUpac Translator Version 2.0
This repository contains STOUT-V2, SMILES to IUPAC name translator using transformers. STOUT-V2 can translate SMILES to IUPAC names and IUPAC names back to a valid SMILES string. STOUT-V1 is already published and for more details check [here](https://github.com/Kohulan/Smiles-TO-iUpac-Translator)

#### OS-Support: Linux, MACOS and Windows (On Windows you can run STOUT inside the Ubuntu shell). But It is highly recommended to use a Linux system.

# Usage

### We suggest to use STOUT inside a Conda environment, which makes the dependencies to install easily.
- Conda can be downloaded as part of the [Anaconda](https://www.anaconda.com/) or the [Miniconda](https://conda.io/en/latest/miniconda.html) plattforms (Python 3.7). We recommend to install miniconda3. Using Linux you can get it with:
```shell
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```
## How to install STOUT

## Install directly using PyPi
```shell
pip install STOUT-pypi
```

## In a conda environment
```shell
conda create --name STOUT python=3.8 
conda activate STOUT
conda install -c decimer stout-pypi
```

## Straight from the repository
```shell
pip install git+https://github.com/Kohulan/Smiles-TO-iUpac-Translator.git
```

## Simple usage
```python3

from STOUT import translate_forward, translate_reverse

# SMILES to IUPAC name translation

SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
IUPAC_name = translate_forward(SMILES)
print("IUPAC name of "+SMILES+" is: "+IUPAC_name)

# IUPAC name to SMILES translation

IUPAC_name = "1,3,7-trimethylpurine-2,6-dione"
SMILES = translate_reverse(IUPAC_name)
print("SMILES of "+IUPAC_name+" is: "+SMILES)

```

#### Happy Brewing... 🍺

## How to cite us?

Rajan, K., Zielesny, A. & Steinbeck, C. STOUT: SMILES to IUPAC names using neural machine translation. J Cheminform 13, 34 (2021). https://doi.org/10.1186/s13321-021-00512-4

## Acknowledgement
- Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC) 

<p align="center">
  <img src="https://user-images.githubusercontent.com/30716951/220350828-913e6645-6a0a-403c-bcb8-160d061d4606.png" width="500" class="center">
</p>

# STOUT-V2 is part of DECIMER project
[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/raw/master/assets/DECIMER.gif)](https://decimer.ai)

# More about Us

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png?raw=true)](https://cheminf.uni-jena.de)

![Alt](https://repobeats.axiom.co/api/embed/c66cc0ff5bc3ae91ccc8a3f7ed20eb05c735d753.svg "Repobeats analytics image")
