<h1 align="center">
  <br>
  <a href="https://github.com/Kohulan/Smiles-TO-iUpac-Translator"><img src="https://github.com/Kohulan/Smiles-TO-iUpac-Translator/blob/development/docs/_static/STOUT.png?raw=true" alt="STOUT Logo" width="400"></a>
  <br>
V2.0
  <br>
</h1>

<h4 align="center">Smiles TO iUpac Translator: Advanced Chemical Nomenclature Translation</h4>

<p align="center">
  <a href="https://opensource.org/licenses/MIt">
    <img src="https://img.shields.io/badge/License-MIT%202.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/Kohulan/Smiles-TO-iUpac-Translator/graphs/commit-activity">
    <img src="https://img.shields.io/badge/Maintained%3F-yes-blue.svg" alt="Maintenance">
  </a>
  <a href="https://github.com/Kohulan/Smiles-TO-iUpac-Translator/actions/workflows/Check_errors.yml">
    <img src="https://github.com/Kohulan/Smiles-TO-iUpac-Translator/actions/workflows/Check_errors.yml/badge.svg" alt="Workflow">
  </a>
  <br>
  <a href="https://GitHub.com/Kohulan/Smiles-TO-iUpac-Translator/issues/">
    <img src="https://img.shields.io/github/issues/Kohulan/Smiles-TO-iUpac-Translator.svg" alt="GitHub issues">
  </a>
  <a href="https://GitHub.com/Kohulan/Smiles-TO-iUpac-Translator/graphs/contributors/">
    <img src="https://img.shields.io/github/contributors/Kohulan/Smiles-TO-iUpac-Translator.svg" alt="GitHub contributors">
  </a>
  <a href="https://www.tensorflow.org">
    <img src="https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00.svg?style=flat&logo=tensorflow" alt="tensorflow">
  </a>
  <br>
  <a href="https://GitHub.com/Kohulan/Smiles-TO-iUpac-Translator/releases/">
    <img src="https://img.shields.io/github/release/Kohulan/Smiles-TO-iUpac-Translator.svg" alt="GitHub release">
  </a>
  <a href="https://pypi.org/project/STOUT-pypi/">
    <img src="https://badge.fury.io/py/STOUT-pypi.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/STOUT-pypi/">
    <img src="https://img.shields.io/pypi/pyversions/STOUT-pypi.svg" alt="Python versions">
  </a>
  <a href="https://doi.org/10.5281/zenodo.13318286">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.13318286.svg" alt="DOI">
  </a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#acknowledgements">Acknowledgements</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://github.com/Kohulan/Smiles-TO-iUpac-Translator/raw/master/docs/_static/STOUT_demo.gif" alt="STOUT Demo">
</p>

## Key Features

<ul>
  <li>🧪 Translate SMILES to IUPAC names</li>
  <li>🔬 Convert IUPAC names back to valid SMILES strings</li>
  <li>🤖 Powered by advanced transformer models</li>
  <li>💻 Cross-platform support (Linux, macOS, Windows via Ubuntu shell)</li>
  <li>🚀 High-performance chemical nomenclature translation</li>
</ul>

## Installation

<p>Choose your preferred installation method:</p>

<details>
<summary><b>📦 PyPI Installation</b></summary>
<pre><code>pip install STOUT-pypi</code></pre>
</details>

<details>
<summary><b>🐍 Conda Environment Setup</b></summary>
<pre><code>conda create --name STOUT python=3.10 
conda activate STOUT
conda install -c decimer stout-pypi</code></pre>
</details>

<details>
<summary><b>📥 Direct Repository Installation</b></summary>
<pre><code>pip install git+https://github.com/Kohulan/Smiles-TO-iUpac-Translator.git</code></pre>
</details>

## How To Use

```python
from STOUT import translate_forward, translate_reverse

# SMILES to IUPAC name translation
SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
IUPAC_name = translate_forward(SMILES)
print(f"🧪 IUPAC name of {SMILES} is: {IUPAC_name}")

# IUPAC name to SMILES translation
IUPAC_name = "1,3,7-trimethylpurine-2,6-dione"
SMILES = translate_reverse(IUPAC_name)
print(f"🔬 SMILES of {IUPAC_name} is: {SMILES}")
```

## Acknowledgements

<p align="center">
  Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)
  <br><br>
  <img src="https://user-images.githubusercontent.com/30716951/220350828-913e6645-6a0a-403c-bcb8-160d061d4606.png" width="200">
</p>

<h2 align="center">Part of the DECIMER Project</h2>

<p align="center">
  <a href="https://decimer.ai">
    <img src="https://github.com/Kohulan/DECIMER-Image-to-SMILES/raw/master/assets/DECIMER.gif" alt="DECIMER Logo" width="400">
  </a>
</p>

<h2 align="center">About Us</h2>

<p align="center">
  <a href="https://cheminf.uni-jena.de">
    <img src="https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png?raw=true" alt="Cheminformatics and Computational Metabolomics Group" width="300">
  </a>
</p>

## Citation

<blockquote>
Rajan, K., Zielesny, A. & Steinbeck, C. STOUT: SMILES to IUPAC names using neural machine translation. J Cheminform 13, 34 (2021). https://doi.org/10.1186/s13321-021-00512-4
</blockquote>

## Model Card

<blockquote>
Rajan, K., Steinbeck, C., & Zielesny, A. (2024). STOUT V2 - Model library. Zenodo. https://doi.org/10.5281/zenodo.13318286
</blockquote>

<h2 align="center">Repository Analytics</h2>

<p align="center">
  <img src="https://repobeats.axiom.co/api/embed/c66cc0ff5bc3ae91ccc8a3f7ed20eb05c735d753.svg" alt="Repobeats analytics image">
</p>

<hr>

<p align="center">
  Made with ❤️ by the <a href="https://cheminf.uni-jena.de">Steinbeck Group</a> 
</p>
