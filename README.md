# STOUT: Smiles TO iUpac Translator
STOUT: Smiles TO iUpac Translator is built using the same concept as a Neural Machine Translation(NMT). STOUT is initially trained on a subset downloaded from Pubchem containing 30Million SMILES. which got converted into SELFIES using the SELFIES package. The same set of SMILES also was converted into IUPAC names using ChemAxon molconvert. Later the textual data was converted into TFRecords(Binary files) for training on Tensor Processing Units(TPUs).
![GitHub Logo](https://github.com/Kohulan/Smiles-TO-iUpac-Translator/blob/main/important_assets/STOUT.png?raw=true)
# Summary of the work.

- We currently don’t have any open-source software to generate IUPAC names for a given molecule. To do that we came up with an idea to use a machine learning model based on the Neural Machine Translation. Our model can translate any given Canonical SMILES to IUPAC.

- We used BLEU scoring for the accuracy calculation.
https://en.wikipedia.org/wiki/BLEU
https://www.nltk.org/_modules/nltk/translate/bleu_score.html

- Also we back-translated the IUPAC names to SMILES using OPSIN for further statistical evaluation.
OPSIN: Open Parser for Systematic IUPAC nomenclature
https://github.com/dan2097/opsin

### OS-Support: Linux and MACOS (On windows make sure you have installed and configured tensorflow properly)

# Usage

```
$ git clone https://github.com/Kohulan/SMILES-to-IUPAC-Translator.git
$ cd SMILES-to-IUPAC-Translator
```
- Download the model from [here](https://storage.googleapis.com/iupac_models_trained/Trained_model/Trained_model.zip) and uncompress the zipped folder and save the contents Trained_model directory 
```
$ conda create --name STOUT python=3.7.9
$ conda activate STOUT
$ conda install pip
$ pip install tensorflow-gpu==2.3.0 selfies matplotlib unicodedata2 #Install tensorflow==2.3.0 if you do not have a nVidia GPU
$ python3 STOUT_V_1.0 --help #Use for help
e.g. python3 STOUT_V_1.0.py 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
```

# STOUT is part of DECIMER project
[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/raw/master/assets/DECIMER.gif)](https://kohulan.github.io/Decimer-Official-Site/)

# More about Us

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png?raw=true)](https://cheminf.uni-jena.de)
