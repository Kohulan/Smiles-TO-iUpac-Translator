# STOUT: Smiles TO iUpac Translator
STOUT: SMILES TO iUpac Translator is built using the same concept as a Neural Machine Translation(NMT). STOUT is initially trained on a subset downloaded from Pubchem[1] containing 30 Million SMILES[2] and 60 Million SMILES. which got converted into SELFIES using the SELFIES package. The same set of SMILES also was converted into IUPAC names using ChemAxon "molconvert", a command-line program in Marvin Suite 20.15 from ChemAxon (https://www.chemaxon.com)[3]. Later the textual data was converted into TFRecords(Binary files) for training on Tensor Processing Units(TPUs).
![GitHub Logo](https://github.com/Kohulan/Smiles-TO-iUpac-Translator/blob/main/important_assets/STOUT.png?raw=true)
# Summary of the work.

- We currently don’t have any open-source software to generate IUPAC names for a given molecule. To do that we came up with an idea to use a machine learning model based on the Neural Machine Translation. Our models can translate any given Canonical SMILES to IUPAC name and back.

- We used BLEU scoring [4] for the accuracy calculation.
https://en.wikipedia.org/wiki/BLEU
https://www.nltk.org/_modules/nltk/translate/bleu_score.html

- Also we back-translated the IUPAC names to SMILES using OPSIN[5] for further statistical evaluation.
OPSIN: Open Parser for Systematic IUPAC nomenclature
https://github.com/dan2097/opsin

### OS-Support: Linux and MACOS (On windows make sure you have installed and configured tensorflow properly)

# Usage

```
$ git clone https://github.com/Kohulan/SMILES-to-IUPAC-Translator.git
$ cd SMILES-to-IUPAC-Translator
$ conda create --name STOUT python=3.7.9
$ conda activate STOUT
$ conda install pip
$ pip install tensorflow-gpu==2.3.0 selfies matplotlib unicodedata2 
```
### Install tensorflow==2.3.0 if you do not have a nVidia GPU (On Mac OS)

```
$ pip install tensorflow==2.3.0 selfies matplotlib unicodedata2
$ python3 STOUT_V_2.0 --help #Use for help
```
- When you run the program for the first time the models will get automatically downloaded(Note: total size is ~4GB). Also you can manually download the models from [here](https://storage.googleapis.com/iupac_models_trained/Trained_model/Trained_model.zip)
```
e.g.: 

python3 STOUT_V_2.0.py 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' #SMILES to IUPAC
python3 STOUT_V_2.0.py --iupac '1,3,7-trimethyl-2,3,6,7-tetrahydro-1H-purine-2,6-dione' #IUPAC to SMILES
```

## How to cite us?

- Todo 

# References

1. Kim S, Chen J, Cheng T, et al (2019) PubChem 2019 update: improved access to chemical data. Nucleic Acids Res 47:D1102–D1109.
2. Weininger D (1988) SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. J Chem Inf Comput Sci 28:31–36.
3. ChemAxon - Software Solutions and Services for Chemistry & Biology. https://www.chemaxon.com. Accessed 23 Nov 2020.
4. Papineni K, Roukos S, Ward T, Zhu W-J (2002) BLEU: a method for automatic evaluation of machine translation. In: Proceedings of the 40th annual meeting of the Association for Computational Linguistics. pp 311–318.
5. Lowe DM, Corbett PT, Murray-Rust P, Glen RC (2011) Chemical name to structure: OPSIN, an open source solution. J Chem Inf Model 51:739–753.

# STOUT is part of DECIMER project
[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/raw/master/assets/DECIMER.gif)](https://kohulan.github.io/Decimer-Official-Site/)

# More about Us

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png?raw=true)](https://cheminf.uni-jena.de)
