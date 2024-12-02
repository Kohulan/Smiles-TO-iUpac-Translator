# STOUT Training

A detailed step by step approach to STOUT Training could be found [here](https://github.com/Kohulan/IWOMI_Tutorials/tree/IWOMI_2024/STOUT_Training)! 🎉🍺

## Overview

The STOUT Training repository contains STOUT-V2, a SMILES (Simplified Molecular Input Line Entry System) to IUPAC (International Union of Pure and Applied Chemistry) name translator using transformers. STOUT-V2 can translate SMILES to IUPAC names and IUPAC names back to valid SMILES strings.

### Prerequisites

Before we begin training STOUT, let's ensure our setup is properly configured:

- **Conda Environment**: We highly recommend using Conda for seamless dependency management. If you're new to Conda, don't worry! You can download it as part of the Anaconda or Miniconda platforms. For optimal performance, we suggest installing Miniconda3.

### Installation Guide

Follow these steps to set up STOUT within a Conda environment:

1. **Clone the Repository**:
   ```shell
   git clone https://github.com/Kohulan/IWOMI_Tutorials.git
   cd IWOMI_Tutorials
   ```
2. **Create Conda Environment**:

   ```shell
   conda create --name STOUT python=3.10.0
   conda activate STOUT
   conda install pip
   python3 -m pip install -U pip
   pip install -r requirements.txt
   ```
3. **Generating Training Data**

- Run Jupyter Notebook on your terminal: `jupyter notebook`
- Open `STOUT_Training_Data_Preparation_Tokenizers.ipynb`
- This will generate the tokenizers and the split text files
- Open `STOUT_TFRecord_Generation.ipynb`
- This will generate the required TFRecord files inside the `Training_data` folder

4. **Training on TPU-VM in Google Cloud**

- Move the data to a Google Cloud Storage bucket in the same location as the TPU VM
- Copy the training script and the model file to the TPU-VM
- Install the required dependencies instead of TensorFlow
- Start the training process

**Note**: It's essential to have a powerful computation resource like a TPU (Tensor Processing Unit) to train large-scale transformer models efficiently. TPUs are specialized hardware designed by Google for accelerating machine learning workloads.

## Additional Resources

- **GitHub Repository**: The source code for STOUT-V2 is available at [https://github.com/Kohulan/Smiles-TO-iUpac-Translator](https://github.com/Kohulan/Smiles-TO-iUpac-Translator)

- **Paper Citation**: Rajan, K., Zielesny, A. & Steinbeck, C. STOUT: SMILES to IUPAC names using neural machine translation. _J Cheminform_ **13**, 34 (2021). https://doi.org/10.1186/s13321-021-00512-4

## Happy Brewing! 💡🍺

<p align="center">
  <img src="https://github.com/Kohulan/cheminf-jena-logos/blob/main/STOUT/STOUT.png" alt="STOUT Logo" width="300">
  <br>
   © 2024 Copyright Kohulan Rajan
</p>
