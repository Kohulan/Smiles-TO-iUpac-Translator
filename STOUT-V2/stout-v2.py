# Initializing and importing necessary libararies

import tensorflow as tf
import os
import pickle
import argparse
import re
import time
import numpy as np
import helper

# Print tensorflow version
print(tf.__version__)

# Always select a GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Scale memory growth as needed
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load important pickle files which consists the tokenizers and the maxlength setting
inp_lang = pickle.load(open("tokenizer_input.pkl", "rb"))
targ_lang = pickle.load(open("tokenizer_target.pkl", "rb"))
inp_max_length = pickle.load(open("max_length_inp.pkl", "rb"))

# Load the packed model 
reloaded = tf.saved_model.load('translator_sig')

# Create a file out
f_out = open("Predictions__" + str(file_in) + ".txt", "w")

def tokenize_input(input_SMILES: str) -> np.array:
    """This function takes a user input SMILES and tokenizes it 
       to feed it to the model.

    Args:
        input_SMILES (string): SMILES string given by the user.

    Returns:
        tokenized_input (np.array): The SMILES get split into meaningful chunks 
        and gets converted into meaningful tokens. The tokens are arrays.
    """
    sentence = helper.preprocess_sentence(input_SMILES)
    inputs = [inp_lang.word_index[i] for i in sentence.split(" ")]
    tokenized_input = tf.keras.preprocessing.sequence.pad_sequences(
            [inputs], maxlen=inp_max_length, padding="post"
        )
    
    return tokenized_input

def detokenize_output(predicted_array: np.array) -> str:
    """This function takes a predited input array and returns
       a IUPAC name by detokenizing the input.

    Args:
        predicted_array (np.array): The predicted_array is returned by the model.

    Returns:
        prediction (str): The predicted array gets detokenized by the tokenizer, 
        The unnessary spaces, start and the end tokens will bve removed and 
        a proper IUPAC name is returned in a string format. 
    """
    outputs = [targ_lang.index_word[i] for i in predicted_array[0].numpy()]
    prediction = ' '.join([str(elem) for elem in outputs]).replace("<start> ","").replace(" <end>","").replace(" ","")
    
    return prediction

def translate(sentence_input:str) -> str:
    """Takes user input splits them into words and generates tokens.
    Tokens are then passed to the model and the model predicted tokens are retrieved.
    The predicted tokens gets detokenized and the final result is returned in a string format.

    Args:
        sentence_input (str): user input SMILES in string format.

    Returns:
        result (str): The predicted IUPAC names in string format.
    """

    splitted_list = list(sentence_input)
    Tokenized_SMILES = re.sub(r'\s+(?=[a-z])','',' '.join(map(str, splitted_list)))
    decoded = tokenize_input(Tokenized_SMILES)

    result = reloaded(decoded)

    return result


