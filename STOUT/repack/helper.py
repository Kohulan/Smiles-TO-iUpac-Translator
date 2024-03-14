import os
import tensorflow as tf
import tensorflow.keras as keras
import re
import unicodedata
import numpy as np
import pystow
from typing import List, Tuple, Dict
import zipfile
from rdkit import Chem


def get_canonical(smiles: str) -> str:
    """
    Generate canonical SMILES representation from a given SMILES string.

    This function takes a SMILES string as input and generates its canonical representation using the RDKit library.
    If the input SMILES string is valid, it returns the canonical SMILES representation; otherwise, it returns
    "Invalid SMILES string".

    Args:
        smiles (str): The input SMILES string.

    Returns:
        str: The canonical SMILES representation of the input molecule.

    Note:
        - The canonical SMILES representation is generated using RDKit's MolToSmiles function with kekuleSmiles
          set to True and isomericSmiles set to True.
        - If the input SMILES string is invalid (cannot be converted to a molecule), the function returns
          None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        canonicalSMILES = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)
        return canonicalSMILES
    else:
        return None


def preprocess_input(input_string: str) -> str:
    """
    Preprocess input string for a natural language processing task.

    This function takes an input string and adds special start and end tokens ("<start>" and "<end>") to it to
    indicate the beginning and end of the input sequence for a natural language processing task.

    Args:
        input_string (str): The input string to be preprocessed.

    Returns:
        str: The preprocessed input string with start and end tokens added.
    """
    input_string = "<start> " + input_string + " <end>"
    return input_string


def split_character(SMILES: str) -> str:
    """
    Split characters in the canonical SMILES representation.

    This function takes a canonical SMILES string as input, splits individual characters, and returns a tokenized
    version of the SMILES string.

    Args:
        SMILES (str): The canonical SMILES representation of a molecule.

    Returns:
        str: The tokenized version of the canonical SMILES string.

    Note:
        - This function first replaces backslash escape sequences with slashes to normalize the input.
        - It then generates the canonical SMILES representation using the get_canonical function.
        - Next, it splits the canonical SMILES string into individual characters and removes any whitespace
          followed by lowercase letters.
        - The resulting tokenized SMILES string is returned.
    """
    SMILES = SMILES.replace("\\/", "/")
    canonical_SMILES = get_canonical(SMILES)
    if canonical_SMILES:
        splitted_list = list(canonical_SMILES)
        tokenized_SMILES = re.sub(
            r"\s+(?=[a-z])", "", " ".join(map(str, splitted_list))
        )
        return tokenized_SMILES


def tokenize_input(input_SMILES: str, inp_lang, inp_max_length: int) -> np.array:
    """This function takes a user input SMILES and tokenizes it
       to feed it to the model.

    Args:
        input_SMILES (string): SMILES string given by the user.
        inp_lang: keras_preprocessing.text.Tokenizer object with input language.
        inp_max_length: maximum number of characters in the input language.

    Returns:
        tokenized_input (np.array): The SMILES get split into meaningful chunks
        and gets converted into meaningful tokens. The tokens are arrays.
    """
    sentence = preprocess_input(input_SMILES)
    inputs = [inp_lang.word_index[i] for i in sentence.split(" ")]
    tokenized_input = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=inp_max_length, padding="post"
    )

    return tokenized_input


def detokenize_output(predicted_array: np.array, targ_lang) -> str:
    """
    Detokenize the predicted output array.

    This function takes a predicted output array, typically generated during natural language processing tasks,
    and detokenizes it using the target language index to word mapping. It removes special tokens like "<start>"
    and "<end>", and replaces any custom tokens such as "§" with spaces.

    Args:
        predicted_array (np.array): The predicted output array, usually containing token indices.
        targ_lang: The target language mapping from index to word.

    Returns:
        str: The detokenized prediction as a single string.

    Note:
        - The function iterates through the predicted array and maps each token index to its corresponding word
          using the target language mapping.
        - Special tokens "<start>" and "<end>" are removed from the detokenized output.
        - Custom tokens, such as "§", are replaced with spaces.
    """
    outputs = [targ_lang.index_word[i] for i in predicted_array[0].numpy()]
    prediction = (
        "".join([str(elem) for elem in outputs])
        .replace("§", " ")
        .replace("<start>", "")
        .replace("<end>", "")
    )
    return prediction


def detokenize_output_add_confidence(
    predicted_array: tf.Tensor,
    confidence_array: tf.Tensor,
    targ_lang: Dict,
) -> List[Tuple[str, float]]:
    """
    This function takes the predicted array of tokens as well as the confidence values
    returned by the Transformer Decoder and returns a list of tuples
    that contain each token and the confidence value.

    Args:
        predicted_array (tf.Tensor): Transformer Decoder output array (predicted tokens)

    Returns:
        str: SMILES string
    """
    prediction_with_confidence = [
        (
            targ_lang.index_word[predicted_array[0].numpy()[i]],
            confidence_array[i].numpy(),
        )
        for i in range(len(confidence_array))
    ]
    prediction_with_confidence_ = prediction_with_confidence[1:-1]
    return prediction_with_confidence_


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# Downloads the model and unzips the file downloaded, if the model is not present on the working directory.
def download_trained_weights(model_url: str, model_path: str, verbose=1):
    """This function downloads the trained models and tokenizers to a default location.
    After downloading the zipped file the function unzips the file automatically.
    If the model exists on the default location this function will not work.

    Args:
        model_url (str): trained model url for downloading.
        model_path (str): model default path to download.

    Returns:
        downloaded model.
    """
    # Download trained models
    if verbose > 0:
        print("Downloading trained model to " + str(model_path))
        model_path = pystow.ensure("STOUT-V2", url=model_url)
        print(model_path)
    if verbose > 0:
        print("... done downloading trained model!")
        with zipfile.ZipFile(model_path.as_posix(), "r") as zip_ref:
            zip_ref.extractall(model_path.parent.as_posix())
