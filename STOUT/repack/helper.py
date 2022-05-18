import tensorflow as tf
import tensorflow.keras as keras
import re
import unicodedata
import numpy as np
import pystow
import subprocess

# Converts the unicode file to ascii
def unicode_to_ascii(s: str) -> str:
    """Converts a unicode string to an ASCII string

    Args:
        s (str): Takes a string in unicode format.

    Returns:
        str: returns a ASCII formatted string.
    """

    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def preprocess_sentence(w: str) -> str:
    """Takes in a sentence, removes white spaces and generates a clean sentence. At the begining of the sentesnce a <start> token will be added
    and at the end an <end> token will be added and the modified sentence will be returned.

    Args:
        w (str): input sentence to be modified.

    Returns:
        str: modified sentence with start and end tokens.
    """
    w = unicode_to_ascii(w.strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    # w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = "<start> " + w + " <end>"
    return w


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
    sentence = preprocess_sentence(input_SMILES)
    inputs = [inp_lang.word_index[i] for i in sentence.split(" ")]
    tokenized_input = keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=inp_max_length, padding="post"
    )

    return tokenized_input


def detokenize_output(predicted_array: np.array, targ_lang) -> str:
    """This function takes a predited input array and returns
       a IUPAC name by detokenizing the input.

    Args:
        predicted_array (np.array): The predicted_array is returned by the model.
        targ_lang: keras_preprocessing.text.Tokenizer object with target language.

    Returns:
        prediction (str): The predicted array gets detokenized by the tokenizer,
        The unnessary spaces, start and the end tokens will bve removed and
        a proper IUPAC name is returned in a string format.
    """
    outputs = [targ_lang.index_word[i] for i in predicted_array[0].numpy()]
    prediction = (
        " ".join([str(elem) for elem in outputs])
        .replace("<start> ", "")
        .replace(" <end>", "")
        .replace(" ", "")
    )

    return prediction


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
        subprocess.run(
            [
                "unzip",
                model_path.as_posix(),
                "-d",
                model_path.parent.as_posix(),
            ]
        )
