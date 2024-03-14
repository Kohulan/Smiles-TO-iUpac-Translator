# Initializing and importing necessary libararies

import tensorflow as tf
import os
import pickle
import pystow
import re
import logging
from .repack import helper

# Silence tensorflow model loading warnings.
logging.getLogger("absl").setLevel("ERROR")

# Silence tensorflow errors. optional not recommened if your model is not working properly.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Print tensorflow version
# print("Tensorflow version: " + tf.__version__)

# Always select a GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Scale memory growth as needed
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set path
default_path = pystow.join("STOUT-V2", "models")

# model download location
model_url = "https://storage.googleapis.com/decimer_weights/models.zip"
model_path = str(default_path) + "/translator_forward/"

# download models to a default location
if not os.path.exists(model_path):
    helper.download_trained_weights(model_url, default_path)


# Load the packed model forward
reloaded_forward = tf.saved_model.load(default_path.as_posix() + "/translator_forward")

# Load the packed model forward
reloaded_reverse = tf.saved_model.load(default_path.as_posix() + "/translator_reverse")


def load_forward_translation_utils():
    # Load important pickle files which consists the tokenizers and the maxlength setting
    inp_lang = pickle.load(
        open(default_path.as_posix() + "/assets/tokenizer_input.pkl", "rb")
    )
    targ_lang = pickle.load(
        open(default_path.as_posix() + "/assets/tokenizer_target.pkl", "rb")
    )
    inp_max_length = pickle.load(
        open(default_path.as_posix() + "/assets/max_length_inp.pkl", "rb")
    )
    return inp_lang, targ_lang, inp_max_length


def load_reverse_translation_utils():
    # Load important pickle files which consists the tokenizers and the maxlength setting
    targ_lang = pickle.load(
        open(default_path.as_posix() + "/assets/tokenizer_input.pkl", "rb")
    )
    inp_lang = pickle.load(
        open(default_path.as_posix() + "/assets/tokenizer_target.pkl", "rb")
    )
    inp_max_length = pickle.load(
        open(default_path.as_posix() + "/assets/max_length_targ.pkl", "rb")
    )
    return inp_lang, targ_lang, inp_max_length


def translate_forward(smiles: str, add_confidence: False) -> str:
    """Takes user input splits them into words and generates tokens.
    Tokens are then passed to the model and the model predicted tokens are retrieved.
    The predicted tokens get detokenized and the final result is returned in a string format.
    If add_confidence is true, a list of tuples is returned, where the first element is
    the token and the second element is the confidence value.

    Args:
        smiles (str): user input SMILES in string format.
        add_confidence (bool): If True, the confidence values of the predicted tokens
            are returned as well.

    Returns:
        result (str): The predicted IUPAC names in string format.
        OR
        result(List[tuples]) Tokens, confidence values
    """
    # TODO: loading this for every call is inefficient
    # --> move to init of a translator class
    inp_lang, targ_lang, inp_max_length = load_forward_translation_utils()

    if len(smiles) == 0:
        return ""
    smiles = smiles.replace("\\/", "/")
    smiles_canon = helper.get_smiles_cdk(smiles)
    if smiles_canon:
        splitted_list = list(smiles_canon)
        tokenized_SMILES = re.sub(
            r"\s+(?=[a-z])", "", " ".join(map(str, splitted_list))
        )
        decoded = helper.tokenize_input(tokenized_SMILES, inp_lang, inp_max_length)
        result_predited, confidence_array = reloaded_forward(decoded)
        if add_confidence:
            result = helper.detokenize_output_add_confidence(
                result_predited, confidence_array, targ_lang
            )
        else:
            result = helper.detokenize_output(result_predited, targ_lang)
        return result
    else:
        return "Could not generate IUPAC name for SMILES provided."


def translate_reverse(iupacname: str, add_confidence: False) -> str:
    """Takes user input splits them into words and generates tokens.
    Tokens are then passed to the model and the model predicted tokens are retrieved.
    The predicted tokens get detokenized and the final result is returned in a string format.
    If add_confidence is true, a list of tuples is returned, where the first element is
    the token and the second element is the confidence value.

    Args:
        iupacname (str): user input IUPAC names in string format.
        add_confidence (bool): If True, the confidence values of the predicted tokens
            are returned as well.

    Returns:
        result (str): The predicted SMILES in string format.
        OR
        result(List[tuples]) Tokens, confidence values
    """
    # TODO: loading this for every call is inefficient
    # --> move to init of a translator class
    inp_lang, targ_lang, inp_max_length = load_reverse_translation_utils()

    splitted_list = list(iupacname)
    tokenized_IUPACname = " ".join(map(str, splitted_list))
    decoded = helper.tokenize_input(tokenized_IUPACname, inp_lang, inp_max_length)

    result_predited, confidence_array = reloaded_forward(decoded)
    if add_confidence:
        result = helper.detokenize_output_add_confidence(
            result_predited, confidence_array, targ_lang
        )
    else:
        result = helper.detokenize_output(result_predited, targ_lang)
    return result
