# Initializing and importing necessary libararies

import tensorflow as tf
import os
import pickle
import pystow
import logging
from .repack import helper

# Silence tensorflow model loading warnings.
logging.getLogger("absl").setLevel("ERROR")

# Silence tensorflow errors. optional not recommened if your model is not working properly.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Always select a GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Scale memory growth as needed
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set path
default_path = pystow.join("STOUT-V2", "models")

# model download location
model_url = "https://zenodo.org/records/12542360/files/models.zip?download=1"
model_path = str(default_path) + "/translator_forward/"
print(model_path)
# download models to a default location
if not os.path.exists(model_path):
    helper.download_trained_weights(model_url, default_path)


# Load the packed model forward
reloaded_forward = tf.saved_model.load(default_path.as_posix() + "/translator_forward")

# Load the packed model forward
reloaded_reverse = tf.saved_model.load(default_path.as_posix() + "/translator_reverse")


def load_forward_translation_utils() -> tuple:
    """Loads essential utilities for forward translation, including input and
    target tokenizers and the maximum input length.

    This function loads pre-trained tokenizers for input and target languages from pickle files and sets the maximum length
    for input sequences. The pickle files are assumed to be located in the 'assets' directory relative to the default path.

    Returns:
        tuple: A tuple containing:
            - inp_lang (Tokenizer): The input language tokenizer.
            - targ_lang (Tokenizer): The target language tokenizer.
            - inp_max_length (int): The maximum length for input sequences.

    Raises:
        FileNotFoundError: If the tokenizer pickle files are not found in the specified directory.
        pickle.UnpicklingError: If there is an error while unpickling the tokenizer files.
    """
    # Load important pickle files which consists the tokenizers and the maxlength setting
    inp_lang = pickle.load(
        open(default_path.as_posix() + "/assets/tokenizer_input.pkl", "rb")
    )
    targ_lang = pickle.load(
        open(default_path.as_posix() + "/assets/tokenizer_target.pkl", "rb")
    )
    inp_max_length = 602
    return inp_lang, targ_lang, inp_max_length


def load_reverse_translation_utils() -> tuple:
    """Loads necessary utilities for reverse translation from pickle files.

    This function loads the input and target tokenizers as well as the
    maximum length setting for input sequences. The tokenizers are loaded
    from pickle files located in the 'assets' directory under the default path.

    Returns:
        tuple: A tuple containing the following elements:
            - inp_lang (object): The tokenizer for the input language.
            - targ_lang (object): The tokenizer for the target language.
            - inp_max_length (int): The maximum length of input sequences.

    Raises:
        FileNotFoundError: If any of the pickle files cannot be found.
        pickle.UnpicklingError: If there is an error unpickling the files.
    """
    targ_lang = pickle.load(
        open(default_path.as_posix() + "/assets/tokenizer_input.pkl", "rb")
    )
    inp_lang = pickle.load(
        open(default_path.as_posix() + "/assets/tokenizer_target.pkl", "rb")
    )
    inp_max_length = 602
    return inp_lang, targ_lang, inp_max_length


def translate_forward(smiles: str, add_confidence: bool = False) -> str:
    """
    Takes user input, splits them into words, and generates tokens. Tokens are
    then passed to the model and the model-predicted tokens are retrieved. The
    predicted tokens get detokenized and the final result is returned in a
    string format. If add_confidence is true, a list of tuples is returned,
    where the first element is the token and the second element is the
    confidence value.

    Args:
        smiles (str): User input SMILES in string format.
        add_confidence (bool): If True, the confidence values of the predicted tokens
        are returned as well.

    Returns:
        result (str): The predicted IUPAC names in string format.
        result (List[tuples]): Tokens and confidence values
    """
    # TODO: loading this for every call is inefficient
    # --> move to init of a translator class
    inp_lang, targ_lang, inp_max_length = load_forward_translation_utils()

    if len(smiles) == 0:
        return "Check SMILES string"
    else:
        canonical_smiles = helper.split_smiles(smiles)
        if canonical_smiles:
            decoded = helper.tokenize_input(canonical_smiles, inp_lang, inp_max_length)
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


def translate_reverse(iupacname: str, add_confidence: bool = False) -> str:
    """Takes user input splits them into words and generates tokens. Tokens are
    then passed to the model and the model predicted tokens are retrieved. The
    predicted tokens get detokenized and the final result is returned in a
    string format. If add_confidence is true, a list of tuples is returned,
    where the first element is the token and the second element is the
    confidence value.

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
    if len(iupacname) == 0:
        return "Check IUPAC name string"
    else:
        splitted_name = helper.split_iupac(iupacname)

    decoded = helper.tokenize_input(splitted_name, inp_lang, inp_max_length)
    result_predited, confidence_array = reloaded_reverse(decoded)
    if add_confidence:
        result = helper.detokenize_output_add_confidence(
            result_predited, confidence_array, targ_lang
        )
    else:
        result = helper.detokenize_output(result_predited, targ_lang)
    return result
