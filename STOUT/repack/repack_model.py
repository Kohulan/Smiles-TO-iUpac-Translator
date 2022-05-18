"""This module was written to wrap the inference functions into a class.
And later wrap the class and the checkpoint into a module and save it into a tf.saved_model.
This helps to reduce the clutter and for faster inference functions.
"""
# Initializing and importing necessary libararies

import tensorflow as tf
import os
import pickle
import transformer_model_4_repack as nmt_model_transformer
import helper

print(tf.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
targ_max_length = pickle.load(open("max_length_targ.pkl", "rb"))
input_vocab_size = len(inp_lang.word_index) + 1
target_vocab_size = len(targ_lang.word_index) + 1
print(inp_max_length)

# Load the original Network parameters
num_layer = 4
d_model = 512
dff = 2048
num_heads = 8
dropout_rate = 0.1

# Initialize the Transformer and the optimizer
transformer = nmt_model_transformer.Transformer(
    num_layer,
    d_model,
    num_heads,
    dff,
    input_vocab_size,
    target_vocab_size,
    inp_max_length,
    targ_max_length,
    rate=dropout_rate,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00051)

# Restarore the last known best performing checkpoint of the trained model
checkpoint_path = "100mil_reverse_character"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))


class Translator(tf.Module):
    """This is a class which takes care of inference. It loads the saved checkpoint and the necessary
    tokenizers. The inference begins with the start token (<start>) and ends when the end token(<end>)
    is met. This class can only work with tf.Tensor objects. The strings shoul gets transformeed into np.arrays
    before feeding them into this class.
    """

    def __init__(
        self, targ_max_length, inp_max_length, inp_lang, targ_lang, transformer
    ):
        """Load the tokenizers, the maximum input and output length and the transformer model.

        Args:
            targ_max_length ([type]): Maximum length of a string which can get predicted.
            inp_max_length ([type]): Maximum length of an input string.
            inp_lang ([type]): Input tokenizer, defines which charater is assigned to what token.
            targ_lang ([type]): Output tokenizer, defines which charater is assigned to what token.
            transformer ([type]): The transformer model.
        """
        self.targ_max_length = targ_max_length
        self.inp_max_length = inp_max_length
        self.inp_lang = inp_lang
        self.targ_lang = targ_lang
        self.transformer = transformer

    def __call__(self, sentence: tf.Tensor[tf.int32]) -> tf.Tensor[tf.int64]:
        """This fuction takes in the tokenized input of a SMILES string or an IUPAC name
        and makes the predicted list of tokens and return the tokens as tf.Tensor array
        before feeding the input array we must define start and the end tokens.

        Args:
            sentence (tf.Tensor[tf.int32]): Input array in tf.Easgertensor format.

        Returns:
            tf.Tensor[tf.int64]: predicted output as an array.
        """
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        output_sentence = tf.convert_to_tensor(sentence)
        encoder_input = output_sentence

        start = tf.cast(
            tf.convert_to_tensor([targ_lang.word_index["<start>"]]), tf.int64
        )
        end = tf.cast(tf.convert_to_tensor([targ_lang.word_index["<end>"]]), tf.int64)

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for t in tf.range(targ_max_length):
            output = tf.transpose(output_array.stack())
            enc_padding_mask, combined_mask, dec_padding_mask = helper.create_masks(
                encoder_input, output
            )

            predictions, _ = self.transformer(
                (
                    encoder_input,
                    output,
                    enc_padding_mask,
                    combined_mask,
                    dec_padding_mask,
                ),
                False,
            )

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)

            output_array = output_array.write(t + 1, predicted_id[0])

            if predicted_id == end:
                break
        output = tf.transpose(output_array.stack())

        return output, sentence


# Create an instance of this Translator class
translator = Translator(
    targ_max_length, inp_max_length, inp_lang, targ_lang, transformer
)


class ExportTranslator(tf.Module):
    """This class wraps the inference class into a module into tf.Module sub-class, with a tf.function on the __call__ method.
    So we could export the model as a tf.saved_model.
    """

    def __init__(self, translator):
        """Import the translator instance."""
        self.translator = translator

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, inp_max_length], dtype=tf.int32)]
    )
    def __call__(self, sentence: tf.Tensor[tf.int32]) -> tf.Tensor[tf.int64]:
        """This fucntion calls the __call__function from the translator class.
        In the tf.function only the output sentence is returned.
        Thanks to the non-strict execution in tf.function any unnecessary values are never computed.

        Args:
            sentence (tf.Tensor[tf.int32]): Input array in tf.Easgertensor format.

        Returns:
            tf.Tensor[tf.int64]: predicted output as an array.
        """

        (result, tokens) = self.translator(sentence)

        return result


# Create an instance of the ExportTranslator module
translator = ExportTranslator(translator)

# Save the model into a directory with assets
tf.saved_model.save(translator, export_dir="STOUT_2_reverse")
