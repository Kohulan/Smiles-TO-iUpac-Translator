import contextlib
import io
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import pickle
from datetime import datetime
from Network import nmt_model
import numpy as np
import unicodedata
import re
import selfies
import subprocess
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main():
  if len(sys.argv) != 2:
    print("\"Usage of this function: evaluate_iupac.py input_SMILES")
  if len(sys.argv) == 2:
    smiles_string = sys.argv[1]

    canonical_smiles = subprocess.check_output(['java', '-cp', 'Java_dependencies/cdk-2.1.1.jar:.' ,'SMILEStoCanonicalSMILES',smiles_string])

    selfies_input = selfies.encoder(canonical_smiles.decode('utf-8').strip())

    translate(selfies_input.replace("][","] ["))
   

#load lengths
max_length_targ = pickle.load(open("important_assets/max_length_targ.pkl","rb"))
max_length_inp = pickle.load(open("important_assets/max_length_inp.pkl","rb"))

# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
  w = unicode_to_ascii(w.strip())
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = w.strip()
  w = '<start> ' + w + ' <end>'
  return w

def evaluate(sentence):
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  sentence = preprocess_sentence(sentence)

  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)


    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence

def translate(sentence):
  result, sentence = evaluate(sentence)

  #print('Input: %s' % (selfies.decoder(sentence.replace(" ","").replace("<start>","").replace("<end>",""))),flush=True)
  print('Predicted translation: {}'.format(result.replace(" ","").replace("<end>","")),flush=True)

#load model

inp_lang = pickle.load(open("important_assets/tokenizer_input.pkl","rb"))
targ_lang = pickle.load(open("important_assets/tokenizer_target.pkl","rb"))

embedding_dim = 256
units = 1024

vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

encoder = nmt_model.Encoder(vocab_inp_size, embedding_dim, units)
decoder = nmt_model.Decoder(vocab_tar_size, embedding_dim, units)

optimizer = tf.keras.optimizers.Adam()

# restoring the latest checkpoint in checkpoint_dir
checkpoint_path = 'Trained_model/'
ckpt = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)

ckpt.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

if __name__ == '__main__':
  main()
