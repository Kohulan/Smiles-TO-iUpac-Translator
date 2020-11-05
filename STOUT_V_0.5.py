#import contextlib
import io
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
''' 
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout
'''
import tensorflow as tf

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
  
  if len(sys.argv) <= 2 and sys.argv[1] == "-help" or sys.argv[1] == "-h":

    print("Usage for 1 SMILES string: STOUT_V_0.5.py input_SMILES\n",
      "For multiple Smiles: STOUT_V_0.5.py input_file outputfile\n",
      "To check the translation accuracy you can re translate the IUPAC names back to SMILES string using OPSIN.\n",
      "To do that use: STOUT_V_0.5.py input_file outputfile -check\n")
    sys.exit()

  if len(sys.argv) == 2:

    smiles_string = sys.argv[1]

    canonical_smiles = subprocess.check_output(['java', '-cp', 'Java_dependencies/cdk-2.1.1.jar:.' ,'SMILEStoCanonicalSMILES',smiles_string])

    iupac_name = translate(selfies.encoder(canonical_smiles.decode('utf-8').strip()).replace("][","] ["))

    print('Predicted translation: {}'.format(iupac_name.replace(" ","").replace("<end>","")),flush=True)

  if len(sys.argv) == 3:

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    out = batch_mode(input_file,output_file)

    print("Batch mode completed, result saved in: ",out)

  if len(sys.argv) == 4 and sys.argv[3] == "-check":

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    check_translation(input_file,output_file)

  else:
    print("See help using python3 STOUT_V_0.5.py -help")


   

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
  return result

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

def batch_mode(input_file,output_file):

  outfile = open(output_file,"w")

  with open(input_file,"r") as f:
    for i,line in enumerate(f):
      smiles_string = line.strip()
      canonical_smiles = subprocess.check_output(['java', '-cp', 'Java_dependencies/cdk-2.1.1.jar:.' ,'SMILEStoCanonicalSMILES',smiles_string])
      iupac_name = translate(selfies.encoder(canonical_smiles.decode('utf-8').strip()).replace("][","] ["))
      outfile.write(iupac_name.replace(" ","").replace("<end>","")+"\n")

  outfile.close()

  return output_file

def check_translation(input_file,output_file):

  out_file = batch_mode(input_file,output_file)

  subprocess.check_output(['java', '-jar', 'Java_dependencies/opsin-2.5.0-jar-with-dependencies.jar' ,'-osmi',out_file,'Re-translated_smiles','&>','Re-translation_error.txt'])

  print("Retranslated SMILES are saved in Retranslated_smiles file, the error_log can be found on Re-translation_error.txt")

if __name__ == '__main__':
  main()
