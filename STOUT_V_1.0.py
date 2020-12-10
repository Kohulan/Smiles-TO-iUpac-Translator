import io
import sys
import os
import urllib
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

os.environ["CUDA_VISIBLE_DEVICES"]="2"

def main():
  
  global max_length_targ,max_length_inp,inp_lang,targ_lang,embedding_dim,units,encoder,decoder
  global model_size
  model_size='30'

  if len(sys.argv) < 3 and sys.argv[1] == "--help" or sys.argv[1] == "--h":

    print("\n Usage for 1 SMILES string:\n python STOUT_V_0.5.py --smiles input_SMILES\n\n",
      "For multiple Smiles:\n python STOUT_V_0.5.py --STI input_file outputfile\n\n",
      "To check the translation accuracy you can re-translate the IUPAC names back to SMILES string using OPSIN.\n",
      "Use this command for retranslation:\n python STOUT_V_0.5.py --STI_check input_file outputfile 30\n\n",
      "The system set to default to choose the model trainined on 30 Mio data, to choose the other model available,\n",
      "at the end of each command add 30 or 60 afer a space:\n",
      "e.g.: python STOUT_V_0.5.py --smiles input_SMILES 60\n\n")
    sys.exit()

  elif (len(sys.argv) == 3 or len(sys.argv) == 4) and sys.argv[1] == '--smiles':
    smiles_string = sys.argv[2]
    if len(sys.argv) == 4 and (sys.argv[3] == '30' or sys.argv[3] == '60'):
      model_size = sys.argv[3]

    max_length_targ,max_length_inp,inp_lang,targ_lang,embedding_dim,units,encoder,decoder =check_model(model_size)

    canonical_smiles = subprocess.check_output(['java', '-cp', 'Java_dependencies/cdk-2.1.1.jar:.' ,'SMILEStoCanonicalSMILES',smiles_string])

    iupac_name = translate(selfies.encoder(canonical_smiles.decode('utf-8').strip()).replace("][","] ["))

    print('Predicted translation: {}'.format(iupac_name.replace(" ","").replace("<end>","")),flush=True)

  elif (len(sys.argv) == 4 or len(sys.argv) == 5) and sys.argv[1] == '--STI':
    if len(sys.argv) == 5 and (sys.argv[4] == '30' or sys.argv[4] == '60'):
      model_size = sys.argv[4]
    
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    max_length_targ,max_length_inp,inp_lang,targ_lang,embedding_dim,units,encoder,decoder =check_model(model_size)

    out = batch_mode(input_file,output_file)

    print("Batch mode completed, result saved in: ",out)

  elif (len(sys.argv) == 4 or len(sys.argv) == 5) and (sys.argv[1] == '--STI_check'):
    if len(sys.argv) == 5 and (sys.argv[4] == '30' or sys.argv[4] == '60'):
      model_size = sys.argv[4]

    input_file = sys.argv[2]
    output_file = sys.argv[3]

    max_length_targ,max_length_inp,inp_lang,targ_lang,embedding_dim,units,encoder,decoder =check_model(model_size)

    check_translation(input_file,output_file)

  else:
    print(len(sys.argv))
    print("See help using python3 STOUT_V_0.5.py --help")

   

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

def download_trained_weights(model_url,model_path,model_size, verbose=1):
  #Download trained models
  if verbose > 0:
    print("Downloading trained model to " + model_path + " ...")
    subprocess.run(['wget',model_url])
  if verbose > 0:
    print("... done downloading trained model!")
    subprocess.run(["unzip", "Trained_models.zip"])

def check_model(model_size):
  #load lengths
  max_length_targ = pickle.load(open("important_assets/"+model_size+"_mil/forward/max_length_targ.pkl","rb"))
  max_length_inp = pickle.load(open("important_assets/"+model_size+"_mil/forward/max_length_inp.pkl","rb"))

  # restoring the latest checkpoint in checkpoint_dir
  checkpoint_path = 'Trained_models/'+model_size
  model_url = 'https://storage.googleapis.com/iupac_models_trained/Trained_model/Trained_models.zip'
  if not os.path.exists(checkpoint_path):
    download_trained_weights(model_url,checkpoint_path,model_size)

  #load model
  inp_lang = pickle.load(open("important_assets/"+model_size+"_mil/forward/tokenizer_input.pkl","rb"))
  targ_lang = pickle.load(open("important_assets/"+model_size+"_mil/forward/tokenizer_target.pkl","rb"))

  vocab_inp_size = len(inp_lang.word_index)+1
  vocab_tar_size = len(targ_lang.word_index)+1

  #return max_length_targ,max_length_inp,vocab_inp_size,vocab_tar_size

  embedding_dim = 256
  units = 1024

  encoder = nmt_model.Encoder(vocab_inp_size, embedding_dim, units)
  decoder = nmt_model.Decoder(vocab_tar_size, embedding_dim, units)

  optimizer = tf.keras.optimizers.Adam()

  ckpt = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)
  ckpt.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

  return max_length_targ,max_length_inp,inp_lang,targ_lang,embedding_dim,units,encoder,decoder

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
  print("Retranslated SMILES are saved in Retranslated_smiles file")
  subprocess.run(['java', '-jar', 'Java_dependencies/opsin-2.5.0-jar-with-dependencies.jar' ,'-osmi',out_file,'Re-translated_smiles'])

if __name__ == '__main__':
  main()
