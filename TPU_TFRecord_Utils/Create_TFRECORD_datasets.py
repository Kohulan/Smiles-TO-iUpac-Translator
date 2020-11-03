import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import unicodedata
import re
import numpy as np
import os
import io
import time
import sys
import pickle
from datetime import datetime

def main():
  input_tensor_train = pickle.load(open("input_tensor_.pkl","rb"))
  target_tensor_train = pickle.load(open("target_tensor_.pkl","rb"))

  print ("Total number of selected SMILES Strings: ",len(target_tensor_train), "\n")
  num_shards = 400 #corresponds to total train files
  
  file_index = 0

  get_train_tfrecord(num_shards,input_tensor_train,target_tensor_train,file_index)

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_train_tfrecord(num_shards,input_tensor_train,target_tensor_train,file_index):

  print("Total number of TFrecords: ",num_shards,flush=True)

  for i in range(num_shards):
    subsets_num = int(len(target_tensor_train) / num_shards)
    sub_split_input = input_tensor_train[i * subsets_num: (i + 1) * subsets_num]
    sub_split_target = target_tensor_train[i * subsets_num: (i + 1) * subsets_num]

    tfrecord_name = 'tfrecords_iupac_30mil/'+ 'train-%02d.tfrecord' % file_index
    writer = tf.io.TFRecordWriter(tfrecord_name)
    counter = 0
    for j in (range(len(sub_split_input))):


      input_ = sub_split_input[j]
      target_ = sub_split_target[j]
      #image_id_ = sub_split_img_id[counter]
      counter = counter+1
      feature = {
        #'image_id': _bytes_feature(image_id_.encode('utf8')),
        'input_selfies': _bytes_feature(input_.tostring()),
        'target_iupac': _bytes_feature(target_.tostring()),
        }
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      
      serialized = example.SerializeToString()
      writer.write(serialized)
    print('%s write to tfrecord success!' % tfrecord_name)
    file_index = file_index + 1

if __name__ == '__main__':
  main()