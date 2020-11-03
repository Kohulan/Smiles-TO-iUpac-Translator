import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import sys
import pickle
from datetime import datetime
import nmt_model
import re

tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='node-1')
print('Running on TPU ', tpu.master())

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


f = open('Training_Report.txt' , 'w')
sys.stdout = f

numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

inp_lang = pickle.load(open("tokenizer_input.pkl","rb"))
targ_lang = pickle.load(open("tokenizer_target.pkl","rb"))


print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Data loaded", flush=True)
total_data = 30720000
BUFFER_SIZE = 10000
BATCH_SIZE = 128 * strategy.num_replicas_in_sync
steps_per_epoch = total_data//BATCH_SIZE
num_steps = total_data // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

AUTO = tf.data.experimental.AUTOTUNE

def read_tfrecord(example):
	feature = {
		#'image_id': tf.io.FixedLenFeature([], tf.string),
		'input_selfies': tf.io.FixedLenFeature([], tf.string),
		'target_iupac': tf.io.FixedLenFeature([], tf.string),
			}

	# decode the TFRecord
	example = tf.io.parse_single_example(example, feature)

	input_selfies = tf.io.decode_raw(example['input_selfies'], tf.int32)
	target_iupac = tf.io.decode_raw(example['target_iupac'], tf.int32)

	return input_selfies,target_iupac


def get_training_dataset(batch_size = BATCH_SIZE,buffered_size = BUFFER_SIZE):

	options = tf.data.Options()
	filenames = sorted(tf.io.gfile.glob( "gs://tpu-test-koh/tfrecords_iupac_30mil/*.tfrecord"), key=numericalSort)
	

	dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
	#dataset_txt = tf.data.Dataset.from_tensor_slices((cap_train))

	train_dataset = (
		dataset
		.with_options(options)
		.map(read_tfrecord, num_parallel_calls=AUTO)
		.shuffle(buffered_size).batch(batch_size, drop_remainder=True)
		.prefetch(buffer_size=AUTO)
	)	
	return train_dataset

with strategy.scope():
	encoder = nmt_model.Encoder(vocab_inp_size, embedding_dim, units)
	decoder = nmt_model.Decoder(vocab_tar_size, embedding_dim, units)

	optimizer = tf.keras.optimizers.Adam()
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
		from_logits=True, reduction='none')

	def loss_function(real, pred):
		mask = tf.math.logical_not(tf.math.equal(real, 0))
		loss_ = loss_object(real, pred)
		
		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask

		return tf.reduce_mean(loss_)

checkpoint_path = 'gs://tpu-test-koh/iupac/training_checkpoints'
ckpt = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
	ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
	start_epoch = int(checkpoint_prefix.latest_checkpoint.split('-')[-1])

per_replica_batch_size = BATCH_SIZE // strategy.num_replicas_in_sync

print("Batch Size",BATCH_SIZE)
print("Per replica",per_replica_batch_size)
train_dataset = strategy.experimental_distribute_dataset(get_training_dataset())

#the loss_plot array will be reset many times
loss_plot = []


@tf.function
def train_step(iterator):
	def step_fn(inputs):
		inp, targ = inputs
		print(inp.shape,targ.shape)
		loss = 0

		hidden = encoder.initialize_hidden_state(batch_size=targ.shape[0])
		print(hidden)



		with tf.GradientTape() as tape:
			enc_output, enc_hidden = encoder(inp, hidden)

			dec_hidden = enc_hidden

			dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * targ.shape[0], 1)

			# Teacher forcing - feeding the target as the next input
			for t in range(1, targ.shape[1]):
				# passing enc_output to the decoder
				predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

				loss += loss_function(targ[:, t], predictions)

				# using teacher forcing
				dec_input = tf.expand_dims(targ[:, t], 1)

		batch_loss = (loss / int(targ.shape[1]))

		variables = encoder.trainable_variables + decoder.trainable_variables

		gradients = tape.gradient(loss, variables)

		optimizer.apply_gradients(zip(gradients, variables))

		return loss, batch_loss

	per_replica_losses, l_loss = strategy.run(step_fn, args=(iterator,))
	return strategy.reduce(tf.distribute.ReduceOp.MEAN, 
		per_replica_losses,axis=None),strategy.reduce(tf.
		distribute.ReduceOp.MEAN, l_loss,axis=None)


EPOCHS = 25

print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Training Started", flush=True)

for epoch in range(start_epoch,EPOCHS):
	start = time.time()

	total_loss = 0

	for (batch, (inp, targ)) in enumerate(train_dataset):
		inputs = (inp, targ)
		batch_loss, t_loss = train_step(inputs)
		total_loss += t_loss

		if batch % 100 == 0:
			print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
				'Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,batch,
					batch_loss.numpy()), flush=True)

		loss_plot.append(total_loss / num_steps)

	if epoch % 1 == 0:
		ckpt_manager.save()

		print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),'Epoch {} Loss {:.4f}'.format(epoch + 1,
			total_loss / steps_per_epoch), flush=True)
		print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),'Time taken for 1 epoch {} sec\n'
			.format(time.time() - start), flush=True)

plt.plot(loss_plot , '-o', label= "Loss value")
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.show()
plt.gcf().set_size_inches(20, 20)
plt.savefig("Lossplot_900k1v3-8_128_tpu_test.jpg")
plt.close()

print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'), "Network Completed", flush=True)
f.close()