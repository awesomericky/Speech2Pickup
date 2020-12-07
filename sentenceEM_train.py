import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
import random
from os import listdir
from sentenceEM import sentenceEM
from utils import read_script_files, read_script_file_data
from processed_data_loader import load_single_npz_data
from process_data import make_word_dictionary
import wandb

#####################
## Model configure ##
#####################

train_model_number = 'SOTA'
batch_size = 36; seed = 1; lr = 0.0005; epochs = 500
n_mels = 40
time_steps = 303
word_dic_size = 43
dropout_rate = 0.2
training_state = True
input_shapes = (None, n_mels, time_steps)
encoder_args = {'num_stacks': 3, 'num_channels':[n_mels for i in range(6)], 'kernel_size':3,
                'dropout_rate': dropout_rate, 'activation': 'leaky-relu', 'return_type': 'end'}
linguistic_decoder_args = {'decoder_type': 'linguistic', 'num_channels': word_dic_size, 'kernel_size': [2, 2],
                           'padding': 'same', 'dropout_rate': dropout_rate, 'activation': 'leaky-relu'}

with tf.device('/device:GPU:0'):
    sen_em_model = sentenceEM(encoder_args=encoder_args, linguistic_decoder_args=linguistic_decoder_args,
                              input_shapes=input_shapes, seed=seed, training_state=training_state)
    sen_em_model.build_seperate_graph()
    sen_em_model.build_total_graph()
    sen_em_model.model_compile(lr=lr)
sen_em_model.model_visualize()


#################
## Model train ##
#################

wandb_config = {'batch_size':batch_size,
                'learning_rate': lr,
                'channel_type': 'single',
                'n_mels': n_mels,
                'seed': seed,
                'date': '2020-12-06',
                'train_model_number': train_model_number,
                'Encoder activation': encoder_args['activation'],
                'Linguistic decoder activation': linguistic_decoder_args['activation'],
                'Decoder kernel':  linguistic_decoder_args['kernel_size'],
                'memo': 'init std=0.1, Dropout_rate: 0.2',
                'model type': 'modified model(only linguistic)'}
wandb_run = wandb.init(project='Speech2Pickup', name='sentenceEM', config=wandb_config)
total_model_file_path = '/content/drive/MyDrive/Speech2Pickup/sentenceEM_model/' + str(train_model_number) + '/total_model/model.ckpt'
encoder_model_file_path = '/content/drive/MyDrive/Speech2Pickup/sentenceEM_model/' + str(train_model_number) + '/encoder_model/model.ckpt'
model_configuration_file = '/content/drive/MyDrive/Speech2Pickup/sentenceEM_model/' + str(train_model_number) + '/model_config.txt'
relative_data_directory_path = '/content/drive/MyDrive/Speech2Pickup/data_v2.2_single_channel'

# Write model configuration in .txt file
with open(model_configuration_file, 'w') as f:
    f.write(str(wandb_config))

# Load data
print('Loading data...')
data_file = listdir(relative_data_directory_path)[0]
data = load_single_npz_data(relative_data_directory_path=relative_data_directory_path, file_name=data_file)
acoustic_data = data['acoustic']
linguistic_data = data['linguistic']

# Train model and save
print('Start training')
try:
    # Train
    with tf.device('/device:GPU:0'):
        sen_em_model.model_train(X_train=acoustic_data, Y_linguistic_train=linguistic_data, batch_size=batch_size, epochs=epochs,
                                 total_model_file_path=total_model_file_path, encoder_model_file_path=encoder_model_file_path)
except KeyboardInterrupt:
    pass



#############################
## Erase model from memory ##
### Finish wandb process ####
#############################

tf.keras.backend.clear_session()
del sen_em_model
wandb_run.finish()



################
## Load model ##
################
# model compiling should be done first

model_path = '/content/drive/MyDrive/Speech2Pickup/sentenceEM_model/' + str(train_model_number) + '/total_model/sentenceEM_total_model'
sen_em_model.model.load_weights(model_path)



###############################
## Check loaded model output ##
###############################

import matplotlib.pyplot as plt
import librosa.display

# Set word dictionary
print('Loading word dictionary..')
relative_script_directory_path = './drive/MyDrive/Speech2Pickup/train_script'
word_dic, word_dic_size = make_word_dictionary(relative_script_directory_path)

# Load data
print('Loading data..')
relative_data_directory_path = '/content/drive/MyDrive/Speech2Pickup/data_v2.2_single_channel'
file_name = 'senEM_preprocessed.npz'
n_data = 30988  # Select n_data between 0~40697
data = load_single_npz_data(relative_data_directory_path=relative_data_directory_path, file_name=file_name)
acoustic_train_batch = [data['acoustic'][n_data]]
linguistic_train_batch = [data['linguistic'][n_data]]
acoustic_train_batch = np.array(acoustic_train_batch, dtype=np.float32)
linguistic_train_batch = np.array(linguistic_train_batch, dtype=np.float32)

# Get model output
print('='*20)
print('Model output')
with tf.device('/device:GPU:0'):
    l_out_full = sen_em_model.model(acoustic_train_batch)

l_out_full = np.squeeze(kb.eval(l_out_full))
l_true = np.squeeze(linguistic_train_batch)
l_out = np.argmax(l_out_full, axis=0)
l_true = np.argmax(l_true, axis=0)

# Get the ground truth and predicted sentence
true_sentence = []
predicted_sentence = []
word_dic_keys = list(word_dic.keys())
n = 0
for i in l_true:
  if n==0:
    true_sentence.append(word_dic_keys[i])
  elif true_sentence[-1] != word_dic_keys[i]:
    true_sentence.append(word_dic_keys[i])
  n += 1
n = 0
for i in l_out:
  if n==0:
    predicted_sentence.append(word_dic_keys[i])
  elif predicted_sentence[-1] != word_dic_keys[i]:
    predicted_sentence.append(word_dic_keys[i])
  n += 1
true_sentence = ' '.join(true_sentence)
predicted_sentence = ' '.join(predicted_sentence)
print('True: {}'.format(true_sentence))
print('Predict: {}'.format(predicted_sentence))

# Plot the ground truth and predicted sentence (one-hot encoded)
fig, ax = plt.subplots(1)
ax.plot(l_out, label='prediction')
ax.plot(l_true, label='ground truth')
ax.set_title('Linguistic feature')
ax.set_xlabel('Time step')
ax.set_ylabel('Freq')
plt.legend()
plt.show()

# Plot the ground truth and model output
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
ax[0].imshow(data['linguistic'][n_data])
ax[1].imshow(l_out_full)

# Get encoder model output
print('='*20)
print('Encoder model output')
with tf.device('/device:GPU:0'):
    l_out_embedding = sen_em_model.encoder_model(acoustic_train_batch)

l_out_embedding = np.squeeze(kb.eval(l_out_embedding))
print(l_out_embedding)



# # Cf) 

# # 1) Debugging with real data

# data = load_single_npz_data(relative_data_directory_path='/content/drive/MyDrive/Speech2Pickup/data_v2.2', file_name='senEM_preprocessed_100.npz')
# data1 = data['arr_0']
# data1 = np.float32(data1)
# data1 = kb.expand_dims(data1, axis=0)
# with tf.device('/device:GPU:0'):
#   l_y, a_y = sen_em_model(data1)
# print(kb.eval(a_y)[0,:,180])
# print('='*20)
# print(kb.eval(l_y)[0,:,180])
# print(len(kb.eval(l_y)[0,:,180]))
# print('='*20)
# print(np.sum(kb.eval(l_y)[0,:,180]))
# print(np.sum(kb.eval(l_y)[0,:,300]))

# # 2) Debugging with toy data

# x = tf.convert_to_tensor(np.random.random((batch_size, 80, 303)), dtype=np.float32)   # batch, dim, seq_len
# l_y, a_y = sen_em_model(x)
# print('='*20)
# print(kb.eval(l_y)[0,:,0])
# print(len(kb.eval(l_y)[0,:,0]))
# print('='*20)
# print(np.sum(kb.eval(l_y)[0,:,0]))
# print(np.sum(kb.eval(l_y)[0,:,100]))