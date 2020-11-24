import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
import random
from sentenceEM import sentenceEM
from utils import data_shuffle
from processed_data_loader import load_single_npz_data
import wandb

#####################
## Model configure ##
#####################

batch_size = 64; seed = 1; lr = 0.001; epochs = 300; loss_weights={'linguistic': 1, 'acoustic': 0.005}
n_mels = 80
time_steps = 303
training_state = True
input_shapes = (batch_size, n_mels, time_steps)
encoder_args = {'num_stacks': 3, 'num_channels':[80, 80, 80, 80, 80, 80], 'kernel_size':3, 'dropout_rate': 0.2, 'return_type': 'end'}
linguistic_decoder_args = {'decoder_type': 'linguistic', 'num_levels':8, 'num_channels': 42, 'kernel_size': [2, 2], 'padding': 'causal',
                            'upsample_size': {2: 5, 3: 2, 0:1}, 'dropout_rate': 0.2, 'output_shape': (batch_size, 42, 303)}
acoustic_decoder_args = {'decoder_type': 'acoustic', 'num_levels':8, 'num_channels': 80, 'kernel_size': [2, 2], 'padding': 'causal',
                            'upsample_size': {2: 5, 3: 2, 0:1}, 'dropout_rate': 0.2, 'output_shape': (batch_size, 80, 303)}

with tf.device('/device:GPU:0'):
  sen_em_model = sentenceEM(encoder_args=encoder_args, linguistic_decoder_args=linguistic_decoder_args, acoustic_decoder_args=acoustic_decoder_args,
                              input_shapes=input_shapes, seed=seed, training_state=training_state)
  # sen_em_model.build(input_shapes)
  sen_em_model.build_graph()
  sen_em_model.model_compile(lr=lr, loss_weights=loss_weights)
# sen_em_model.model_visualize()



#################
## Model train ##
#################

relative_data_directory_path = '/content/drive/MyDrive/Speech2Pickup/data_v2.2'
wandb.init(project='Speech2Pickup', name='sentenceEM_lr:0.001')
model_save_freq = 5
for i in range(epochs):
    try:
        print('='*20)
        print('Epoch: {}'.format(i+1))

        # Prepare training
        one_shot_load_num = 50
        start = 0
        end = start + batch_size*one_shot_load_num
        epoch_loss = []
        extra_train_state = False
        if i == 0:
            data_files = data_shuffle(relative_data_directory_path)
        else:
            random.shuffle(data_files)
          
        if len(data_files) % (batch_size*one_shot_load_num) != 0:
            extra_train_state = True
            print('extra train needed')
        
        # Train
        while end <= len(data_files):
            print('Processing {}/{}'.format(end//(batch_size*one_shot_load_num), len(data_files)//(batch_size*one_shot_load_num))) if extra_train_state == False \
            else print('Processing {}/{}'.format(end//(batch_size*one_shot_load_num), (len(data_files)//(batch_size*one_shot_load_num))+1))
            batch_data_files = data_files[start:end]
            acoustic_train_batch = []
            linguistic_train_batch = []
            for ii in range(batch_size*one_shot_load_num):
                data = load_single_npz_data(relative_data_directory_path=relative_data_directory_path, file_name=batch_data_files[ii])
                acoustic_train_batch.append(data['arr_0'])
                linguistic_train_batch.append(data['arr_1'])
            acoustic_train_batch = np.array(acoustic_train_batch)
            linguistic_train_batch = np.array(linguistic_train_batch)
            with tf.device('/device:GPU:0'):
              temp_loss = sen_em_model.model_train(X_train=acoustic_train_batch, Y_linguistic_train=linguistic_train_batch,
                                                  Y_acoustic_train=acoustic_train_batch, batch_size=batch_size)
              
              for iii in range(len(temp_loss)):
                wandb.log({'temp_loss': temp_loss[iii]})
              epoch_loss.extend(temp_loss)
            start = end
            end = start + batch_size*one_shot_load_num
            
        if extra_train_state:
            print('Processing {}/{}'.format((len(data_files)//(batch_size*one_shot_load_num))+1, (len(data_files)//(batch_size*one_shot_load_num))+1))
            batch_data_files = data_files[start:]
            acoustic_train_batch = []
            linguistic_train_batch = []
            for ii in range(len(batch_data_files)):
                data = load_single_npz_data(relative_data_directory_path=relative_data_directory_path, file_name=batch_data_files[ii])
                acoustic_train_batch.append(data['arr_0'])
                linguistic_train_batch.append(data['arr_1'])
            acoustic_train_batch = np.array(acoustic_train_batch)
            linguistic_train_batch = np.array(linguistic_train_batch)
            with tf.device('/device:GPU:0'):
              temp_loss = sen_em_model.model_train(X_train=acoustic_train_batch, Y_linguistic_train=linguistic_train_batch,
                                                  Y_acoustic_train=acoustic_train_batch, batch_size=batch_size)
              for iii in range(len(temp_loss)):
                wandb.log({'temp_loss': temp_loss[iii]})
              epoch_loss.extend(temp_loss)

        epoch_loss = np.mean(np.array(epoch_loss))
        wandb.log({'epoch_loss': epoch_loss})

        # Model save
        if (i+1) % model_save_freq == 0:
          print('Finished training {} epochs'.format(i+1))
          sen_em_model.model.save_weights(filepath='/content/drive/MyDrive/Speech2Pickup/sentenceEM_model/sentenceEM_model', overwrite=True)
          print('Model saving complete!')
    except KeyboardInterrupt:
        pass



#############################
## Erase model from memory ##
#############################

tf.keras.backend.clear_session()
del sen_em_model



################
## Load model ##
################
# model compiling should be done first

sen_em_model.model.load_weights('/content/drive/MyDrive/Speech2Pickup/sentenceEM_model/sentenceEM_model')



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