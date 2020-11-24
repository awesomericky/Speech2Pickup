import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as kb
from tensorflow.keras.callbacks import Callback
from plot_model import plot_model
import random
from TCN_and_decoder import TempConvnet
from TCN_and_decoder import TempConvnet_Decoder
from processed_data_loader import load_single_npz_data
from utils import read_script_files
import pdb
import wandb

class sentenceEM(tf.keras.Model):
    def __init__(self, encoder_args, linguistic_decoder_args, acoustic_decoder_args, input_shapes, seed, training_state):
        """
        1) encoder_args: 
        num_stacks, num_channels, kernel_size, dropout_rate, return_type, seed
        2) linguistic_decoder_args:
        decoder_type, num_levels, num_channels, kernel_size, padding, upsample_size, dropout_rate, output_shape, seed, training_state
        3) acoustic_decoder_args:
        decoder_type, num_levels, num_channels, kernel_size, padding, upsample_size, dropout_rate, output_shape, seed, training_state
        """
        super(sentenceEM, self).__init__()
        self.input_shapes = input_shapes
        self.input_reshape_block = layers.Permute((2, 1))
        self.encoder = TempConvnet(num_stacks=encoder_args['num_stacks'], num_channels=encoder_args['num_channels'],
                                        kernel_size=encoder_args['kernel_size'], dropout_rate=encoder_args['dropout_rate'],
                                        return_type=encoder_args['return_type'], seed=seed, training_state=training_state)
        self.linguistic_decoder = TempConvnet_Decoder(decoder_type=linguistic_decoder_args['decoder_type'], num_levels=linguistic_decoder_args['num_levels'],
                                                        num_channels=linguistic_decoder_args['num_channels'], kernel_size=linguistic_decoder_args['kernel_size'],
                                                        padding=linguistic_decoder_args['padding'], upsample_size=linguistic_decoder_args['upsample_size'],
                                                        dropout_rate=linguistic_decoder_args['dropout_rate'], output_shape=linguistic_decoder_args['output_shape'],
                                                        seed=seed, training_state=training_state)
        self.acoustic_decoder = TempConvnet_Decoder(decoder_type=acoustic_decoder_args['decoder_type'], num_levels=acoustic_decoder_args['num_levels'],
                                                        num_channels=acoustic_decoder_args['num_channels'], kernel_size=acoustic_decoder_args['kernel_size'],
                                                        padding=acoustic_decoder_args['padding'], upsample_size=acoustic_decoder_args['upsample_size'],
                                                        dropout_rate=acoustic_decoder_args['dropout_rate'], output_shape=acoustic_decoder_args['output_shape'],
                                                        seed=seed, training_state=training_state)
    
    def call(self, x):
        embedded_outputs = self.input_reshape_block(x)
        print(embedded_outputs.shape)
        embedded_outputs = self.encoder(embedded_outputs)
        embedded_outputs = kb.expand_dims(embedded_outputs, axis=1)
        print(embedded_outputs.shape)
        linguistic_outputs = self.linguistic_decoder(embedded_outputs)
        acoustic_outputs = self.acoustic_decoder(embedded_outputs)

        return linguistic_outputs, acoustic_outputs

    def build_graph(self):
        x = tf.keras.Input(batch_shape=self.input_shapes)
        linguistic_outputs, acoustic_outputs = self.call(x)

        self.model = tf.keras.Model(inputs=x, outputs=[linguistic_outputs, acoustic_outputs])
    
    def model_visualize(self):
        self.model.summary()
        plot_model(self.model, to_file='/content/drive/MyDrive/Speech2Pickup/sentenceEM.png')
    
    def model_compile(self, lr, loss_weights):
        self.model.compile(optimizer='adam', loss=[self.linguistic_loss_function, self.acoustic_loss_function], loss_weights=[loss_weights['linguistic'], loss_weights['acoustic']])

    def model_train(self, X_train, Y_linguistic_train, Y_acoustic_train, batch_size):
        hist = History()
        self.model.fit(x=X_train, y=[Y_linguistic_train, Y_acoustic_train], batch_size=batch_size, epochs=1, verbose=1, callbacks=[hist])
        return hist.losses
    
    def linguistic_loss_function(self, y_actual, y_predicted):
        epsil = 1e-5
        y_predicted = -kb.log(y_predicted + epsil)
        loss = layers.multiply([y_actual, y_predicted])
        loss = kb.sum(loss, axis=1)
        loss = kb.mean(loss, keepdims=False)
        return loss
    
    def acoustic_loss_function(self, y_actual, y_predicted):
        loss = layers.subtract([y_actual, y_predicted])
        loss = layers.multiply([loss, loss])
        loss = kb.mean(loss, keepdims=False)
        return loss


class History(Callback):
    def on_train_begin(self,logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))