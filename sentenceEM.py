import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as kb
from tensorflow.keras.callbacks import Callback
# from plot_model import plot_model
import random
from TCN_and_decoder import TempConvnet
from TCN_and_decoder import TempConvnet_Decoder
from processed_data_loader import load_single_npz_data
from utils import read_script_files
import pdb
import wandb

class sentenceEM(tf.keras.Model):
    def __init__(self, encoder_args, linguistic_decoder_args, input_shapes, seed, training_state):
        """
        1) encoder_args: 
        num_stacks, num_channels, kernel_size, dropout_rate, activation, return_type, seed
        2) linguistic_decoder_args:
        decoder_type, num_channels, kernel_size, padding, dropout_rate, activation, seed, training_state
        """
        super(sentenceEM, self).__init__()
        self.input_shapes = input_shapes
        self.input_reshape_block = layers.Permute((2, 1))
        self.encoder = TempConvnet(num_stacks=encoder_args['num_stacks'], num_channels=encoder_args['num_channels'],
                                   kernel_size=encoder_args['kernel_size'], dropout_rate=encoder_args['dropout_rate'], activation=encoder_args['activation'],
                                   return_type=encoder_args['return_type'], seed=seed, training_state=training_state)
        self.linguistic_decoder = TempConvnet_Decoder(decoder_type=linguistic_decoder_args['decoder_type'], num_channels=linguistic_decoder_args['num_channels'], 
                                                      kernel_size=linguistic_decoder_args['kernel_size'], padding=linguistic_decoder_args['padding'],
                                                      dropout_rate=linguistic_decoder_args['dropout_rate'], activation=linguistic_decoder_args['activation'],
                                                      seed=seed, training_state=training_state)
    
    def call(self, x):
        embedded_outputs = self.input_reshape_block(x)
        embedded_outputs = self.encoder(embedded_outputs)
        embedded_outputs = kb.expand_dims(embedded_outputs, axis=1)
        linguistic_outputs = self.linguistic_decoder(embedded_outputs)
        return linguistic_outputs
    
    def encoder_call(self, x):
        embedded_outputs = self.input_reshape_block(x)
        embedded_outputs = self.encoder(embedded_outputs)
        return embedded_outputs
    
    def decoder_call(self, x):
        embedded_outputs = kb.expand_dims(x, axis=1)
        linguistic_outputs = self.linguistic_decoder(embedded_outputs)
        return linguistic_outputs
    
    def build_seperate_graph(self):
        # Encoder graph
        x = tf.keras.Input(batch_shape=self.input_shapes)
        embedded_outputs = self.encoder_call(x)
        self.embedded_outputs_shape = embedded_outputs.shape
        self.encoder_model = tf.keras.Model(inputs=x, outputs=embedded_outputs)

        # Decoder graph
        embeddings = tf.keras.Input(batch_shape=self.embedded_outputs_shape)
        linguistic_outputs = self.decoder_call(embeddings)
        self.decoder_model = tf.keras.Model(inputs=embeddings, outputs=[linguistic_outputs])
    
    def build_total_graph(self):
        x = tf.keras.Input(batch_input_shape=self.input_shapes)
        y = self.encoder_model(x)
        l_y = self.decoder_model(y)
        self.model = tf.keras.Model(inputs=x, outputs=[l_y])

    def model_visualize(self):
        self.model.summary()
        # plot_model(self.model, to_file='/home/awesomericky/Lab_intern/Prof_Oh/Code/Speech2Pickup/image/sentenceEM.png')
    
    def encoder_model_compile(self, lr):
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.encoder_model.compile(optimizer=optimizer)

    def model_compile(self, lr):
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        metric = self.custom_metric()
        self.model.compile(optimizer=optimizer, loss=[self.linguistic_loss_function], metrics=[metric])

    def model_train(self, X_train, Y_linguistic_train, batch_size, epochs, total_model_file_path, encoder_model_file_path):
        sess = kb.get_session()
        total_saver = tf.train.Saver(var_list=tf.trainable_variables())
        encoder_saver = tf.train.Saver(var_list=[v for v in tf.trainable_variables() if v.name.split('/')[0] == 'temp_convnet'])
        custom_callback = CustomCallback(total_model_file_path, encoder_model_file_path, sess, total_saver, encoder_saver)
        self.model.fit(x=X_train, y=[Y_linguistic_train], batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[custom_callback])

    def custom_metric(self):
        def categorical_accuracy(y_true, y_pred):
            ind_actual = tf.math.argmax(y_true, axis=1)
            ind_predict = tf.math.argmax(y_pred, axis=1)
            batch_size = ind_predict.shape[0]
            time_size = ind_predict.shape[1]
            count = tf.reduce_sum(tf.cast(tf.math.equal(ind_actual, ind_predict), tf.int32), axis=1)
            accuracy = tf.math.reduce_mean(tf.math.divide(count, time_size))
            return accuracy
        categorical_accuracy.__name__ = 'categorical_accuracy'
        return categorical_accuracy

    def linguistic_loss_function(self, y_actual, y_predicted):
        epsil = 1e-5
        y_predicted_correct = -kb.log(y_predicted + epsil)
        # y_predicted_incorrect = -kb.log(1 - y_predicted + epsil)
        # y_actual_op = tf.math.subtract(tf.ones_like(y_actual), y_actual)

        correct_loss = layers.multiply([y_actual, y_predicted_correct])
        # incorrect_loss = layers.multiply([y_actual_op, y_predicted_incorrect])
        # loss = tf.math.add(correct_loss, incorrect_loss)
        loss = kb.sum(correct_loss, axis=1)
        loss = kb.mean(loss, keepdims=False)
        return loss

class CustomCallback(Callback):
    def __init__(self, total_model_file_path, encoder_model_file_path, sess, total_saver, encoder_saver):
        self.total_model_file_path = total_model_file_path
        self.encoder_model_file_path = encoder_model_file_path
        self.sess = sess
        self.total_saver = total_saver
        self.encoder_saver = encoder_saver
        self.monitor = 'categorical_accuracy'
        self.monitor_op = np.greater
        self.best = 0
        self.temp_loss = []
        self.temp_linguistic_accuracy = []
    
    def on_batch_end(self, batch, logs=None):
        self.temp_loss.append(logs.get('loss'))
        self.temp_linguistic_accuracy.append(logs.get('categorical_accuracy'))

    def on_epoch_end(self, epoch, logs=None):
        self.temp_loss.append(logs.get('loss'))
        self.temp_linguistic_accuracy.append(logs.get('categorical_accuracy'))

        # Logging data
        for iii in range(len(self.temp_loss)):
            wandb.log({'temp_loss': self.temp_loss[iii]})
            wandb.log({'temp_linguistic_accuracy': self.temp_linguistic_accuracy[iii]})
        self.temp_loss = []
        self.temp_linguistic_accuracy = []
        
        # Save model
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            self.best = current
            self.total_saver.save(self.sess, self.total_model_file_path)
            self.encoder_saver.save(self.sess, self.encoder_model_file_path)
            

class sentenceEM_encoder(tf.keras.Model):
    def __init__(self, encoder_args, input_shapes, seed, training_state):
        """
        encoder_args: 
        num_stacks, num_channels, kernel_size, dropout_rate, activation, return_type, seed
        """
        super(sentenceEM_encoder, self).__init__()
        self.input_shapes = input_shapes
        self.input_reshape_block = layers.Permute((2, 1))
        self.encoder = TempConvnet(num_stacks=encoder_args['num_stacks'], num_channels=encoder_args['num_channels'],
                                   kernel_size=encoder_args['kernel_size'], dropout_rate=encoder_args['dropout_rate'], activation=encoder_args['activation'],
                                   return_type=encoder_args['return_type'], seed=seed, training_state=training_state)

    def call(self, x):
        embedded_outputs = self.input_reshape_block(x)
        embedded_outputs = self.encoder(embedded_outputs)
        return embedded_outputs
    
    def build_graph(self):
        x = tf.keras.Input(batch_shape=self.input_shapes)
        embedded_outputs = self.call(x)
        self.encoder_model = tf.keras.Model(inputs=x, outputs=embedded_outputs)