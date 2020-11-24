import numpy as np
import tensorflow as tf
from plot_model import plot_model
# from tensorflow.keras.utils import plot_model

class ResidualBlock(tf.keras.Model):
    def __init__(self, dilation_rate, num_filters, kernel_size, padding, 
                        dropout_rate, seed, training_state):
        super(ResidualBlock, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1)
        layers = tf.keras.layers
        assert padding in ['causal', 'same']

        self.training_state = training_state

        # Block1
        self.conv1 = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, data_format='channels_last',
                                    dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
        self.batch1 = layers.BatchNormalization(axis=1, trainable=True)
        self.ac1 = layers.LeakyReLU(alpha=0.2)
        self.drop1 = layers.Dropout(rate=dropout_rate)

        # Block2
        self.conv2 = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, data_format='channels_last',
                                    dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
        self.batch2 = layers.BatchNormalization(axis=1, trainable=True)
        self.ac2 = layers.LeakyReLU(alpha=0.2)
        self.drop2 = layers.Dropout(rate=dropout_rate)

        self.downsample = layers.Conv1D(filters=num_filters, kernel_size=1,
                                        padding='same', kernel_initializer=init)
        self.ac3 = layers.LeakyReLU(alpha=0.2)
    
    def call(self, x):
        # Block1
        prev_x = x
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.ac1(x)
        x = self.drop1(x) if self.training_state else x

        # Block2
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.ac2(x)
        x = self.drop2(x) if self.training_state else x

        # Match dimention
        if prev_x.shape[-1] != x.shape[-1]:
            prev_x = self.downsample(prev_x)
        assert prev_x.shape == x.shape

        # skip connection
        return self.ac3(prev_x + x)

# # Test
# x = tf.convert_to_tensor(np.random.random((100, 80, 303)))   # batch, dim, seq_len
# model = ResidualBlock(dilation_rate=1, num_filters=80, kernel_size=3, padding='causal', dropout_rate=0.2, seed=1, training_state=True)
# y = model(x)
# print(y.shape)
# model.summary()

# # Check model
# x = tf.keras.Input(batch_shape=(10, 80, 303))
# model = ResidualBlock(dilation_rate=1, num_filters=80, kernel_size=3, padding='causal', dropout_rate=0.2, seed=1, training_state=True)
# model.build((10, 80, 303))
# model = tf.keras.Model(inputs=x, outputs=model.call(x))
# plot_model(model, to_file='residual_block.png')


class TemporalBlock(tf.keras.Model):
    def __init__(self, num_channels, kernel_size, dropout_rate, seed, training_state):
        # num_channels is a list contains hidden channel numbers of Conv1D
        # len(num_channels) is number of convolutional layers in one Temporal Block
        super(TemporalBlock, self).__init__()
        assert isinstance(num_channels, list)

        self.num_levels = len(num_channels)
        self.resi_blocks = [0]*self.num_levels
        for i in range(self.num_levels):
            dilation_rate = 2**i
            self.resi_blocks[i] = ResidualBlock(dilation_rate, num_channels[i], kernel_size, padding='causal',
                            dropout_rate=dropout_rate, seed=seed, training_state=training_state)
    
    def call(self, x):
        for i in range(self.num_levels):
            x = self.resi_blocks[i](x)
        return x

# # Test
# x = tf.convert_to_tensor(np.random.random((100, 80, 303)))   # batch, dim, seq_len
# model = TemporalBlock(num_channels=[80, 80, 80], kernel_size=3, dropout_rate=0.2, seed=1, training_state=True)
# y = model(x)
# print(y.shape)
# model.summary()

# # Check model
# x = tf.keras.Input(batch_shape=(10, 80, 303))
# model = TemporalBlock(num_channels=[80, 80, 80, 80, 80, 80], kernel_size=3, dropout_rate=0.2, seed=1, training_state=True)
# model.build((10, 80, 303))
# model = tf.keras.Model(inputs=x, outputs=model.call(x))
# plot_model(model, to_file='temporal_block.png')

class TempConvnet(tf.keras.Model):
    def __init__(self, num_stacks, num_channels, kernel_size, dropout_rate, return_type, seed, training_state):
        # num_stacks number of Temporal Blocks in Temporal convolutional network
        super(TempConvnet, self).__init__()
        assert isinstance(num_stacks, int)
        assert isinstance(num_channels, list)
        assert return_type in ['whole', 'end']

        self.num_stacks = num_stacks
        self.temp_blocks = [0]*self.num_stacks
        self.return_type = return_type
        for i in range(num_stacks):
            self.temp_blocks[i] = TemporalBlock(num_channels, kernel_size=kernel_size, dropout_rate=dropout_rate, seed=seed, training_state=training_state)
    
    def call(self, x):
        for i in range(self.num_stacks):
            x = self.temp_blocks[i](x)
        
        if self.return_type == 'whole':
            return x
        elif self.return_type == 'end':
            return x[:, -1, :]

# # Test
# x = tf.convert_to_tensor(np.random.random((100, 303, 80)))   # batch, dim, seq_len
# model = TempConvnet(num_stacks=3, num_channels=[80, 80, 80], kernel_size=3, dropout_rate=0.2, return_type='end', seed=1, training_state=True)
# y = model(x)
# print(y.shape)
# model.summary()

# # Check model
# x = tf.keras.Input(batch_shape=(10, 80, 303))
# model = TempConvnet(num_stacks=3, num_channels=[80, 80, 80, 80, 80, 80], kernel_size=3, dropout_rate=0.2, return_type='end', seed=1, training_state=True)
# model.build((10, 80, 303))
# model = tf.keras.Model(inputs=x, outputs=model.call(x))
# plot_model(model, to_file='temporal_CNN.png')

class TempConvnet_Decoder(tf.keras.Model):
    def __init__(self, decoder_type, num_levels, num_channels, kernel_size, padding, upsample_size, dropout_rate, output_shape, seed, training_state):
        super(TempConvnet_Decoder, self).__init__()
        assert isinstance(num_channels, int)
        assert isinstance(kernel_size, list)
        assert isinstance(upsample_size, dict)
        assert padding in ['causal', 'same']
        assert decoder_type in ['linguistic', 'acoustic']
        assert num_levels == upsample_size[2] + upsample_size[3] + upsample_size[0]

        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1)
        layers = tf.keras.layers

        self.training_state = training_state

        self.size2_upsample_num = upsample_size[2]
        self.size3_upsample_num = upsample_size[3]
        self.reshape_conv_num = 1
        self.num_levels = num_levels
        self.final_kernel_size = kernel_size[1]
        self.output_len = output_shape[-1]  ## ex) (num_batch, num_time_steps, dictionary_length)

        self.upsample_blocks = [0]*(num_levels-1)
        self.conv_blocks = [0]*(num_levels-1)
        self.batchnorm_blocks = [0]*(num_levels-1)
        self.ac_blocks = [0]*(num_levels-1)
        self.drop_blocks = [0]*(num_levels-1)

        for i in range(num_levels-self.size3_upsample_num-self.reshape_conv_num):
            self.upsample_blocks[i] = layers.UpSampling1D(size=2)
            self.conv_blocks[i] = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                                    padding=padding, kernel_initializer=init)
            self.batchnorm_blocks[i] = layers.BatchNormalization(axis=-1, trainable=True)
            self.ac_blocks[i] = layers.ReLU() if decoder_type == 'linguistic' else layers.LeakyReLU(alpha=0.2)
            self.drop_blocks[i] = layers.Dropout(rate=dropout_rate)
        for i in range(num_levels-self.size3_upsample_num-self.reshape_conv_num, num_levels-self.reshape_conv_num):
            self.upsample_blocks[i] = layers.UpSampling1D(size=3)
            self.conv_blocks[i] = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                                    padding=padding, kernel_initializer=init)
            self.batchnorm_blocks[i] = layers.BatchNormalization(axis=-1, trainable=True)
            self.ac_blocks[i] = layers.Activation('tanh') if decoder_type == 'linguistic' else layers.LeakyReLU(alpha=0.2)
            self.drop_blocks[i] = layers.Dropout(rate=dropout_rate)
        self.final_conv_block = layers.Conv1D(filters=num_channels, kernel_size=self.final_kernel_size, data_format='channels_last',
                                                    padding='valid', kernel_initializer=init)
        self.final_ac_block = layers.Softmax(axis=-1) if decoder_type == 'linguistic' else layers.LeakyReLU(alpha=0.2)
        self.final_reshape_block = layers.Permute((2, 1))
    
    def call(self, x):
        for i in range(self.num_levels-1):
            x = self.upsample_blocks[i](x)
            x = self.conv_blocks[i](x)
            x = self.batchnorm_blocks[i](x)
            x = self.ac_blocks[i](x)
            x = self.drop_blocks[i](x) if self.training_state else x
        len_padding = (self.output_len - (x.shape[1] - self.final_kernel_size) - 1)//2
        x = tf.keras.layers.ZeroPadding1D(padding=(len_padding, len_padding))(x)
        x = self.final_conv_block(x)
        x = self.final_ac_block(x)
        y = self.final_reshape_block(x)
        return y

# # Test
# x = tf.convert_to_tensor(np.random.random((100, 1, 80)))   # batch, seq_len, dim
# model = TempConvnet_Decoder(decoder_type='linguistic', num_levels=8, num_channels=42, kernel_size=[2, 2], 
#                             padding='causal', upsample_size={2: 5, 3: 2, 0:1}, dropout_rate=0.2, output_shape=(100, 42, 303), seed=1, training_state=True)
# y = model(x, True)
# print(y.shape)
# model.summary()

# # Check model
# # linguistic decoder
# x = tf.keras.Input(batch_shape=(10, 1, 80)) # batch_size: 10
# model = TempConvnet_Decoder(decoder_type='linguistic', num_levels=8, num_channels=42, kernel_size=[2, 2], 
#                             padding='causal', upsample_size={2: 5, 3: 2, 0:1}, dropout_rate=0.2, output_shape=(None, 42, 303), seed=1, training_state=True)
# model.build((10, 1, 80))
# model = tf.keras.Model(inputs=x, outputs=model.call(x))
# plot_model(model, to_file='linguistic_decoder.png')

# # acoustic decoder
# x = tf.keras.Input(batch_shape=(10, 1, 80)) # batch_size: 10
# model = TempConvnet_Decoder(decoder_type='acoustic', num_levels=8, num_channels=80, kernel_size=[2, 2], 
#                             padding='causal', upsample_size={2: 5, 3: 2, 0:1}, dropout_rate=0.2, output_shape=(None, 80, 303), seed=1, training_state=True)
# model.build((10, 1, 80))
# model = tf.keras.Model(inputs=x, outputs=model.call(x))
# plot_model(model, to_file='acoustic_decoder.png')