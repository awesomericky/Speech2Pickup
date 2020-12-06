import numpy as np
import tensorflow as tf
from plot_model import plot_model
# from tensorflow.keras.utils import plot_model

class ResidualBlock(tf.keras.Model):
    def __init__(self, dilation_rate, num_filters, kernel_size, padding, 
                        dropout_rate, activation, seed, training_state):
        super(ResidualBlock, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=seed)
        layers = tf.keras.layers
        assert padding in ['causal', 'same']

        self.training_state = training_state

        # Block1
        self.conv1 = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, data_format='channels_last',
                                    dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
        self.batch1 = layers.BatchNormalization(axis=1, trainable=True)
        if activation == 'leaky-relu':
            self.ac1 = layers.LeakyReLU(alpha=0.3)
        else:
            self.ac1 = layers.Activation(activation)
        self.drop1 = layers.Dropout(rate=dropout_rate)

        # Block2
        self.conv2 = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, data_format='channels_last',
                                    dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
        self.batch2 = layers.BatchNormalization(axis=1, trainable=True)
        if activation == 'leaky-relu':
            self.ac2 = layers.LeakyReLU(alpha=0.3)
        else:
            self.ac2 = layers.Activation(activation)
        self.drop2 = layers.Dropout(rate=dropout_rate)

        self.downsample = layers.Conv1D(filters=num_filters, kernel_size=1,
                                        padding='same', kernel_initializer=init)
        if activation == 'leaky-relu':
            self.ac3 = layers.LeakyReLU(alpha=0.3)
        else:
            self.ac3 = layers.Activation(activation)
    
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
        assert prev_x.shape[1:] == x.shape[1:]

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
    def __init__(self, num_channels, kernel_size, dropout_rate, activation, seed, training_state):
        # num_channels is a list contains hidden channel numbers of Conv1D
        # len(num_channels) is number of convolutional layers in one Temporal Block
        super(TemporalBlock, self).__init__()
        assert isinstance(num_channels, list)

        self.num_levels = len(num_channels)
        self.resi_blocks = [0]*self.num_levels
        for i in range(self.num_levels):
            dilation_rate = 2**i
            self.resi_blocks[i] = ResidualBlock(dilation_rate, num_channels[i], kernel_size, padding='causal',
                            dropout_rate=dropout_rate, activation=activation, seed=seed, training_state=training_state)
    
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
    def __init__(self, num_stacks, num_channels, kernel_size, dropout_rate, activation, return_type, seed, training_state):
        # num_stacks number of Temporal Blocks in Temporal convolutional network
        super(TempConvnet, self).__init__()
        assert isinstance(num_stacks, int)
        assert isinstance(num_channels, list)
        assert isinstance(activation, str)
        assert return_type in ['whole', 'end']

        self.num_stacks = num_stacks
        self.temp_blocks = [0]*self.num_stacks
        self.return_type = return_type
        for i in range(num_stacks):
            self.temp_blocks[i] = TemporalBlock(num_channels, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation, seed=seed, training_state=training_state)
    
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
    def __init__(self, decoder_type, num_channels, kernel_size, padding, dropout_rate, activation, seed, training_state):
        super(TempConvnet_Decoder, self).__init__()
        assert isinstance(num_channels, int)
        assert isinstance(kernel_size, list)
        assert isinstance(activation, str)
        assert padding in ['causal', 'same']
        assert decoder_type in ['linguistic', 'acoustic']

        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=seed)
        layers = tf.keras.layers

        self.training_state = training_state

        # Block1
        self.upsample_block1 = layers.UpSampling1D(size=3)
        self.conv_block1 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block1 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block1 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block1 = layers.Dropout(rate=dropout_rate)

        # Extra block0
        self.conv_block_e0 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block_e0 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block_e0 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block_e0 = layers.Dropout(rate=dropout_rate)

        # Block2
        self.upsample_block2 = layers.UpSampling1D(size=3)
        self.conv_block2 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block2 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block2 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block2 = layers.Dropout(rate=dropout_rate)

        # Extra block1
        self.conv_block_e1 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block_e1 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block_e1 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block_e1 = layers.Dropout(rate=dropout_rate)

        # Block3
        self.upsample_block3 = layers.UpSampling1D(size=2)
        self.conv_block3 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block3 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block3 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block3 = layers.Dropout(rate=dropout_rate)

        # Extra block2
        self.conv_block_e2 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block_e2 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block_e2 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block_e2 = layers.Dropout(rate=dropout_rate)

        # Block4
        self.conv_block4 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding='valid', kernel_initializer=init)
        self.batchnorm_block4 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block4 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block4 = layers.Dropout(rate=dropout_rate)

        # Extra block3
        self.conv_block_e3 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block_e3 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block_e3 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block_e3 = layers.Dropout(rate=dropout_rate)

        # Block5
        self.upsample_block5 = layers.UpSampling1D(size=2)
        self.conv_block5 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block5 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block5 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block5 = layers.Dropout(rate=dropout_rate)

        # Extra block4
        self.conv_block_e4 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block_e4 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block_e4 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block_e4 = layers.Dropout(rate=dropout_rate)

        # Block6
        self.upsample_block6 = layers.UpSampling1D(size=3)
        self.conv_block6 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block6 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block6 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block6 = layers.Dropout(rate=dropout_rate)

        # Extra block5
        self.conv_block_e5 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block_e5 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block_e5 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block_e5 = layers.Dropout(rate=dropout_rate)

        # Block7
        self.conv_block7 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding='valid', kernel_initializer=init)
        self.batchnorm_block7 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block7 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block7 = layers.Dropout(rate=dropout_rate)

        # Extra block6
        self.conv_block_e6 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block_e6 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block_e6 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block_e6 = layers.Dropout(rate=dropout_rate)

        # Block8
        self.upsample_block8 = layers.UpSampling1D(size=3)
        self.conv_block8 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[0], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block8 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block8 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block8 = layers.Dropout(rate=dropout_rate)

        # Extra block7
        self.conv_block_e7 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[1], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block_e7 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block_e7 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block_e7 = layers.Dropout(rate=dropout_rate)

        # Extra block8
        self.conv_block_e8 = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[1], data_format='channels_last',
                                            padding=padding, kernel_initializer=init)
        self.batchnorm_block_e8 = layers.BatchNormalization(axis=-1, trainable=True)
        self.ac_block_e8 = layers.Activation(activation) if activation != 'leaky-relu' else layers.LeakyReLU(alpha=0.2)
        self.drop_block_e8 = layers.Dropout(rate=dropout_rate)

        # Block9 (final block)
        self.final_conv_block = layers.Conv1D(filters=num_channels, kernel_size=kernel_size[1], data_format='channels_last',
                                                    padding=padding, kernel_initializer=init)
        self.final_ac_block = layers.Softmax(axis=-1)
        self.final_reshape_block = layers.Permute((2, 1))
    
    def call(self, x):
        x = self.upsample_block1(x)
        x = self.conv_block1(x)
        x = self.batchnorm_block1(x)
        x = self.ac_block1(x)
        x = self.drop_block1(x) if self.training_state else x
        x = self.conv_block_e0(x)
        x = self.batchnorm_block_e0(x)
        x = self.ac_block_e0(x)
        x = self.drop_block_e0(x) if self.training_state else x
        x = self.upsample_block2(x)
        x = self.conv_block2(x)
        x = self.batchnorm_block2(x)
        x = self.ac_block2(x)
        x = self.drop_block2(x) if self.training_state else x
        x = self.conv_block_e1(x)
        x = self.batchnorm_block_e1(x)
        x = self.ac_block_e1(x)
        x = self.drop_block_e1(x) if self.training_state else x
        x = self.upsample_block3(x)
        x = self.conv_block3(x)
        x = self.batchnorm_block3(x)
        x = self.ac_block3(x)
        x = self.drop_block3(x) if self.training_state else x
        x = self.conv_block_e2(x)
        x = self.batchnorm_block_e2(x)
        x = self.ac_block_e2(x)
        x = self.drop_block_e2(x) if self.training_state else x
        x = self.conv_block4(x)
        x = self.batchnorm_block4(x)
        x = self.ac_block4(x)
        x = self.drop_block4(x) if self.training_state else x
        x = self.conv_block_e3(x)
        x = self.batchnorm_block_e3(x)
        x = self.ac_block_e3(x)
        x = self.drop_block_e3(x) if self.training_state else x
        x = self.upsample_block5(x)
        x = self.conv_block5(x)
        x = self.batchnorm_block5(x)
        x = self.ac_block5(x)
        x = self.drop_block5(x) if self.training_state else x
        x = self.conv_block_e4(x)
        x = self.batchnorm_block_e4(x)
        x = self.ac_block_e4(x)
        x = self.drop_block_e4(x) if self.training_state else x
        x = self.upsample_block6(x)
        x = self.conv_block6(x)
        x = self.batchnorm_block6(x)
        x = self.ac_block6(x)
        x = self.drop_block6(x) if self.training_state else x
        x = self.conv_block_e5(x)
        x = self.batchnorm_block_e5(x)
        x = self.ac_block_e5(x)
        x = self.drop_block_e5(x) if self.training_state else x
        x = self.conv_block7(x)
        x = self.batchnorm_block7(x)
        x = self.ac_block7(x)
        x = self.drop_block7(x) if self.training_state else x
        x = self.conv_block_e6(x)
        x = self.batchnorm_block_e6(x)
        x = self.ac_block_e6(x)
        x = self.drop_block_e6(x) if self.training_state else x
        x = self.upsample_block8(x)
        x = self.conv_block8(x)
        x = self.batchnorm_block8(x)
        x = self.ac_block8(x)
        x = self.drop_block8(x) if self.training_state else x
        x = self.conv_block_e7(x)
        x = self.batchnorm_block_e7(x)
        x = self.ac_block_e7(x)
        x = self.drop_block_e7(x) if self.training_state else x
        x = self.conv_block_e8(x)
        x = self.batchnorm_block_e8(x)
        x = self.ac_block_e8(x)
        x = self.drop_block_e8(x) if self.training_state else x
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