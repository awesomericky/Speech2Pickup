import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from residual import Residual
from sentenceEM import sentenceEM_encoder

stddev = 0.01

def hourglass_with_senEM(curr_input, senEM_output, numDepth, numIn, numOut, dr_rate):
    up1 = curr_input
    up1 = Residual(up1, numIn, numOut, dr_rate)

    low1 = tf.layers.max_pooling2d(curr_input, 2, 2)
    low1 = Residual(low1, numIn, numOut, dr_rate)
    
    if numDepth > 1:
        low2 = hourglass_with_senEM(low1, senEM_output, numDepth-1, numIn, numOut, dr_rate)
    else:
        low2 = Residual(low1, numIn, numOut, dr_rate)
    
    low3 = Residual(low2, numIn, numOut-1, dr_rate)
    
    with tf.variable_scope("speech_weights", reuse=True):
        exec('_W_o%d = tf.get_variable('"'W_o%d'"')' % (numDepth, numDepth))
        exec('_b_o%d = tf.get_variable('"'b_o%d'"')' % (numDepth, numDepth))
    
    exec('fitted_senEM = tf.matmul(senEM_output, _W_o%d)+_b_o%d' % (numDepth, numDepth))  
    fitted_senEM = tf.nn.dropout(fitted_senEM, 1-dr_rate)
    
    fitted_senEM = tf.reshape(fitted_senEM, [tf.shape(senEM_output)[0], 
                                        tf.to_int32(tf.sqrt(tf.to_float(tf.shape(fitted_senEM)[1]))),
                                        tf.to_int32(tf.sqrt(tf.to_float(tf.shape(fitted_senEM)[1])))])
    fitted_senEM = tf.expand_dims(fitted_senEM, axis=3)
    
    fuse_low3 = tf.concat([low3, fitted_senEM], axis=3)
        
    up2 = tf.image.resize_nearest_neighbor(fuse_low3, 2*tf.shape(fuse_low3)[1:3])

    return tf.add(up1, up2)

def lin(curr_input, numIn, numOut, dr_rate):
    l = tf.layers.conv2d(curr_input, numOut, 1, padding='Same')
    l = tf.layers.dropout(l, rate=dr_rate)
    l = tf.layers.batch_normalization(l)
    return tf.nn.relu(l)

def createModel(curr_img, curr_speech, img_size, num_hg_Depth, dim_hg_feat, dim_output, dr_rate,
                encoder_args, input_shapes, seed, training_state):
    
    # image size must be 256 by 256.
    senEM = sentenceEM_encoder(encoder_args, input_shapes, seed, training_state)
    senEM_output = senEM.call(curr_speech)
    dim_embedded_output = senEM_output.shape

    curr_senEM_encoder = senEM_encoder_w_connection(img_size=img_size, hg_depth=num_hg_Depth, senEM=senEM,
                                                    dim_embedded_output=dim_embedded_output)
    senEM_output = curr_senEM_encoder.encode(curr_speech)

    with vs.variable_scope('HGN'):
        with vs.variable_scope('pre'):
            cnv1 = tf.layers.conv2d(curr_img, filters=dim_hg_feat/4, kernel_size=7, strides=2, padding='Same')
            cnv1 = tf.layers.dropout(cnv1, rate=dr_rate) if training_state else cnv1

            cnv1 = tf.layers.batch_normalization(cnv1)
            cnv1 = tf.nn.relu(cnv1)
        
        with vs.variable_scope('r1'):
            r1 = Residual(cnv1, dim_hg_feat/4, dim_hg_feat/2, dr_rate)

        pool = tf.layers.max_pooling2d(r1, 2, 2)

        with vs.variable_scope('r4'):    
            r4 = Residual(pool, dim_hg_feat/2, dim_hg_feat/2, dr_rate)

        with vs.variable_scope('r5'):    
            r5 = Residual(r4, dim_hg_feat/2, dim_hg_feat, dr_rate)    
    
        with vs.variable_scope('hg'):  
            hg = hourglass_with_senEM(r5, senEM_output, num_hg_Depth, dim_hg_feat, dim_hg_feat, dr_rate)

        with vs.variable_scope('ll'):      
            ll = Residual(hg, dim_hg_feat, dim_hg_feat, dr_rate)
            ll = lin(ll, dim_hg_feat, dim_hg_feat, dr_rate)
            
        with vs.variable_scope('out'): 
            Out = tf.layers.conv2d(ll, filters=dim_output, kernel_size=1, strides=1, padding='Same')
            Out = tf.layers.dropout(Out, rate=dr_rate) if training_state else Out
    
    return Out

class senEM_encoder_w_connection:
    def __init__(self, img_size, hg_depth, senEM, dim_embedded_output):
        self.img_size = img_size
        self.hg_depth = hg_depth
        self.senEM = senEM
        self.dim_embedded_output = dim_embedded_output

        with tf.variable_scope('HGN/hg/speech_weights'):
            for i in range(hg_depth):
                exec('self.W_o%d = tf.get_variable('"'W_o%d'"', dtype=tf.float32,\
                                        initializer=tf.random_normal([self.dim_embedded_output, \
                                                    (self.img_size/(2**(self.hg_depth+1-%d)))**2], \
                                                    stddev=stddev))' % (i+1, i+1, i+1))
                exec('self.b_o%d = tf.get_variable('"'b_o%d'"', dtype=tf.float32,\
                                       initializer=tf.random_normal([(self.img_size/(2**(self.hg_depth+1-%d)))**2], \
                                                                     stddev=stddev))' % (i+1, i+1, i+1))
    def encode(self, x):
        with tf.variable_scope('HGN/senEM'):
            self.senEM.build_graph()
            senEM_model = self.senEM.encoder_model
            embedded_outputs = senEM_model(x)

        return embedded_outputs