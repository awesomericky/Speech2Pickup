import tensorflow as tf
import numpy as np
from hourglass_with_senEM import createModel

def define_model(n_mels, time_steps, num_hg_Depth, dim_hg_feat,
                encoder_args, input_shapes, seed, training_state):
    img_resize = 256
    heatmap_resize = 64

    ph_image = tf.placeholder(dtype=tf.float32, shape=[None, img_resize, img_resize, 3])
    ph_speech = tf.placeholder(dtype=tf.float32, shape=[None, n_mels, time_steps])
    ph_dropout = tf.placeholder(tf.float32)

    result_heatmap, _ = createModel(curr_img = ph_image,
                                    curr_speech=ph_speech,
                                    img_size=heatmap_resize, 
                                    num_hg_Depth=num_hg_Depth,
                                    dim_hg_feat=dim_hg_feat,
                                    dim_output=1,
                                    dr_rate=ph_dropout,
                                    encoder_args=encoder_args,
                                    input_shapes=input_shapes,
                                    seed=seed,
                                    training_state=training_state)
    return result_heatmap, ph_image, ph_speech, ph_dropout
    
def session_run(result_heatmap, ph_image, ph_speech, ph_dropout,
                restore_path, curr_test_img, curr_speech, dropout_rate):
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    T = 100

    with tf.Session(config=config) as sess:
        sess.run(init)

        saver.restore(sess, restore_path)

        test_feed_dict = {ph_image: np.tile(curr_test_img, (T, 1, 1, 1)),
                          ph_speech: np.tile(curr_speech, (T, 1, 1)),
                          ph_dropout: dropout_rate}
        test_heatmap = sess.run(result_heatmap, feed_dict=test_feed_dict)

    test_heatmap = np.squeeze(test_heatmap)
    mean_of_esti = np.mean(test_heatmap, axis=0)
    mean_of_squared_esti = np.mean(test_heatmap**2, axis=0)
    squared_mean_of_esti = mean_of_esti ** 2
    uncertainty = np.sqrt(mean_of_squared_esti - squared_mean_of_esti)
    
    return mean_of_esti, uncertainty
