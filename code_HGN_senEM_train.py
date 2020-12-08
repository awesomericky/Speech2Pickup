import tensorflow as tf
import numpy as np
import random
import time
import wandb
from hourglass_with_senEM import createModel

def train(img_resize, heatmap_resize, num_hg_Depth, dim_hg_feat,
          n_mels, time_steps, encoder_model_path, encoder_args, input_shapes, seed, training_state,
          batch_size, max_epoch, num_train, save_stride, learning_rate, dropout_rate,
          restore_flag, restore_path, restore_epoch,
          total_images, total_heatmaps, train_speech_inputs, train_img_idx, train_pos_outputs, model_save_path):
    
    num_batch = num_train // batch_size
    
    ph_image = tf.placeholder(dtype=tf.float32, shape=[None, img_resize, img_resize, 3])
    ph_speech = tf.placeholder(dtype=tf.float32, shape=[None, n_mels, time_steps])
    ph_heatmap = tf.placeholder(dtype=tf.float32, shape=[None, heatmap_resize, heatmap_resize, 1])
    ph_dropout = tf.placeholder(tf.float32)

    result_heatmap, senEM_encoder = createModel(curr_img = ph_image,
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
    loss = tf.reduce_mean((result_heatmap-ph_heatmap)**2)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(var_list=tf.trainable_variables())
    # encoder_saver = tf.train.Saver(var_list=[v for v in tf.trainable_variables() if v.name.split('/')[0] == 'temp_convnet'])

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    print('Now ready to start the session.')

    with tf.Session(config=config) as sess:
        sess.run(init)

        if restore_flag == 0:
            senEM_encoder.load_weights(encoder_model_path)
            # encoder_saver.restore(sess, encoder_model_path)
        elif restore_flag == 1:
            saver.restore(sess, restore_path)
        
        for _epoch in range(max_epoch-restore_epoch):
            random.seed(_epoch)
            batch_shuffle = [i for i in range(num_train)]
            random.shuffle(batch_shuffle)

            total_train_loss = 0.0
            
            epoch_start_time = time.time()
            for i in range(num_batch):
                batch_idx = [batch_shuffle[idx] 
                             for idx in range(i * batch_size, (i + 1) * batch_size)]
                batch_speech_inputs = train_speech_inputs[batch_idx, :, :]
                batch_img_idx = train_img_idx[batch_idx, :]
                batch_pos_output = train_pos_outputs[batch_idx, :]

                batch_images = np.zeros((batch_size, img_resize, img_resize, 3))
                batch_heatmaps = np.zeros((batch_size, heatmap_resize, heatmap_resize, 1))

                for ii in range(len(batch_idx)):
                    tmp_img = total_images[batch_img_idx[ii, 0]]

                    batch_images[ii, :, :, :] = tmp_img        

                    tmp_heatmap = total_heatmaps[(batch_img_idx[ii, 0], batch_pos_output[ii, 0], batch_pos_output[ii, 1])]

                    batch_heatmaps[ii, :, :, 0] = tmp_heatmap

                train_feed_dict = {ph_image: batch_images, ph_speech: batch_speech_inputs,
                                   ph_heatmap: batch_heatmaps, ph_dropout: dropout_rate}

                sess.run(optimizer, feed_dict=train_feed_dict)
                curr_train_loss = sess.run(loss, feed_dict=train_feed_dict)
                total_train_loss += curr_train_loss

                batch_end_time = time.time()
                total_time = batch_end_time - epoch_start_time 
                if i % 100 == 0:
                    print("batch loss : %s -> about %0.3f second left to finish this epoch" 
                          % (curr_train_loss, (total_time/(i+1))*(num_batch-i) ))
                    wandb.log({'Batch train loss': curr_train_loss})

            total_train_loss = total_train_loss / num_batch

            print('current epoch : ' + str(_epoch+1+restore_epoch), 
                  ', current train loss : ' + str(total_train_loss))
            wandb.log({'Epoch train loss': total_train_loss})
            
            if (_epoch+1+restore_epoch) % save_stride == 0:
                model_saved_path = saver.save(sess, model_save_path)
                print("Model saved in file : %s" % model_saved_path)