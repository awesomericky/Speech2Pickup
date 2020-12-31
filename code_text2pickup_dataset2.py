from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf
import random
import time
from skimage import io
from skimage.transform import resize
import wandb
from hourglass_with_rnn import createModel

class Text2Pickup_dataset2:
    def __init__(self):
        pass
    
    def load_metadata_and_image(self, processed_file_path, img_path):
        processed_files = [f for f in listdir(processed_file_path) if isfile(join(processed_file_path, f))]
        img_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]

        # Load processed data
        print('Loading processed data')
        for f in processed_files:
            print('Loading {}/{}'.format(processed_files.index(f)+1, len(processed_files)))
            
            npzfile = np.load(join(processed_file_path, f))

            if processed_files.index(f) == 0:
                img_idx = npzfile['img_idx']
                sen_len = npzfile['seq_len']
                text_inputs = npzfile['inputs']
                pos_outputs = npzfile['outputs']
            else:
                img_idx = np.concatenate((img_idx, npzfile['img_idx']), axis=0)
                sen_len = np.concatenate((sen_len, npzfile['seq_len']), axis=0)
                text_inputs = np.concatenate((text_inputs, npzfile['inputs']), axis=0)
                pos_outputs = np.concatenate((pos_outputs, npzfile['outputs']), axis=0)

        # Load image
        img_resize = 256
        print('Loading image..')
        total_images = dict()
        n = 0
        for f in img_files:
            n += 1
            print('Loading {}/{}'.format(n, len(img_files)))
            tmp_img = io.imread(join(img_path, f))
            tmp_img = resize(tmp_img, [img_resize, img_resize], preserve_range=True)
            tmp_img = tmp_img / 255.0
            total_images[int(f.split('.')[0])] = tmp_img
        
        self.num_data = img_idx.shape[0]
        self.dim_sentence = text_inputs.shape[1]
        self.max_step_sentence = text_inputs.shape[2]

        return img_idx, sen_len, text_inputs, pos_outputs, total_images
    
    def shuffle_data(self, heatmap_load_batch_size=1500, shuffle_first=True):
        if shuffle_first:
            heatmap_load_freq = self.num_data//heatmap_load_batch_size
            self.heatmap_load_index = [i*heatmap_load_batch_size for i in range(heatmap_load_freq+1)]
            self.heatmap_load_index.append(self.num_data)

            self.data_order=list(range(self.num_data))

        random.shuffle(self.data_order)
    
    def load_heatmap(self, img_idx, pos_outputs, heatmap_path, heatmap_start_index, heatmap_end_index):
        for k in range(heatmap_start_index, heatmap_end_index):
            data_index = self.data_order[k]
            heatmap_name = ('/%04d_%04d_%04d_%04d_%04d.npz')% (img_idx[data_index, 0], pos_outputs[data_index, 0], pos_outputs[data_index, 1], pos_outputs[data_index, 2], pos_outputs[data_index, 3])
            heatmap_total_path = heatmap_path + heatmap_name
            heatmap = np.load(heatmap_total_path)['heatmap'][np.newaxis, :, :]

            if k == heatmap_start_index:
                heatmaps = heatmap
            else:
                heatmaps = np.concatenate((heatmaps, heatmap), axis=0)
        return heatmaps
    
    def train(self, img_resize, heatmap_resize,
          batch_size, max_epoch, save_stride,
          num_hg_Depth, dim_hg_feat, dim_rnn_cell,
          restore_flag, restore_path, restore_epoch,
          total_images, total_heatmaps_path, heatmap_load_batch_size,
          train_text_inputs, train_sen_len, train_img_idx, train_pos_outputs,
          learning_rate, model_save_path):
    
        ph_image = tf.placeholder(dtype=tf.float32, shape=[None, img_resize, img_resize, 3])
        ph_sen = tf.placeholder(dtype=tf.float32, shape=[None, self.dim_sentence, self.max_step_sentence])
        ph_sen_len = tf.placeholder(tf.int32, [None, 1])
        ph_heatmap = tf.placeholder(dtype=tf.float32, shape=[None, heatmap_resize, heatmap_resize, 1])
        ph_dropout = tf.placeholder(tf.float32)

        result_heatmap = createModel(curr_img = ph_image,
                                    curr_sen = ph_sen,
                                    curr_sen_len = ph_sen_len,
                                    img_size=heatmap_resize, 
                                    dim_sentence=self.dim_sentence,
                                    max_step_sentence=self.max_step_sentence, 
                                    num_hg_Depth=num_hg_Depth,
                                    dim_hg_feat=dim_hg_feat,
                                    dim_rnn_cell=dim_rnn_cell,
                                    dim_output=1,
                                    dr_rate=ph_dropout)
        loss = tf.reduce_mean((result_heatmap-ph_heatmap)**2)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()
        
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        print('Now ready to start the session.')
        
        with tf.Session(config=config) as sess:
            sess.run(init)

            if restore_flag == 1:
                saver.restore(sess, restore_path)

            for _epoch in range(max_epoch-restore_epoch):
                random.seed(_epoch)
                if _epoch == 0:
                    self.shuffle_data(heatmap_load_batch_size=heatmap_load_batch_size, shuffle_first=True)
                else:
                    self.shuffle_data(heatmap_load_batch_size=heatmap_load_batch_size, shuffle_first=False)

                total_train_loss = 0.0
                
                epoch_start_time = time.time()
                for h in range(len(self.heatmap_load_index)-1):
                    h_start = self.heatmap_load_index[h]
                    h_end = self.heatmap_load_index[h+1]

                    load_s_time = time.time()
                    partial_heatmaps = self.load_heatmap(train_img_idx, train_pos_outputs, total_heatmaps_path, h_start, h_end)
                    load_e_time = time.time()
                    print('heatmap loading time : %0.3f' % (load_e_time-load_s_time))

                    num_batch = heatmap_load_batch_size // batch_size
                    batch_shuffle = self.data_order[h_start:h_end]

                    for i in range(num_batch):
                        batch_idx = [batch_shuffle[idx] 
                                    for idx in range(i * batch_size, (i + 1) * batch_size)]
                        batch_inputs = train_text_inputs[batch_idx, :, :]
                        batch_seq_len = train_sen_len[batch_idx, :]
                        batch_img_idx = train_img_idx[batch_idx, :]
                        # batch_pos_output = train_pos_outputs[batch_idx, :]

                        batch_images = np.zeros((batch_size, img_resize, img_resize, 3))
                        batch_heatmaps = np.zeros((batch_size, heatmap_resize, heatmap_resize, 1))

                        for ii in range(len(batch_idx)):
                            tmp_img = total_images[batch_img_idx[ii, 0]]

                            batch_images[ii, :, :, :] = tmp_img        

                            tmp_heatmap = partial_heatmaps[i*batch_size + ii, :, :]

                            batch_heatmaps[ii, :, :, 0] = tmp_heatmap

                        train_feed_dict = {ph_sen: batch_inputs, ph_sen_len: batch_seq_len,
                                        ph_image: batch_images, ph_heatmap: batch_heatmaps,
                                        ph_dropout: 0.0}

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