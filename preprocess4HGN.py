### Import Dependencies
import numpy as np
from os import listdir, remove, makedirs
from os.path import isfile, join, isdir
from scipy import misc as misc
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mulnorm
# from utils import load_w2v
from processed_data_loader import load_single_data
from process_data import return_mel_spec_single_channel, return_mel_spec_three_channel
import random

# def preprocess_text2pickup():
#     ### Import word2vec model
#     w2v_path = './data/GoogleNews-vectors-negative300.bin'
#     w2v_model = load_w2v(w2v_path)

#     dim_embed = w2v_model['woman'].shape[0]

#     ### Calculate Maximum length of the language command
#     script_path = './data/train_script'
#     files = [f for f in listdir(script_path) if isfile(join(script_path, f))]

#     total_words = set()
#     max_len = 0
#     num_data = 0

#     for tmp_file in files:
#         curr_file = open(join(script_path, tmp_file), 'r')
#         curr_lines = curr_file.readlines()
#         for line in curr_lines:
#             words = line.split()
#             for word in words:
#                 if words.index(word) > 1:
#                     total_words.add(word)
#             if max_len < len(words)-2:
#                 max_len = len(words)-2
#         num_data += len(curr_lines)

#     total_words = list(total_words)

#     max_len = max_len + 5

#     print('Total data number: %d' % num_data)

#     ### Set Empty data arrays
#     inputs = np.zeros((num_data, dim_embed, max_len))
#     outputs = np.zeros((num_data, 2))
#     img_idx = np.zeros((num_data, 1))
#     seq_len = np.zeros((num_data, 1))
#     tmp_num = 0

#     ### Start Preprocess : delete # for re-generate heatmaps
#     for file_idx in (files):
#         curr_file = open(join(script_path, file_idx), 'r')
#         curr_lines = curr_file.readlines()

#         print('Now processing : %s/%s ... ' % (files.index(file_idx), len(files)))

#         prev_output = [0, 0]
#         for line in curr_lines:
#             words = line.split()
#             for i, word in enumerate(words):
#                 if i <= 1:
#                     outputs[tmp_num, i] = float(word)
#                 else:
#                     if word not in w2v_model.vocab.keys():
#                         inputs[tmp_num, :, i-2] = np.zeros((300,))
#                     else:
#                         inputs[tmp_num, :, i-2] = w2v_model[word]
#             img_idx[tmp_num, 0] = float(file_idx[0:4])
#             seq_len[tmp_num, 0] = len(words)-2

#             '''
#             if prev_output[0] != outputs[tmp_num, 0] or prev_output[1] != outputs[tmp_num, 1]:
#                 tmp_heatmap = np.zeros((250, 250))
#                 tmp_output = outputs[tmp_num, :].astype(int)
#                 print 'generate heatmap at %03d, %03d' % (tmp_output[0], tmp_output[1])

#                 tmp_cov = [[500, 0], [0, 500]]
#                 mvn = mulnorm([tmp_output[1], tmp_output[0]], tmp_cov)

#                 for k in range(tmp_heatmap.shape[0]):
#                     for kk in range(tmp_heatmap.shape[1]):
#                         tmp_heatmap[k, kk] = mvn.pdf([k, kk])

#                 tmp_heatmap = tmp_heatmap / np.max(tmp_heatmap)
#                 np.savez(('./data/train_heatmap/%s_%03d_%03d.npz')%(file_idx[0:4], tmp_output[0], tmp_output[1] ), tmp_heatmap)

#             prev_output = outputs[tmp_num, :]
#             '''        
#             tmp_num += 1        

#     ### Save preprocessed data
#     np.savez('./data/preprocessed4HGN.npz', img_idx, seq_len, inputs, outputs, total_words)


#############################
### Added to original file ##
#############################
def preprocess_speech2pickup(relative_data_directory_path, relative_save_data_directory_path):
    if not isdir(relative_save_data_directory_path):
        makedirs(relative_save_data_directory_path)
    
    data_files = [f for f in listdir(relative_data_directory_path) if isfile(join(relative_data_directory_path, f))]
    random.shuffle(data_files)

    img_idx = []
    seq_len = []
    inputs = []
    outputs = []
    sentence = []
    DATA_INDEX_IMG_IDXS = 0
    DATA_INDEX_POSE_OUTPUTS = 1
    DATA_INDEX_SENTENCE_LENS = 2
    DATA_INDEX_SAMPLED_AUDIOS = 3
    DATA_INDEX_SAMPLED_RATES = 4
    DATA_INDEX_TEXT_COMMANDS = 5

    # Set mel spectogram configuration
    n_fft = 2048
    hop_length = int(n_fft/8)
    win_length = int(n_fft/2)
    n_mels = 40
    mel_feature_type = relative_save_data_directory_path.split('_')[-2]

    for data_file in data_files:
        print('Processing {} ..'.format(data_file))
        data = load_single_data(relative_data_directory_path, data_file)

        img_idx.extend(data[DATA_INDEX_IMG_IDXS])
        seq_len.extend(data[DATA_INDEX_SENTENCE_LENS])
        outputs.extend(data[DATA_INDEX_POSE_OUTPUTS])
        sentence.extend(data[DATA_INDEX_TEXT_COMMANDS])

        for i in range(len(data[DATA_INDEX_SAMPLED_RATES])):
            if mel_feature_type == 'single':
                mel_spec = return_mel_spec_single_channel(sampled_audio=data[DATA_INDEX_SAMPLED_AUDIOS][i], sample_rate=data[DATA_INDEX_SAMPLED_RATES][i], \
                                                            n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
            elif mel_feature_type == 'three':
                mel_spec = return_mel_spec_three_channel(sampled_audio=data[DATA_INDEX_SAMPLED_AUDIOS][i], sample_rate=data[DATA_INDEX_SAMPLED_RATES][i], \
                                                            n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
            else:
                raise ValueError('Unsupported mel feature type')
            inputs.append(mel_spec)
    
    """
    # Generate heatmap (delete # to re-generate heatmaps)
    prev_output = [0, 0]
    for i in range(len(outputs)):
        if prev_output[0] != outputs[i][0] or prev_output[1] != outputs[i][1]:
            tmp_heatmap = np.zeros((250, 250))
            tmp_output = outputs[i].astype(int)
            print('generate heatmap at ({}, {})'.format(tmp_output[0], tmp_output[1]))

            tmp_cov = [[500, 0], [0, 500]]
            mvn = mulnorm([tmp_output[1], tmp_output[0]], tmp_cov)

            for k in range(tmp_heatmap.shape[0]):
                for kk in range(tmp_heatmap.shape[1]):
                    tmp_heatmap[k, kk] = mvn.pdf([k, kk])

            tmp_heatmap = tmp_heatmap / np.max(tmp_heatmap)
            np.savez('./data/train_heatmap/{}_{}_{}.npz'.format(img_idx[i], tmp_output[0], tmp_output[1]), tmp_heatmap)

        prev_output = outputs[i]
    """
    
    img_idx = np.asarray(img_idx)[:, np.newaxis]
    seq_len = np.asarray(seq_len)[:, np.newaxis]
    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs)
    print('img_idx shape: {}'.format(np.shape(img_idx)))
    print('seq_len shape: {}'.format(np.shape(seq_len)))
    print('inputs shape: {}'.format(np.shape(inputs)))
    print('outputs shape: {}'.format(np.shape(outputs)))
    print('sentence shape: {}'.format(len(sentence)))

    ### Save preprocessed data
    file_name = 'preprocessed4HGN_speech2pickup.npz'
    file_name = join(relative_save_data_directory_path, file_name)
    np.savez(file_name, img_idx=img_idx, seq_len=seq_len, inputs=inputs, outputs=outputs, sentence=sentence)

# # Save data_v1.2
# relative_data_directory_path = './data/data_v1.1'
# relative_save_data_directory_path = './data/data_v1.2_single_channel'
# preprocess_speech2pickup(relative_data_directory_path=relative_data_directory_path, relative_save_data_directory_path=relative_save_data_directory_path)