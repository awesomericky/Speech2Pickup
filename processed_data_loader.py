import os
from os import listdir
from os.path import isfile, join, isdir
import pickle
import numpy as np
import cv2
import librosa.display
import matplotlib.pyplot as plt
import pdb
from utils import read_script_files, read_script_file_data

def load_data(relative_data_directory_path, data_type):
    total_data = []
    data_files = [f for f in listdir(relative_data_directory_path) if isfile(join(relative_data_directory_path, f))]
    data_files.sort()
    for data_file in data_files:
        if data_type == 'pickle':
            with open(join(relative_data_directory_path, data_file), 'rb') as f:
                data_list = pickle.load(f)
                total_data.append(data_list)

                # # Logging for 'Data_v1.0'
                # print('='*20)
                # print(len(data_list))
                # print(len(data_list[0])); print(len(data_list[1])); print(len(data_list[2])); print(len(data_list[3])); print(len(data_list[4])); print(len(data_list[5]))
                # print(data_list[0][0]); print(data_list[1][0]); print(data_list[2][0]); print(data_list[3][0]); print(data_list[4][0]); print(data_list[5][0])

                # # Logging for 'Data_v2.0' and 'Data_v2.1'
                # print('='*20)
                # print(len(data_list))
                # print(len(data_list[0])); print(len(data_list[1])); print(len(data_list[2]))
                # print(data_list[0][0]); print(data_list[1][0]); print(data_list[2][0])
                # print(len(data_list[0][0])); print(len(data_list[0][10])); print(len(data_list[0][40]))
        elif data_type == 'np':
            # Logging for 'Data_v2.2'
            data_list = np.load(join(relative_data_directory_path, data_file))
            word_dic, _ = make_word_dictionary('./data/train_script')
            word_list = list(word_dic.keys())
            word_label_list = np.array(list(word_dic.values()))
            # total_data.append(data_list)

            # print('='*20)
            # print(np.shape(data_list['arr_0'])); print(np.shape(data_list['arr_1']))
            # print(data_list['arr_0']); print(data_list['arr_1'])

            # Show corresponding sentence
            sentence = []
            new_sentence = []
            mul_result = np.matmul(data_list['arr_1'].T, word_label_list)
            word_indexs = np.argwhere(mul_result==1)
            word_indexs = word_indexs[:, 1]
            for word_index in word_indexs:
                sentence.append(word_list[word_index])
            for word in sentence:
                if word not in new_sentence:
                    new_sentence.append(word)
            new_sentence = ' '.join(new_sentence)
            print(new_sentence)

            # Show corresponding Mel-spectogram & Binary image
            fig, ax = plt.subplots()
            img = librosa.display.specshow(data_list['arr_0'], hop_length=256, x_axis='time',
                            y_axis='mel', sr=16000, cmap='viridis',
                            fmax=8000, ax=ax)
            fig.colorbar(img, ax=ax)
            plt.savefig('mel')
            cv2.imshow('Binary', data_list['arr_1'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            pdb.set_trace()
        else:
            raise ValueError('Unavailable data type')
    return total_data

def load_single_data(relative_data_directory_path, file_name):
    with open(join(relative_data_directory_path, file_name), 'rb') as f:
        data = pickle.load(f)
    return data

def load_single_npz_data(relative_data_directory_path, file_name):
    file_path = join(relative_data_directory_path, file_name)
    data = np.load(file=file_path)
    return data

def make_word_dictionary(relative_script_directory_path):
    # Check whole word candidates
    script_files = read_script_files(relative_script_directory_path)
    total_words = set()
    for script_file in script_files:
        curr_file_lines = read_script_file_data(relative_script_directory_path, script_file)
        for curr_file_line in curr_file_lines:
            total_words.update(set(curr_file_line.split()))
    total_words = list(total_words)
    total_words.append("")
    total_words.sort()

    # One-hot encoding
    word_dictionary = dict()
    word_dictionary_size = len(total_words)
    for i in range(word_dictionary_size):
        one_hot = np.zeros(word_dictionary_size)
        one_hot[i] = 1
        word_dictionary[total_words[i]] = one_hot
    # print(word_dictionary)
    return word_dictionary, word_dictionary_size

# # Load data_v1.0 (total data)
# relative_data_directory_path = './data/data_v1.0'
# load_data(relative_data_directory_path)

# # Load data_v2.0 (total data)
# relative_data_directory_path = './data/data_v2.0'
# load_data(relative_data_directory_path)

# # Load data_v2.1 (total data)
# relative_data_directory_path = './data/data_v2.1'
# load_data(relative_data_directory_path, 'pickle')

# # Load data_v2.1 (total data)
# relative_data_directory_path = './data/data_v2.2'
# load_data(relative_data_directory_path, 'np')

# # Load data_v2.2 (single data)
# relative_data_directory_path = './data/data_v2.2'
# a = load_single_npz_data(relative_data_directory_path, 'senEM_preprocessed_10.npz')
# print(a['arr_0'])
# print(a['arr_1'])