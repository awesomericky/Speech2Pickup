import librosa
import numpy as np
from os import listdir, makedirs
from os.path import join, isfile, isdir
import pickle
from processed_data_loader import load_single_data
from utils import read_script_files, read_script_file_data

def audio_length_equalize_and_save(relative_data_directory_path, relative_save_data_directory_path):
    data_type = relative_data_directory_path.split('/')[-1]
    if data_type == 'data_v1.0':
        sampled_audios_idx = 3
        sample_rates_idx = 4

        # Load data
        data_files = [f for f in listdir(relative_data_directory_path) if isfile(join(relative_data_directory_path, f))]
        data_files.sort()
        total_file_num = len(data_files)
        print('{} file loaded'.format(total_file_num))

        # Find max sampled audio length
        max_sampled_audio_len = 0        
        for data_file in data_files:
            data = load_single_data(relative_data_directory_path, data_file)
            sampled_audios = data[sampled_audios_idx]
            for sampled_audio in sampled_audios:
                curr_sampled_audio_len = len(sampled_audio)
                if curr_sampled_audio_len > max_sampled_audio_len:
                    max_sampled_audio_len = curr_sampled_audio_len
        
        # Modify 'sampled_audios'
        # of 'data_v1.0' to make audio length same
        for i in range(total_file_num):
            print('Processing {}/{}'.format(i+1, total_file_num))
            data = load_single_data(relative_data_directory_path, data_files[i])
            sampled_audios = data[sampled_audios_idx]
            sampled_rates = data[sample_rates_idx]

            for ii in range(len(sampled_audios)):
                # Add zero padding to 'sampled audio'
                len_zero_padding = max_sampled_audio_len - len(sampled_audios[ii])
                sampled_audios[ii].extend([0]*len_zero_padding)

            data[sampled_audios_idx] = sampled_audios
            result = save_single_data(relative_save_data_directory_path, data, i+1)

    elif data_type == 'data_v2.0':
        sampled_audios_idx = 0
        sample_rates_idx = 1
        word_time_intervals_idx = 2

        # Load data
        data_files = [f for f in listdir(relative_data_directory_path) if isfile(join(relative_data_directory_path, f))]
        data_files.sort()
        total_file_num = len(data_files)
        print('{} file loaded'.format(total_file_num))
        
        # Find max sampled audio length
        max_sampled_audio_len = 0        
        for data_file in data_files:
            data = load_single_data(relative_data_directory_path, data_file)
            sampled_audios = data[sampled_audios_idx]
            for i in range(len(sampled_audios)):
                curr_sampled_audio_len = len(sampled_audios[i])
                if curr_sampled_audio_len > max_sampled_audio_len:
                    max_sampled_audio_len = curr_sampled_audio_len
                    # print(max_sampled_audio_len)
                    # print(data[2][i])
        
        # Modify 'sampled_audios' and 'word_time_intervals'
        # of 'data_v2.0' to make audio length same
        for i in range(total_file_num):
            print('Processing {}/{}'.format(i+1, total_file_num))
            data = load_single_data(relative_data_directory_path, data_files[i])
            sampled_audios = data[sampled_audios_idx]
            sampled_rates = data[sample_rates_idx]
            word_time_intervals = data[word_time_intervals_idx]

            for ii in range(len(sampled_audios)):
                # Add zero padding to 'sampled audio'
                len_zero_padding = max_sampled_audio_len - len(sampled_audios[ii])
                sampled_audios[ii] = np.append(sampled_audios[ii], [0]*len_zero_padding)

                # Add silent part to 'word_time_interval'
                fixed_end_time = round(max_sampled_audio_len/float(sampled_rates[ii]), 3)
                curr_end_time = word_time_intervals[ii][-1][-1]
                word_time_intervals[ii].append(["", curr_end_time, fixed_end_time])

            data[sampled_audios_idx] = sampled_audios
            data[word_time_intervals_idx] = word_time_intervals

            result = save_single_data(relative_save_data_directory_path, data, i+1)
    else:
        raise ValueError('Unavailable data directory path for audio zero padding')

def return_mel_spec_single_channel(sampled_audio, sample_rate, n_fft, hop_length, win_length, n_mels, window='hann', log_scale=True):
    audio_mel_spec = librosa.feature.melspectrogram(y=sampled_audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, n_mels=n_mels)
    if log_scale:
        audio_mel_spec = librosa.power_to_db(audio_mel_spec)
    return audio_mel_spec

def return_mel_spec_three_channel(sampled_audio, sample_rate, n_fft, hop_length, win_length, n_mels, window='hann', log_scale=True):
    audio_mel_spec = librosa.feature.melspectrogram(y=sampled_audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, n_mels=n_mels)
    if log_scale:
        audio_mel_spec = librosa.power_to_db(audio_mel_spec)
    mel_first_derivative = librosa.feature.delta(audio_mel_spec, width=5, axis=-1, mode='interp')
    mel_second_derivative = librosa.feature.delta(mel_first_derivative, width=5, axis=-1, mode='interp')
    audio_mel_feature = np.stack((audio_mel_spec, mel_first_derivative, mel_second_derivative), axis=-1)
    return audio_mel_feature

def word_interval_continualize(len_mel_spec, sample_rate, hop_length, word_time_interval, word_dictionary, word_dictionary_size):
    frame_time_interval = hop_length/float(sample_rate)
    continuous_word_interval = np.ones((word_dictionary_size, len_mel_spec))
    for word_time in word_time_interval:
        word = word_time[0]
        curr_word_start_time = word_time[1]
        curr_word_end_time = word_time[2]

        word = np.array(word_dictionary[word])[:, np.newaxis]
        curr_word_start_index = int(curr_word_start_time/frame_time_interval)
        curr_word_end_index = int(curr_word_end_time/frame_time_interval)
        continuous_word_interval[:, curr_word_start_index:curr_word_end_index] = np.multiply(continuous_word_interval[:, curr_word_start_index:curr_word_end_index], word)
    continuous_word_interval[:, curr_word_end_index:] = word_dictionary[''][:, np.newaxis]
    return continuous_word_interval

def make_word_dictionary(relative_script_directory_path):
    # Check whole word candidates
    script_files = read_script_files(relative_script_directory_path)
    total_words = set()
    for script_file in script_files:
        curr_file_lines = read_script_file_data(relative_script_directory_path, script_file)
        for curr_file_line in curr_file_lines:
            total_words.update(set(curr_file_line.split()))
    total_words = list(total_words)
    total_words.append("")  # blank
    total_words.append("OOV")  # Out of vocabulary
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

def save_single_data(relative_save_data_directory_path, data, save_file_num):
    # Check directiry to save data
    if not isdir(relative_save_data_directory_path):
        makedirs(relative_save_data_directory_path)
    
    # Save
    num_data = len(data[0])
    each_num_data = int(num_data/4)
    distrib_num_file = 4
    start = 0; end = start + each_num_data
    for i in range(1, distrib_num_file):
        file_name = relative_save_data_directory_path + '/senEM_preprocessed_{}.pkl'.format(distrib_num_file*(save_file_num-1)+i)
        print('Saving {} data'.format(distrib_num_file*(save_file_num-1)+i))
        with open(file_name, 'wb') as f:
            pickle.dump([data[0][start:end], data[1][start:end], data[2][start:end]], f)
        start = end
        end += each_num_data

    file_name = relative_save_data_directory_path + '/senEM_preprocessed_{}.pkl'.format(distrib_num_file*save_file_num)
    print('Saving {} data'.format(distrib_num_file*save_file_num))
    with open(file_name, 'wb') as f:
        pickle.dump([data[0][start:], data[1][start:], data[2][start:]], f)
    return True

# # Check make_word_dictionary()
# relative_script_directory_path = './data/train_script'
# make_word_dictionary(relative_script_directory_path)