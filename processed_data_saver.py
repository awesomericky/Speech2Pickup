"""
[Data type]

1. 'Data_v1.0'
    Purpose: Training whole speech2pickup network
    Content: [img_idxs, pose_outputs, sentence_lens, sampled_audios, sample_rates, text_commands]
2. 'Data_v1.1'
    Purpose: Training whole speech2pickup network (audio length same)
    Content: [img_idxs, pose_outputs, sentence_lens, sampled_audios, sample_rates, text_commands]
3. 'Data_v2.0'
    Purpose: Training sentence embedding network
    Content: [sampled_audios, sample_rates, word_time_intervals]
4. 'Data_v2.1'
    Purpose: Training sentence embedding network (audio length same)
    Content: [sampled_audios, sample_rates, word_time_intervals]
5. 'Data_v2.2'
    Purpose: Training sentence embedding network (audio length same, words are one-hot encoded)
    Content: [mel_spectograms, word_label_in_time_interval]
"""
import os
from os import listdir, makedirs
from os.path import isfile, join, isdir
import librosa
import numpy as np
import pickle
import textgrid
from utils import read_audio_files
from process_data import audio_length_equalize_and_save, return_mel_spec_single_channel, return_mel_spec_three_channel, word_interval_continualize, make_word_dictionary
from processed_data_loader import load_single_data

def save_data_v1_0(relative_audio_directory_path, sample_sr, relative_save_data_directory_path):
    # Read script data
    relative_script_directory_path = './data/train_script'
    script_files = [f for f in listdir(relative_script_directory_path) if isfile(join(relative_script_directory_path, f))]
    script_files.sort()
    total_num_data = 0
    total_lines = []
    for script_file in script_files:
        curr_file = open(join(relative_script_directory_path, script_file), 'r')
        curr_lines = curr_file.readlines()
        total_lines.append(curr_lines)
        total_num_data += len(curr_lines)

    # Set empty data list
    current_num_data = 0
    save_file_change_freq = 5000
    data_dir = relative_save_data_directory_path
    img_idxs = []
    pose_outputs = []
    sentence_lens = []
    sampled_audios = []
    sample_rates = []
    text_commands = []

    # Check whether extra saving is needed for unsaved data
    extra_save_bool = False
    if not total_num_data % save_file_change_freq == 0:
        extra_save_bool = True
    
    # Check directiry to save data
    if not isdir(data_dir):
        makedirs(data_dir)

    # Saving file config
    voice_config = relative_audio_directory_path.split('/')[-1].split('_')
    
    # Read audio data
    audio_files = [f for f in listdir(relative_audio_directory_path) if isfile(join(relative_audio_directory_path, f))]
    audio_files.sort()
    for audio_file in audio_files:
        current_num_data += 1
        print('Procressing: {}/{}'.format(current_num_data, total_num_data))

        audio_file_feature = audio_file.split('_')
        image_idx = audio_file_feature[1]
        sentence_idx = audio_file_feature[2][:-4]

        idx = int(image_idx)
        sentence_idx = int(sentence_idx) - 1
        sentence = total_lines[idx][sentence_idx]
        words = sentence.split()
        sentence = ' '.join(words[2: ])

        # Sample audio
        wav_file_path = join(relative_audio_directory_path, audio_file)
        sr = sample_sr
        sampled_audio, sample_rate = librosa.load(wav_file_path, sr=sr)

        # Append data to save
        img_idxs.append(idx)
        pose_outputs.append([float(words[0]), float(words[1])])
        sentence_lens.append(len(words)-2)
        sampled_audios.append(sampled_audio)
        sample_rates.append(sample_rate)
        text_commands.append(sentence)

        if current_num_data % save_file_change_freq == 0:
            # Save data
            file_name = data_dir + '/senEM_preprocessed_{}_{}_{}.pkl'.format(voice_config[1], voice_config[2], current_num_data)
            total_data_list = [img_idxs, pose_outputs, sentence_lens, sampled_audios, sample_rates, text_commands]
            with open(file_name, 'wb') as f:
                pickle.dump(total_data_list, f)
            print('{} saved'.format(file_name.split('/')[-1]))

            # Empty data list (due to memory problem)
            img_idxs = []
            pose_outputs = []
            sentence_lens = []
            sampled_audios = []
            sample_rates = []
            text_commands = []

    if extra_save_bool:
        # Save unsaved data
        file_name = data_dir + '/senEM_preprocessed_{}_{}_{}.pkl'.format(voice_config[1], voice_config[2], current_num_data)
        total_data_list = [img_idxs, pose_outputs, sentence_lens, sampled_audios, sample_rates, text_commands]
        with open(file_name, 'wb') as f:
            pickle.dump(total_data_list, f)
        print('{} saved'.format(file_name.split('/')[-1]))

def save_data_v2_0(relative_audio_directory_path, sr, relative_aligned_data_directory_path, sub_dirs, relative_save_data_directory_path):
    # Process and save data
    for sub_dir in sub_dirs:
        print('='*20)
        print('{} processing'.format(sub_dir))

        # Read audio data
        audio_files = read_audio_files(join(relative_audio_directory_path, sub_dir))
        total_num_data = len(audio_files)

        # Set empty data list
        current_num_data = 0
        save_file_change_freq = 5000
        data_dir = relative_save_data_directory_path
        sampled_audios = []
        sample_rates = []
        word_time_intervals = []

        # Check directiry to save data
        if not isdir(data_dir):
            makedirs(data_dir)

        # Check whether extra saving is needed for unsaved data
        extra_save_bool = False
        if not total_num_data % save_file_change_freq == 0:
            extra_save_bool = True
        
        # Saving file config
        voice_config = sub_dir.split('_')

        for audio_file in audio_files:
            current_num_data += 1
            print('Procressing: {}/{}'.format(current_num_data, total_num_data))

            # Sample audio
            wav_file_path = join(join(relative_audio_directory_path, sub_dir), audio_file)
            sampled_audio, sample_rate = librosa.load(wav_file_path, sr=sr)

            # Read textgrid data
            textgrid_file = audio_file.split('.')[0] + '.TextGrid'
            textgrid_file = join(join(relative_aligned_data_directory_path, sub_dir), textgrid_file)
            tg = textgrid.TextGrid.fromFile(textgrid_file)
            word_time_interval = []
            for i in range(len(tg[0])):
                word_time_interval.append([tg[0][i].mark, tg[0][i].minTime, tg[0][i].maxTime])

            # Append data to save
            sampled_audios.append(sampled_audio)
            sample_rates.append(sample_rate)
            word_time_intervals.append(word_time_interval)

            if current_num_data % save_file_change_freq == 0:
                # Save data
                file_name = data_dir + '/senEM_preprocessed_{}_{}_{}.pkl'.format(voice_config[1], voice_config[2], current_num_data)
                total_data_list = [sampled_audios, sample_rates, word_time_intervals]
                with open(file_name, 'wb') as f:
                    pickle.dump(total_data_list, f)
                print('{} saved'.format(file_name.split('/')[-1]))

                # Empty data list (due to memory problem)
                sampled_audios = []
                sample_rates = []
                word_time_intervals = []

        if extra_save_bool:
            # Save unsaved data
            file_name = data_dir + '/senEM_preprocessed_{}_{}_{}.pkl'.format(voice_config[1], voice_config[2], current_num_data)
            total_data_list = [sampled_audios, sample_rates, word_time_intervals]
            with open(file_name, 'wb') as f:
                pickle.dump(total_data_list, f)
            print('{} saved'.format(file_name.split('/')[-1]))

def save_data_v2_1(relative_data_directory_path, relative_save_data_directory_path):
    audio_length_equalize_and_save(relative_data_directory_path, relative_save_data_directory_path)

def save_data_v2_2(relative_data_directory_path, relative_script_directory_path, relative_save_data_directory_path, word_dic, word_dic_size, mel_feature_type):
    assert mel_feature_type in ['single', 'three']

    # Read data file
    data_files = [f for f in listdir(relative_data_directory_path) if isfile(join(relative_data_directory_path, f))]
    data_files.sort()

    # Set configuration
    n_fft = 2048
    hop_length = int(n_fft/8)
    win_length = int(n_fft/2)
    n_mels = 40
    sampled_audios_idx = 0
    sample_rates_idx = 1
    word_time_intervals_idx = 2
    num_total_data_count = 0
    mel_specs =[]
    word_labels = []

    # # Due to the delay time in google drive, '/data_v2.2' folder should be already prepared in google drive
    # # Check directiry to save data
    # if not isdir(relative_save_data_directory_path):
    #     makedirs(relative_save_data_directory_path)
    
    # Process needed data
    for i in range(len(data_files)):
        print('Processing {}/{}'.format(i+1, len(data_files)))
        data = load_single_data(relative_data_directory_path, data_files[i])
        num_data = len(data[0])
        for ii in range(num_data):
            num_total_data_count += 1

            if mel_feature_type == 'single':
                mel_spec = return_mel_spec_single_channel(sampled_audio=data[sampled_audios_idx][ii], sample_rate=data[sample_rates_idx][ii], \
                    n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
            elif mel_feature_type == 'three':
                mel_spec = return_mel_spec_three_channel(sampled_audio=data[sampled_audios_idx][ii], sample_rate=data[sample_rates_idx][ii], \
                    n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
            
            word_label = word_interval_continualize(len_mel_spec=mel_spec.shape[1], sample_rate=data[sample_rates_idx][ii], \
                hop_length=hop_length, word_time_interval=data[word_time_intervals_idx][ii], \
                word_dictionary=word_dic, word_dictionary_size=word_dic_size)

            mel_specs.append(mel_spec)
            word_labels.append(word_label)
        print('Finished processing {} data'.format(num_data))
    
    save_file_name = relative_save_data_directory_path + '/senEM_preprocessed.npz'
    np.savez_compressed(save_file_name, acoustic=mel_specs, linguistic=word_labels)

# # Save data_v1.0
# relative_save_data_directory_path = './data/data_v1.0'
# relative_audio_directory_path = './data/train_speech/Case_en-US_MALE'
# save_data_v1_0(relative_audio_directory_path, sample_sr=16000, relative_save_data_directory_path)
# relative_audio_directory_path = './data/train_speech/Case_en-US_FEMALE'
# save_data_v1_0(relative_audio_directory_path, sample_sr=16000, relative_save_data_directory_path)

# # Save data_v2.0
# relative_audio_directory_path = './data/train_speech'
# sr = 16000
# relative_aligned_data_directory_path = './data/aligned_speech_wo_train'
# sub_dirs = ['Case_en-US_MALE', 'Case_en-US_FEMALE']
# relative_save_data_directory_path = './data/data_v2.0'
# save_data_v2_0(relative_audio_directory_path, sr, relative_aligned_data_directory_path, sub_dirs, relative_save_data_directory_path)

# # Save data_v2.1
# relative_data_directory_path = './data/data_v2.0'
# relative_save_data_directory_path = './data/data_v2.1'
# save_data_v2_1(relative_data_directory_path, relative_save_data_directory_path)

# Save data_v2.2_single_channel
# relative_data_directory_path = '/content/drive/MyDrive/Speech2Pickup/data_v2.1'
# relative_save_data_directory_path = '/content/drive/MyDrive/Speech2Pickup/data_v2.2_single_channel_grouping'
# save_data_v2_2(relative_data_directory_path, relative_script_directory_path, relative_save_data_directory_path, word_dic, word_dic_size, mel_feature_type='single')

# # Save data_v2.2_three_channel
# relative_data_directory_path = '/content/drive/MyDrive/Speech2Pickup/data_v2.1'
# relative_save_data_directory_path = '/content/drive/MyDrive/Speech2Pickup/data_v2.2_three_channel_grouping'
# save_data_v2_2(relative_data_directory_path, relative_script_directory_path, relative_save_data_directory_path, word_dic, word_dic_size, mel_feature_type='three')