import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pdb
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def word_check():
    w2v_path = './data/GoogleNews-vectors-negative300.bin.gz'
    w2v_model = load_w2v(w2v_path)
    dim_embed = w2v_model['woman'].shape[0]
    em_out = w2v_model['woman']
    print(em_out)
    print(dim_embed)

def total_data_count():
    # Read text data file list
    script_path = './data/train_script'
    files = [f for f in listdir(script_path) if isfile(join(script_path, f))]

    num_data = 0
    idx_set = [(0,0), (0,1)]
    print(files)
    print(len(files))
    # Start counting data
    for idx in idx_set:
        for file_idx in files:
            curr_file = open(join(script_path, file_idx), 'r')
            curr_lines = curr_file.readlines()
            for line in curr_lines:
                num_data += 1

                words = line.split()
                sentence = ' '.join(words[2: ])
                # if sentence == 'pick up the right purple block':
                    # print(file_idx)
    print('Total number of data: {}'.format(num_data))
    print('Number of data for each case: {}'.format(num_data/len(idx_set)))

def mel_spectogram(relative_file_directory, file_name):
    full_file_path = join(relative_file_directory, file_name)
    sr = 16000
    audio_data, audio_sr = librosa.load(full_file_path, sr=sr)
    n_fft = 2048
    hop_length = int(n_fft/8)
    win_length = int(n_fft/2)

    ##
    file_length = 45094
    padding = [0]*(file_length - len(audio_data))
    audio_data = np.append(audio_data, padding)
    ##
    
    audio_mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=audio_sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', n_mels=80)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(audio_mel_spec)
    # print(np.shape(np.array(S_dB)))
    # S_dB[S_dB<0] = 0  ## relu
    S_dB = np.tanh(S_dB)  ## tanh
    img = librosa.display.specshow(S_dB, hop_length=hop_length, x_axis='time',
                            y_axis='mel', sr=sr, cmap='viridis',
                            fmax=sr/2, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram: {}'.format(file_name))
    pic_name = file_name[:-4]
    plt.savefig(pic_name + '1')

def mfcc(relative_file_directory, file_name):
    full_file_path = join(relative_file_directory, file_name)
    sr = 16000
    audio_data, audio_sr = librosa.load(full_file_path, sr=sr)

    ##
    file_length = 45094
    padding = [0]*(file_length - len(audio_data))
    audio_data = np.append(audio_data, padding)
    ##

    mfccs = librosa.feature.mfcc(y=audio_data, sr=audio_sr, n_mfcc=13)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', sr=audio_sr,
                            fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC: {}'.format(file_name))
    pic_name = file_name[:-4]
    plt.savefig(pic_name+'_mfcc')

def cosine_similarity(x, y):
    x_absolute = np.sqrt(np.matmul(x,x))
    y_absolute = np.sqrt(np.matmul(y,y))
    return np.matmul(x, y)/(x_absolute*y_absolute)

def mfcc_similairty(relative_file_directory1, file_name1, relative_file_directory2, file_name2):
    full_file_path1 = []
    full_file_path2 = []
    for i in range(len(file_name1)):
        full_file_path1.append(join(relative_file_directory1, file_name1[i]))
        full_file_path2.append(join(relative_file_directory2, file_name2[i]))
    sr = 16000
    n_mfcc = 40
    mfcc_sim = np.zeros((len(full_file_path1),len(full_file_path2)))

    for i in range(len(full_file_path1)):
        for j in range(len(full_file_path2)):
            audio_data1, audio_sr1 = librosa.load(full_file_path1[i], sr=sr)
            audio_data2, audio_sr2 = librosa.load(full_file_path2[j], sr=sr)
            print(len(audio_data1))
            print(len(audio_data2))

            ##
            file_length = 45094
            padding1 = [0]*(file_length - len(audio_data1))
            padding2 = [0]*(file_length - len(audio_data2))
            audio_data1 = np.append(audio_data1, padding1)
            audio_data2 = np.append(audio_data2, padding2)
            ##

            mfccs1 = librosa.feature.mfcc(y=audio_data1, sr=audio_sr1, n_mfcc=n_mfcc)
            mfccs2 = librosa.feature.mfcc(y=audio_data2, sr=audio_sr2, n_mfcc=n_mfcc)
            # aver_mfccs1 = np.mean(mfccs1, axis=0)
            # aver_mfccs2 = np.mean(mfccs2, axis=0)
            # mfcc_sim[i,j] = cosine_similarity(aver_mfccs1, aver_mfccs2)
            flat_mfccs1 = mfccs1.flatten()
            flat_mfccs2 = mfccs2.flatten()
            mfcc_sim[i,j] = cosine_similarity(flat_mfccs1, flat_mfccs2)
    print(mfcc_sim)


    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.plot(aver_mfccs1, label='1')
    # ax.plot(aver_mfccs2, label='2')
    # plt.legend()
    # plt.show()

# mel

# # MALE
relative_file_directory = './data/train_speech/Case_en-US_MALE'
file_name = 'output_0320_28.wav'
mel_spectogram(relative_file_directory, file_name)

# relative_file_directory = './data/train_speech/Case_en-US_MALE'
# file_name = 'output_0320_25.wav'
# mel_spectogram(relative_file_directory, file_name)

# relative_file_directory = './data/train_speech/Case_en-US_MALE'
# file_name = 'output_0320_26.wav'
# mel_spectogram(relative_file_directory, file_name)

# # FEMALE
# relative_file_directory = './data/train_speech/Case_en-US_FEMALE'
# file_name = 'output_1000_1.wav'
# mel_spectogram(relative_file_directory, file_name)

# relative_file_directory = './data/train_speech/Case_en-US_FEMALE'
# file_name = 'output_1000_2.wav'
# mel_spectogram(relative_file_directory, file_name)

# relative_file_directory = './data/train_speech/Case_en-US_FEMALE'
# file_name = 'output_1000_3.wav'
# mel_spectogram(relative_file_directory, file_name)

# mfcc

# # MALE
# relative_file_directory = './data/train_speech/Case_en-US_MALE'
# file_name = 'output_0320_24.wav'
# mfcc(relative_file_directory, file_name)

# relative_file_directory = './data/train_speech/Case_en-US_MALE'
# file_name = 'output_0320_25.wav'
# mfcc(relative_file_directory, file_name)

# relative_file_directory = './data/train_speech/Case_en-US_MALE'
# file_name = 'output_0320_26.wav'
# mfcc(relative_file_directory, file_name)

# # FEMALE
# relative_file_directory = './data/train_speech/Case_en-US_FEMALE'
# file_name = 'output_1000_1.wav'
# mfcc(relative_file_directory, file_name)

# relative_file_directory = './data/train_speech/Case_en-US_FEMALE'
# file_name = 'output_1000_2.wav'
# mfcc(relative_file_directory, file_name)

# relative_file_directory = './data/train_speech/Case_en-US_FEMALE'
# file_name = 'output_1000_3.wav'
# mfcc(relative_file_directory, file_name)

# relative_file_directory1 = './data/train_speech/Case_en-US_MALE'
# file_name1 = ['output_0320_24.wav', 'output_0320_25.wav', 'output_0320_26.wav']
# relative_file_directory2 = './data/train_speech/Case_en-US_FEMALE'
# file_name2 = ['output_1000_1.wav', 'output_1000_2.wav', 'output_1000_3.wav']
# mfcc_similairty(relative_file_directory1, file_name1, relative_file_directory2, file_name2)

# # Mel spectogram practice
# mel_spectogram(relative_file_directory1, file_name1[0])

# image_file = 'output_0320_24.png'
# image = mpimg.imread(image_file)
# print(image.shape)
# print(image[:,100,2])
    