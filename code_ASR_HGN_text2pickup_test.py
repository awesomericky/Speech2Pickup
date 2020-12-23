import numpy as np
from os.path import join
import os
from text_to_speech import text_to_speech_for_model_test
from speech_to_text import speech_to_text_for_model_test
from process_data import add_noise
import time


########################
## Test without noise ##
########################

def test_wo_noise():
    script_dir_path = './data/train_script'
    npzfile = np.load('./data/divide_img_idx.npz')
    test_img_idx = npzfile['arr_1']
    voice_config1 = {'gender': 'MALE', 'accent': 'en-US'}
    voice_config2 = {'gender': 'FEMALE', 'accent': 'en-US'}
    voice_configs = [voice_config1, voice_config2]

    img_idxs = []
    pos_outputs = []
    real_text_inputs = []
    STT_text_inputs = []
    total_data_count = 0
    STT_correct_count = 0
    STT_error_data = []

    for idx in test_img_idx:
        script_file = '%04d.txt' % idx
        
        curr_file = open(join(script_dir_path, script_file), 'r')
        curr_file_lines = curr_file.readlines()
        for i in range(len(curr_file_lines)):
            pos_output = np.asarray(curr_file_lines[i].split()[ :2], dtype=np.float32)
            words = curr_file_lines[i].split()[2: ]
            real_text_input = ' '.join(words)

            for voice_config in voice_configs:
                total_data_count += 1
                print('Processing {} data'.format(total_data_count))

                wav_file_path = './data/random_speech/test_data/{}.wav'.format(total_data_count)
                _ = text_to_speech_for_model_test(sentence=real_text_input, voice_config=voice_config, file_path=wav_file_path)
                STT_text_input = speech_to_text_for_model_test(audio_path=wav_file_path)
                STT_text_input = STT_text_input.lower()

                img_idxs.append(idx)
                pos_outputs.append(pos_output)
                real_text_inputs.append(real_text_input)
                STT_text_inputs.append(STT_text_input)

                if real_text_input != STT_text_input:
                    STT_error_data.append((real_text_input, STT_text_input))
                else:
                    STT_correct_count += 1

    accuracy = (STT_correct_count/total_data_count)*100
    print('Total number of data: {}'.format(total_data_count))
    print('Total number of correct STT data: {}'.format(STT_correct_count))
    print('STT accuracy: {}%'.format(accuracy))
    print(STT_error_data)
    np.savez_compressed('./data/ASR_text2pickup_evaluate', img_idxs=np.asarray(img_idxs), pos_outputs=np.asarray(pos_outputs), real_text_inputs=real_text_inputs, STT_text_inputs=STT_text_inputs)







################################
## Check model inference time ##
################################

total_num_data = len(os.listdir('./data/random_speech/test_data'))
times = []

for i in range(total_num_data):
    print('Processing {}/{}'.format(i+1, total_num_data))
    file_name = './data/random_speech/test_data/{}.wav'.format(i+1)
    start = time.time()
    STT_text_input = speech_to_text_for_model_test(audio_path=file_name)
    STT_text_input = STT_text_input.lower()
    end = time.time()

    inter_time = end - start
    times.append(inter_time)

times = np.asarray(times)
np.savez_compressed('./data/ASR_STT_time_evaluate', times=times)






            
#####################
## Test with noise ##
#####################

def test_w_noise():
    script_dir_path = './data/train_script'
    npzfile = np.load('./data/divide_img_idx.npz')
    test_img_idx = npzfile['arr_1']

    img_idxs = []
    pos_outputs = []
    real_text_inputs = []
    STT_text_inputs = []
    total_data_count = 0
    STT_correct_count = 0
    STT_error_data = []

    for idx in test_img_idx:
        script_file = '%04d.txt' % idx
        
        curr_file = open(join(script_dir_path, script_file), 'r')
        curr_file_lines = curr_file.readlines()
        for i in range(len(curr_file_lines)):
            total_data_count += 1
            print('Processing {} data'.format(total_data_count))

            pos_output = np.asarray(curr_file_lines[i].split()[ :2], dtype=np.float32)
            words = curr_file_lines[i].split()[2: ]
            real_text_input = ' '.join(words)

            wav_file_path = './data/random_speech/test_data/{}.wav'.format(total_data_count)
            noise_wav_file_path = './data/random_speech/test_data_w_noise/{}.wav'.format(total_data_count)
            _ = add_noise(audio_file=wav_file_path, noise_file='./data/random_speech/noise.wav')
            STT_text_input = speech_to_text_for_model_test(audio_path=noise_wav_file_path)
            STT_text_input = STT_text_input.lower()

            img_idxs.append(idx)
            pos_outputs.append(pos_output)
            real_text_inputs.append(real_text_input)
            STT_text_inputs.append(STT_text_input)
            import pdb; pdb.set_trace()

            if real_text_input != STT_text_input:
                STT_error_data.append((real_text_input, STT_text_input))
            else:
                STT_correct_count += 1

    accuracy = (STT_correct_count/total_data_count)*100
    print('Total number of data: {}'.format(total_data_count))
    print('Total number of correct STT data: {}'.format(STT_correct_count))
    print('STT accuracy: {}%'.format(accuracy))
    print(STT_error_data)
    np.savez_compressed('./data/ASR_text2pickup_evaluate_w_noise', img_idxs=np.asarray(img_idxs), pos_outputs=np.asarray(pos_outputs), real_text_inputs=real_text_inputs, STT_text_inputs=STT_text_inputs)




