"""Synthesizes speech from the input string of text or ssml.

Note: ssml must be well-formed according to:
    https://www.w3.org/TR/speech-synthesis/
"""
from google.cloud import texttospeech
import os
from os import listdir
from os.path import isfile, join, isdir
import librosa
import numpy as np
import pickle

def text_to_speech(client, sentence, voice_config, voice_file_name_config, case_config):
    """
    voice_config[0]: english type (en-US/en-GB)  # en-US: USA, en-GB: England
    voice_config[1]: person sex (MALE/FEMALE)
    """
    # Set directory to save audio file and name audio file
    dir_name = 'data/train_speech/{}'.format(case_config)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    wav_file_name = 'output' + voice_file_name_config + '.wav'
    wav_file_name = join(dir_name, wav_file_name)

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=sentence)

    # Build the voice request, select the language code and the ssml
    if voice_config[1] == 'MALE':
        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_config[0], ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
    elif voice_config[1] == 'FEMALE':
        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_config[0], ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
    else:
        raise ValueError('Unavailable voice configuration')

    # Select the type of audio file (LINEAR16: .wav)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Write the response to the output file.
    # The response's audio_content is binary.
    with open(wav_file_name, "wb") as out:
        out.write(response.audio_content)

def text_to_speech_preprocess():
    # Read text data file list
    script_path = './data/train_script'
    files = [f for f in listdir(script_path) if isfile(join(script_path, f))]
    files.sort()

    # text-to-speech configuration
    client = texttospeech.TextToSpeechClient()  # Instantiates a text-to-speech client
    config = [['en-US', 'en-GB'], ['MALE', 'FEMALE']]
    """
    [idx_set description]
    (0,0): {Tone: en-US / Sex: MALE}
    (0,1): {Tone: en-US / Sex: FEMALE}
    (1,0): {Tone: en-GB / Sex: MALE}
    (1,1): {Tone: en-GB / Sex: FEMALE}
    """
    # idx_set = [(0,0), (0,1)]
    idx_set = [(0,1)]
    total_num_data = 0

    # Start preprocessing data
    for idx in idx_set:
        case_config = 'Case_{}_{}'.format(config[0][idx[0]], config[1][idx[1]])
        print('='*20)
        print(case_config)

        # Set empty data list
        num_data = 0
        for file_idx in files[36: ]:
            print('Now processing : %s/%s ... ' % (files.index(file_idx)+1, len(files)))
            curr_file = open(join(script_path, file_idx), 'r')
            curr_lines = curr_file.readlines()
            line_num = 0
            for line in curr_lines:
                num_data += 1
                line_num += 1
                words = line.split()
                sentence = ' '.join(words[2: ])
                voice_config = [config[0][idx[0]], config[1][idx[1]]]
                voice_file_name_config = '_{}_{}'.format(file_idx[0:4], line_num)

                text_to_speech(client, sentence, voice_config, voice_file_name_config, case_config)

        total_num_data += num_data
        print('Number of data for {}: {}'.format(case_config, num_data))
    print('Total number of data: {}'.format(total_num_data))

# # Changing text to speech
# text_to_speech_preprocess()