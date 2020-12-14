import io
import os

"""
For more detail check https://cloud.google.com/speech-to-text/docs/libraries
"""

# Imports the Google Cloud client library
from google.cloud import speech

# Instantiates a client
client = speech.SpeechClient()

def speech_to_text_for_model_test(audio_path='./data/random_speech/output.wav'):
    # The name of the audio file to transcribe
    file_name = audio_path

    # Loads the audio into memory
    with io.open(file_name, "rb") as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=24000,
        language_code="en-US",
    )

    # Detects speech in the audio file
    response = client.recognize(config=config, audio=audio)
    assert len(response.results) == 1

    for result in response.results:
        # print("Transcript: {}".format(result.alternatives[0].transcript))
        transcribed_sentence = result.alternatives[0].transcript
    return transcribed_sentence