import subprocess

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from openwakeword.model import Model

client = OpenAI()

AUDIO_FILE = 'command.wav'

DEVICE_SAMPLE_RATE = 48000  # keep this if 48000 worked for your speakerphone
WAKE_SAMPLE_RATE = 16000
RECORD_SECONDS = 5
WAKE_THRESHOLD = 0.5

wake_model = Model()
print(wake_model.models.keys())


def resample_to_16k(audio_chunk):
    audio_chunk = np.squeeze(audio_chunk).astype(np.float32)

    original_length = len(audio_chunk)
    target_length = int(original_length * WAKE_SAMPLE_RATE / DEVICE_SAMPLE_RATE)

    original_indexes = np.linspace(0, original_length - 1, original_length)
    target_indexes = np.linspace(0, original_length - 1, target_length)

    resampled = np.interp(target_indexes, original_indexes, audio_chunk)

    return resampled.astype(np.int16)


def wait_for_wake_word():
    print('Listening for wake word...')

    chunk_size = int(DEVICE_SAMPLE_RATE * 0.08)

    with sd.InputStream(
        channels=1,
        samplerate=DEVICE_SAMPLE_RATE,
        dtype='int16',
        blocksize=chunk_size,
    ) as stream:
        while True:
            audio_chunk, _ = stream.read(chunk_size)
            wake_audio = resample_to_16k(audio_chunk)

            prediction = wake_model.predict(wake_audio)

            if not prediction:
                continue

            best_wake_word = max(prediction, key=prediction.get)
            best_score = prediction[best_wake_word]

            if best_score > 0.1:
                print(f'{best_wake_word}: {best_score:.2f}')

            if best_score > WAKE_THRESHOLD:
                print(f'Wake word detected: {best_wake_word}')
                subprocess.run(['espeak', 'yes'])
                return


def record_audio(seconds=RECORD_SECONDS):
    print('Listening for command...')

    audio = sd.rec(
        int(seconds * DEVICE_SAMPLE_RATE),
        samplerate=DEVICE_SAMPLE_RATE,
        channels=1,
        dtype='int16',
    )

    sd.wait()
    sf.write(AUDIO_FILE, audio, DEVICE_SAMPLE_RATE)

    print('Done recording.')


def transcribe_audio():
    with open(AUDIO_FILE, 'rb') as audio_file:
        transcription = client.audio.transcriptions.create(
            model='whisper-1',
            file=audio_file,
        )

    return transcription.text


def ask_llm(text):
    response = client.responses.create(
        model='gpt-4.1-mini',
        input=f'You are my home assistant. Keep answers short. User said: {text}',
    )

    return response.output_text


def speak(text):
    subprocess.run(['espeak', text])


while True:
    wait_for_wake_word()

    record_audio()

    text = transcribe_audio()
    print('You:', text)

    answer = ask_llm(text)
    print('Assistant:', answer)

    speak(answer)