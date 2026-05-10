import subprocess

import sounddevice as sd
import soundfile as sf
from openai import OpenAI

client = OpenAI()

AUDIO_FILE = 'command.wav'
SAMPLE_RATE = 16000
RECORD_SECONDS = 5


def record_audio(seconds=RECORD_SECONDS):
    print('Listening...')

    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16',
    )

    sd.wait()
    sf.write(AUDIO_FILE, audio, SAMPLE_RATE)

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
    input('Press Enter, then ask your question...')

    record_audio()

    text = transcribe_audio()
    print('You:', text)

    answer = ask_llm(text)
    print('Assistant:', answer)

    speak(answer)