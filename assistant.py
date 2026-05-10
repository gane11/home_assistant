import subprocess
from openai import OpenAI

client = OpenAI()

while True:
    text = input("Say command text for now: ")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"You are my home assistant. Keep answer short. User said: {text}",
    )

    answer = response.output_text
    print("Assistant:", answer)

    subprocess.run(["espeak", answer])