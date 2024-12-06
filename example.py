from openai import OpenAI

client = OpenAI(api_key="cant-be-empty", base_url="http://localhost:48210/v1/")

audio_file = open("/root/whisper/Model-Faster-Whisper/output.wav", "rb")
transcript = client.audio.transcriptions.create(
    model="deepdml/faster-whisper-large-v3-turbo-ct2", file=audio_file
)
print(transcript.text)
