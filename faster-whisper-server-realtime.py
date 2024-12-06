import requests
import pyaudio
import wave
import time
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

SERVER_URL = "http://localhost:5000/transcribe"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = -40
MIN_SILENCE_LEN = 1000

def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        files = {'file': f}
        response = requests.post(SERVER_URL, files=files)
    if response.status_code == 200:
        print("Transcription:", response.json()['transcription'])
    else:
        print("Erreur:", response.status_code, response.text)

def record_and_transcribe():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Enregistrement... (appuyez sur Ctrl+C pour arrêter)")
    frames = []
    silence_counter = 0
    silence_threshold = int(RATE / CHUNK * MIN_SILENCE_LEN / 1000)
    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
            audio_segment = AudioSegment(
                data=b''.join(frames),
                sample_width=audio.get_sample_size(FORMAT),
                frame_rate=RATE,
                channels=CHANNELS
            )
            non_silent_ranges = detect_nonsilent(
                audio_segment,
                min_silence_len=MIN_SILENCE_LEN,
                silence_thresh=SILENCE_THRESHOLD
            )
            if not non_silent_ranges:
                silence_counter += 1
            else:
                silence_counter = 0
            if silence_counter > silence_threshold:
                print("Silence détecté, envoi de l'audio pour transcription...")
                wf = wave.open("temp.wav", 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                transcribe_audio("temp.wav")
                frames = []
                silence_counter = 0
    except KeyboardInterrupt:
        print("Enregistrement arrêté")
    stream.stop_stream()
    stream.close()
    audio.terminate()
record_and_transcribe()
