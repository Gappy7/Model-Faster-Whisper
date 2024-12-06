import sounddevice as sd
import numpy as np
import wave
from faster_whisper import WhisperModel
import pyttsx3
import openai
from dotenv import load_dotenv
import os
import huggingface_hub

load_dotenv()

class AI_Assistant:
    def __init__(self):
        openai.api_type = "azure"
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")
        openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 300)
        self.tts_engine.setProperty('volume', 0.9)
        model_id = "openai/whisper-small"
        model_path = huggingface_hub.snapshot_download(model_id)
        self.model = WhisperModel(model_path, device="cpu", local_files_only=True)
      
    def record_audio(self, filename="audio.wav", duration=10, samplerate=16000):
        """
        Enregistre l'audio pendant une durée fixe.
        :param filename: Nom du fichier de sortie
        :param duration: Durée d'enregistrement en secondes
        :param samplerate: Taux d'échantillonnage
        """
        print(f"Enregistrement en cours pour {duration} secondes...")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
        sd.wait()
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        print("Enregistrement terminé.")
      
    def transcribe_audio(self, filename="audio.wav"):
        """Transcrit l'audio en texte avec Faster Whisper."""
        print("Transcription en cours...")
        segments, _ = self.model.transcribe(filename, language="fr")
        transcription = " ".join([segment.text.strip() for segment in segments])
        return transcription
      
    def generate_ai_response(self, transcript):
        print(f"\nPatient: {transcript}", end="\r\n")
        self.full_transcript = [
            {"role": "system", "content": "You are a receptionist at a dental clinic. Be resourceful and efficient."},
            {"role": "user", "content": transcript},
        ]
        response = openai.ChatCompletion.create(
            deployment_id=self.azure_deployment_name, 
            messages=self.full_transcript,
            max_tokens=1000
        )
        ai_response = response['choices'][0]['message']['content']
        self.generate_audio(ai_response)
      
    def generate_audio(self, text):
        print(f"\nAI Receptionist: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

greeting = "Hello, how can I assist you today?"
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)
while True:
    ai_assistant.record_audio(duration=10)
    transcript_text = ai_assistant.transcribe_audio()
    if transcript_text:
        ai_assistant.generate_ai_response(transcript_text)
    else:
        print("Aucun texte détecté, veuillez réessayer.")
