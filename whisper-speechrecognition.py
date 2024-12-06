import speech_recognition as sr
from faster_whisper import WhisperModel
import pyttsx3
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class AI_Assistant:
    def __init__(self):
        openai.api_type = "azure"
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")
        openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 200)
        self.tts_engine.setProperty('volume', 0.9)

        self.model = WhisperModel("/root/whisper", device="cpu", local_files_only=True)

    def record_audio(self):
        """Enregistre l'audio"""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Parlez maintenant...")
            try:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source)
                print("Enregistrement terminé.")
                with open("audio.wav", "wb") as file:
                    file.write(audio.get_wav_data())
                return "audio.wav"
            except sr.WaitTimeoutError:
                print("Temps d'attente dépassé, veuillez réessayer.")
                return None

    def transcribe_audio(self, filename):
        print("Transcription en cours...")
        try:
            segments, _ = self.model.transcribe(filename, language="fr")
            transcription = " ".join([segment.text.strip() for segment in segments])
            return transcription
        except Exception as e:
            print(f"Erreur pendant la transcription : {e}")
            return ""

    def generate_ai_response(self, transcript):
        print(f"\nUtilisateur : {transcript}")
        messages = [
            {"role": "system", "content": "You are a friendly assistant."},
            {"role": "user", "content": transcript},
        ]
        try:
            response = openai.ChatCompletion.create(
                deployment_id=self.azure_deployment_name,
                messages=messages,
                max_tokens=100
            )
            ai_response = response['choices'][0]['message']['content']
            self.generate_audio(ai_response)
            return ai_response
        except Exception as e:
            print(f"Erreur pendant la génération de réponse : {e}")
            return "Je n'ai pas pu répondre, désolé."

    def generate_audio(self, text):
        """pyttsx3"""
        print(f"\nAI Assistant : {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

ai_assistant = AI_Assistant()
ai_assistant.generate_audio("Bonjour, comment puis-je vous aider aujourd'hui ?")

while True:
    audio_file = ai_assistant.record_audio()
    if not audio_file:
        continue

    transcript = ai_assistant.transcribe_audio(audio_file)
    if not transcript:
        print("Réessayez.")
        continue

    ai_assistant.generate_ai_response(transcript)
