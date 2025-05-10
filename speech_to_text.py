import speech_recognition as sr
from pydub import AudioSegment
import os

def convert_voice_to_text(audio_path):
    try:
        # Convert .opus to .wav using pydub (needs ffmpeg)
        audio = AudioSegment.from_file(audio_path)  # auto-detect format

        temp_wav_path = "converted_audio.wav"
        audio.export(temp_wav_path, format="wav")

        # Transcribe with SpeechRecognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                return text
            except sr.UnknownValueError:
                return "❌ Could not understand the audio."
            except sr.RequestError:
                return "⚠️ API error or connection issue."
    except Exception as e:
        return f"Error: {str(e)}"
