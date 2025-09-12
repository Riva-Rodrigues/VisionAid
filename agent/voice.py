import pyttsx3
import speech_recognition as sr
import threading

class VoiceSystem:
    def __init__(self):
        self.setup_voice()
        self.lock = threading.Lock()  # Add a lock for thread safety

    def setup_voice(self):
        try:
            self.tts = pyttsx3.init()
            self.tts.setProperty('rate', 180)
            self.tts.setProperty('volume', 0.8)
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("✓ Voice systems initialized")
        except Exception as e:
            print(f"Error initializing voice systems: {e}")

    def speak(self, text):
        with self.lock:  # Ensure only one thread accesses the TTS engine at a time
            self.tts.say(text)
            self.tts.runAndWait()

    def listen(self, timeout=1, phrase_time_limit=5):
        try:
            with self.microphone as source:
                print("Listening for commands...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)  # Re-adjust for noise
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            command = self.recognizer.recognize_google(audio).lower()
            print(f"Command received: {command}")
            return command
        except sr.WaitTimeoutError:
            print("Listening timed out. No command detected.")
            return None
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            return None