import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('voice', 'spanish')  # o 'english' según el sistema

def speak(text):
    print(f"[AUDIO] {text}")
    engine.say(text)
    engine.runAndWait()
