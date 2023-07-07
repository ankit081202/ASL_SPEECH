import gtts
from playsound import playsound
import os
def play(text):
    sound = gtts.gTTS(text,lang="en")
    sound.save("test_sound.mp3")
    playsound("test_sound.mp3")
    os.remove("test_sound.mp3")
