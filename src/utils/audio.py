import threading
from playsound import playsound
import streamlit as st

def play_alarm_sound(sound_path):
    """Play the alarm sound safely."""
    try:
        playsound(sound_path)
    except Exception as e:
        print(f"Audio alert failed: {e}")
        # Note: We can't easily show st.warning from a thread without script context context
