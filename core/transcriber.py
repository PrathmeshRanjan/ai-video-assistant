import whisper
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None

def load_model():
    global _model
    if _model is None:
        print("Loading model...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully")

    return _model    

