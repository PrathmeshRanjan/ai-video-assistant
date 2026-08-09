import whisper
import os
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None
_sarvam_client = None

def load_model():
    global _model
    if _model is None:
        print("Loading Whisper model...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully")
    return _model

def _get_sarvam_client():
    global _sarvam_client
    if _sarvam_client is None:
        _sarvam_client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
    return _sarvam_client

def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    model = load_model()
    task = 'translate' if translate else 'transcribe'
    result = model.transcribe(chunk_path, task=task)
    return result['text']

def transcribe_all(chunks: list, translate: bool = False) -> str:
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}")
        text = transcribe_chunk(chunk, translate=translate)
        full_transcript += text + " "
    print("Transcription completed")
    return full_transcript.strip()

def transcribe_chunk_sarvam(chunk_path: str) -> str:
    # mode="translate" converts Hindi speech directly to English text
    client = _get_sarvam_client()
    try:
        with open(chunk_path, "rb") as f:
            response = client.speech_to_text.transcribe(
                file=f,
                model="saaras:v3",
                mode="translate",
            )
        return response.transcript
    except ApiError as e:
        print(f"Sarvam API error {e.status_code}: {e.body}")
        raise

def transcribe_all_sarvam(chunks: list) -> str:
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        print(f"Translating chunk {i+1}/{len(chunks)} (Hindi → English) via Sarvam AI...")
        text = transcribe_chunk_sarvam(chunk)
        full_transcript += text + " "
    print("Transcription completed")
    return full_transcript.strip()