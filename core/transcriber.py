import whisper
import os
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# BCP-47 language codes Sarvam supports for translation to English
_SARVAM_SUPPORTED_LANGS = {"hi", "bn", "kn", "ml", "mr", "or", "pa", "ta", "te", "gu"}

_model = None
_sarvam_client = None

def load_model():
    global _model
    if _model is None:
        print("Loading Whisper model...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully")
    return _model

def detect_language(audio_path: str) -> str:
    # runs Whisper's language detection on the first 30s of audio; returns ISO 639-1 code e.g. "hi", "en"
    model = load_model()
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected = max(probs, key=probs.get)
    print(f"Detected language: {detected}")
    return detected

def transcribe_auto(chunks: list) -> str:
    # detects language from the first chunk and routes to Sarvam (Indian langs) or Whisper (everything else)
    lang = detect_language(chunks[0])
    if lang in _SARVAM_SUPPORTED_LANGS:
        print(f"Routing to Sarvam AI for '{lang}' → English translation")
        return transcribe_all_sarvam(chunks)
    print(f"Routing to Whisper for '{lang}' transcription")
    return transcribe_all(chunks)

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