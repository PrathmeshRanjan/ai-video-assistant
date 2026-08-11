import os
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import yt_dlp          # downloads audio from YouTube and other platforms
from pydub import AudioSegment  # audio manipulation: format conversion and chunking

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")

def _youtube_opts(output_path: str, cookie_path: str | None, format_selector: str) -> dict:
    opts = {
        "format": format_selector,
        "outtmpl": output_path,
        "noplaylist": True,
        "cachedir": False,
        "js_runtimes": {"node": {}},
        "remote_components": {"ejs:github"},
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    if cookie_path:
        opts["cookiefile"] = cookie_path
    return opts

def _youtube_metadata_opts(cookie_path: str | None = None) -> dict:
    opts = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "cachedir": False,
        "js_runtimes": {"node": {}},
        "remote_components": {"ejs:github"},
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        },
    }
    if cookie_path:
        opts["cookiefile"] = cookie_path
    return opts

def _select_caption(info: dict) -> dict | None:
    subtitles = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}
    language_order = ["en", "en-US", "en-orig"]

    for captions in [subtitles, automatic]:
        for lang in language_order:
            if lang in captions:
                return _select_caption_format(captions[lang])
        for lang, entries in captions.items():
            if lang.startswith("en"):
                return _select_caption_format(entries)
        for entries in captions.values():
            return _select_caption_format(entries)
    return None

def _select_caption_format(entries: list[dict]) -> dict | None:
    for preferred_ext in ["json3", "srt", "vtt"]:
        for entry in entries:
            if entry.get("ext") == preferred_ext and entry.get("url"):
                return entry
    return next((entry for entry in entries if entry.get("url")), None)

def _download_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")

def _json3_to_text(raw: str) -> str:
    payload = json.loads(raw)
    lines = []
    for event in payload.get("events", []):
        text = "".join(seg.get("utf8", "") for seg in event.get("segs", []))
        text = " ".join(text.split())
        if text:
            lines.append(text)
    return " ".join(lines)

def _caption_text_to_plain_text(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.isdigit() or "-->" in line or line.startswith("WEBVTT"):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            lines.append(line)
    return " ".join(lines)

def fetch_youtube_transcript(url: str, cookie_path: str | None = None) -> str | None:
    try:
        with yt_dlp.YoutubeDL(_youtube_metadata_opts(cookie_path)) as ydl:
            info = ydl.extract_info(url, download=False)

        caption = _select_caption(info)
        if not caption:
            return None

        raw = _download_text(caption["url"])
        transcript = _json3_to_text(raw) if caption.get("ext") == "json3" else _caption_text_to_plain_text(raw)
        return transcript.strip() or None
    except (yt_dlp.utils.DownloadError, urllib.error.URLError, json.JSONDecodeError):
        return None

def download_youtube_audio(url: str, cookie_path: str | None = None) -> str:
    # yt-dlp template: fills in video title and original container extension at runtime
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    format_selectors = [
        "bestaudio/best[acodec!=none]/best",
        "bestaudio[ext=m4a]/bestaudio[ext=webm]/best[acodec!=none]/best",
        "18/best",
    ]

    last_error = None
    for format_selector in format_selectors:
        ydl_opts = _youtube_opts(output_path, cookie_path, format_selector)
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return str(Path(ydl.prepare_filename(info)).with_suffix(".wav"))
        except yt_dlp.utils.DownloadError as exc:
            last_error = exc

    raise RuntimeError(
        "Could not download audio from this YouTube link. Last yt-dlp error: "
        f"{last_error}"
    ) from last_error

def convert_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    # mono 16 kHz is the standard format expected by most speech-to-text models
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path

def chunk_audio(wav_path: str, chunk_seconds: int = 600) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_seconds * 1000  # pydub works in milliseconds

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks

def process_input(source: str, cookie_path: str | None = None) -> list:
    # route to downloader or local converter based on whether source is a URL
    if is_url(source):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source, cookie_path=cookie_path)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
