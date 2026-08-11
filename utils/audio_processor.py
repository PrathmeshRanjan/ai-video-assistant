import os
from pathlib import Path

import yt_dlp          # downloads audio from YouTube and other platforms
from pydub import AudioSegment  # audio manipulation: format conversion and chunking
from pytubefix import YouTube

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

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

    try:
        return download_youtube_audio_with_pytubefix(url)
    except Exception as fallback_error:
        raise RuntimeError(
            "Could not download audio from this YouTube link. "
            f"yt-dlp error: {last_error}. pytubefix fallback error: {fallback_error}"
        ) from fallback_error

def download_youtube_audio_with_pytubefix(url: str) -> str:
    yt = YouTube(url, "ANDROID_VR")
    stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
    if stream is None:
        raise RuntimeError("No audio-only streams were found.")

    downloaded_path = stream.download(
        output_path=DOWNLOAD_DIR,
        filename=f"{yt.video_id}-pytubefix",
        skip_existing=False,
    )
    return convert_to_wav(downloaded_path)

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
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source, cookie_path=cookie_path)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
