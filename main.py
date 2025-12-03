import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydub import AudioSegment
from openai import OpenAI


# ---------- Setup ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NEET Audio Conversation Renderer")

# Serve files in /audio as static URLs
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


# ---------- Data models ----------

class Segment(BaseModel):
    speaker: str  # e.g. "DR_ARJUN" or "RIYA"
    text: str     # the dialogue line


class RenderRequest(BaseModel):
    topic_id: str
    segments: List[Segment]


# Map speakers to OpenAI TTS voices
VOICE_MAP = {
    "DR_ARJUN": "onyx",   # deeper, mentor voice
    "RIYA": "nova",       # lighter, student voice
}

TTS_MODEL = "gpt-4o-mini-tts"


# ---------- Helper: call OpenAI TTS for one line ----------

def tts_line_to_mp3_bytes(text: str, voice: str, speed: float) -> bytes:
    """
    Call OpenAI TTS for the given text+voice, with speed control.
    Returns raw MP3 bytes.
    """
    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=voice,
        input=text,
        speed=speed
    ) as response:
        buf = BytesIO()
        for chunk in response.iter_bytes():
            buf.write(chunk)
        return buf.getvalue()


# ---------- Core render logic ----------

def render_conversation_bytes(req: RenderRequest) -> bytes:
    """
    Generate full conversation MP3 with:
    - two distinct voices
    - micro-pauses between lines (0.3–0.5s)
    - fade in / fade out
    """
    # Start with 0.5 sec of silence at the beginning
    final_audio = AudioSegment.silent(duration=500)

    for seg in req.segments:
        text = seg.text.strip()
        if not text:
            continue

        speaker = seg.speaker.upper()
        voice = VOICE_MAP.get(speaker, "onyx")  # default to mentor voice

        # Speed tuning per speaker
        if speaker == "DR_ARJUN":
            speed = 0.95   # slightly slower, more serious
        elif speaker == "RIYA":
            speed = 1.05   # slightly faster, more energetic
        else:
            speed = 1.0

        # Get TTS audio as MP3 bytes
        audio_bytes = tts_line_to_mp3_bytes(text, voice, speed)

        # Load bytes into an AudioSegment
        clip = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")

        # Simple "time stretch" via frame_rate change for speed adjustment
        # (optional and subtle; comment out if not desired)
        # new_frame_rate = int(clip.frame_rate * speed)
        # clip = clip._spawn(clip.raw_data, overrides={"frame_rate": new_frame_rate}).set_frame_rate(clip.frame_rate)

        # Emotional micro-pause logic
        emotional_words = ["honestly", "ahh", "umm", "wait", "sir", "hmm"]
        if any(w in text.lower() for w in emotional_words):
            pause = AudioSegment.silent(duration=500)  # 0.5s pause
        else:
            pause = AudioSegment.silent(duration=300)  # 0.3s pause

        final_audio += clip + pause

    # Fade in & fade out (no background music)
    final_audio = final_audio.fade_in(1200).fade_out(1500)

    # Export to bytes
    out_buf = BytesIO()
    final_audio.export(out_buf, format="mp3")
    return out_buf.getvalue()


# ---------- API endpoint ----------

@app.post("/render-conversation")
def render_conversation(req: RenderRequest, request: Request):
    if not req.segments:
        raise HTTPException(status_code=400, detail="No segments provided")

    try:
        audio_bytes = render_conversation_bytes(req)
    except Exception as e:
        print("Render error:", e)
        raise HTTPException(status_code=500, detail=f"Render failed: {e}")

    # Create a unique filename
    safe_topic = req.topic_id.lower().replace(" ", "-")
    unique = uuid.uuid4().hex[:8]
    filename = f"{safe_topic}_{unique}.mp3"
    out_path = AUDIO_DIR / filename

    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    # Build a full URL to the file using request.base_url
    base_url = str(request.base_url).rstrip("/")
    audio_url = f"{base_url}/audio/{filename}"

    # Get duration (ms) for metadata (optional)
    try:
        audio_segment = AudioSegment.from_mp3(out_path)
        duration_ms = len(audio_segment)
    except Exception:
        duration_ms = None

    return {
        "status": "ok",
        "audio_url": audio_url,
        "file_name": filename,
        "duration_ms": duration_ms,
            }
