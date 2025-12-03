import os
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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
    "DR_ARJUN": "onyx",   # mentor voice
    "RIYA": "shimmer",        # student voice
}

TTS_MODEL = "gpt-4o-mini-tts"


# ---------- Helper: call OpenAI TTS for one line ----------
from io import BytesIO

def tts_line_to_mp3_bytes(text: str, voice: str) -> bytes:
    """
    Call OpenAI TTS and return MP3 bytes for the given text+voice.
    Uses streaming API (no 'format' argument needed).
    """
    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=voice,
        input=text,
    ) as response:
        buf = BytesIO()
        for chunk in response.iter_bytes():
            buf.write(chunk)
        return buf.getvalue()

# ---------- Core render logic ----------

def render_conversation_bytes(req: RenderRequest) -> bytes:
    """
    For all segments in the request, call TTS and concatenate MP3 bytes.
    NOTE: This is a simple concatenation of MP3 data. Most players handle this fine.
    """
    all_bytes = b""

    for seg in req.segments:
        text = seg.text.strip()
        if not text:
            continue

        speaker = seg.speaker.upper()
        voice = VOICE_MAP.get(speaker, "alloy")  # default to Dr Arjun voice

        audio_bytes = tts_line_to_mp3_bytes(text, voice)
        all_bytes += audio_bytes

    return all_bytes


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

    return {
        "status": "ok",
        "audio_url": audio_url,
        "file_name": filename,
    }
