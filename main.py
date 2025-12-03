import os
import uuid
from io import BytesIO
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
    speaker: str  # "DR_ARJUN" or "RIYA"
    text: str     # one dialogue line


class RenderRequest(BaseModel):
    topic_id: str
    segments: List[Segment]


# ---------- Voice config ----------

VOICE_MAP = {
    "DR_ARJUN": "onyx",   # deeper mentor-style voice
    "RIYA": "nova",       # lighter student-style voice
}

TTS_MODEL = "gpt-4o-mini-tts"


# ---------- TTS helper ----------

def tts_line_to_mp3_bytes(text: str, voice: str, speed: float) -> bytes:
    """
    Call OpenAI TTS for the given text+voice, with speed control.
    Returns MP3 bytes.
    """
    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=voice,
        input=text,
        speed=speed,
    ) as response:
        buf = BytesIO()
        for chunk in response.iter_bytes():
            buf.write(chunk)
        return buf.getvalue()


# ---------- Core render logic (simple MP3 concatenation) ----------

def render_conversation_bytes(req: RenderRequest) -> bytes:
    """
    Generate full conversation MP3 bytes by:
    - looping over segments
    - choosing voice + speed by speaker
    - concatenating all MP3 chunks

    Note: This is a simple byte-level concatenation. Most players
    handle this fine for sequential TTS segments.
    """
    all_bytes = b""

    for seg in req.segments:
        text = seg.text.strip()
        if not text:
            continue

        speaker = seg.speaker.upper()
        voice = VOICE_MAP.get(speaker, "onyx")  # default to mentor voice

        # Speed tuning per speaker
        if speaker == "DR_ARJUN":
            speed = 0.95   # slightly slower
        elif speaker == "RIYA":
            speed = 1.05   # slightly faster
        else:
            speed = 1.0

        audio_bytes = tts_line_to_mp3_bytes(text, voice, speed)

        # Append to running MP3 byte stream
        all_bytes += audio_bytes

        # (Optional) Tiny artificial gap: append a very small silence MP3 here
        # if you later add a pre-generated "silence_300ms.mp3" file and read its bytes.

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

    # Build URL
    base_url = str(request.base_url).rstrip("/")
    audio_url = f"{base_url}/audio/{filename}"

    return {
        "status": "ok",
        "audio_url": audio_url,
        "file_name": filename,
    }
