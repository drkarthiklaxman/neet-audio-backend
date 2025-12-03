import os
from io import BytesIO
from typing import List

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from openai import OpenAI

# ---------- Setup ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="NEET Audio Conversation Renderer")

# ---------- Data models ----------

class Segment(BaseModel):
    speaker: str  # e.g. "DR_ARJUN" or "RIYA"
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

# ---------- Core render logic ----------

def build_conversation_mp3(req: RenderRequest) -> bytes:
    """
    Generate full conversation MP3 bytes by:
    - looping over segments
    - choosing voice + speed by speaker
    - concatenating all MP3 chunks
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
            speed = 1.00   # slightly slower
        elif speaker == "RIYA":
            speed = 1.05   # slightly faster
        else:
            speed = 1.0

        audio_bytes = tts_line_to_mp3_bytes(text, voice, speed)
        all_bytes += audio_bytes

    if not all_bytes:
        raise ValueError("No audio generated from segments")

    return all_bytes

# ---------- API endpoint ----------

@app.post("/render-conversation")
def render_conversation(req: RenderRequest):
    if not req.segments:
        raise HTTPException(status_code=400, detail="No segments provided")

    try:
        audio_bytes = build_conversation_mp3(req)
    except Exception as e:
        print("Render error:", e)
        raise HTTPException(status_code=500, detail=f"Render failed: {e}")

    # Return raw MP3 bytes; Apps Script will save to Google Drive
    return Response(content=audio_bytes, media_type="audio/mpeg")
