# In app/api/schemas.py
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field

# Remove the Voice enum or make it non-restrictive
class Voice(str):
    """Voice options for CSM model - allowing any string value"""
    pass

class ResponseFormat(str, Enum):
    mp3 = "mp3"
    opus = "opus"
    aac = "aac"
    flac = "flac"
    wav = "wav"

class TTSRequest(BaseModel):
    model: Optional[str] = Field("csm-1b", description="The TTS model to use")
    input: str = Field(..., description="The text to generate audio for")
    # Change this line to accept any string for voice
    voice: Optional[str] = Field("alloy", description="The voice to use for generation")
    response_format: Optional[ResponseFormat] = Field(ResponseFormat.mp3, description="The format of the audio response")
    speed: Optional[float] = Field(1.0, description="The speed of the audio", ge=0.25, le=4.0)
    # These are CSM-specific parameters that aren't in standard OpenAI API
    max_audio_length_ms: Optional[float] = Field(90000, description="Maximum audio length in milliseconds")
    temperature: Optional[float] = Field(0.9, description="Sampling temperature", ge=0.0, le=2.0)
    topk: Optional[int] = Field(50, description="Top-k for sampling", ge=1, le=100)

    # Make all optional fields truly optional in JSON
    class Config:
        populate_by_name = True
        extra = "ignore"  # Allow extra fields without error

class TTSResponse(BaseModel):
    """Only used for API documentation"""
    pass