import os
import io
import time
import tempfile
from typing import Dict, List, Optional, Any, Union

import torch
import torchaudio
import numpy as np
from fastapi import APIRouter, Request, Response, Depends, HTTPException, Body
from fastapi.responses import StreamingResponse

from app.api.schemas import TTSRequest, ResponseFormat, Voice
from app.models import Segment
from app.voice_memory import get_voice_context, update_voice_memory
from app.voice_embeddings import initialize_voices, get_voice_sample, update_voice_sample

router = APIRouter()

# Mapping of response_format to MIME types
MIME_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
}

@router.post("/audio/speech", summary="Generate speech from text")
async def text_to_speech(
    request: Request,
    body: Dict[str, Any] = Body(...)
):
    """
    OpenAI compatible TTS endpoint that generates speech from text using the CSM-1B model.
    """
    # Get generator from app state
    generator = request.app.state.generator
    
    # Validate model availability
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Log the request body to debug
    print(f"Received request body: {body}")
    
    # Extract parameters with fallbacks
    text = body.get("input", "")
    if not text:
        # Try 'text' as an alternative to 'input'
        text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'input' field in request")
    
    # Handle voice parameter
    voice_name = body.get("voice", "alloy")
    # Convert string voice name to speaker ID
    try:
        if voice_name in ["0", "1", "2", "3", "4", "5"]:
            # Already a numeric string
            speaker = int(voice_name)
            voice_name = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"][speaker]
        elif voice_name in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            # Convert named voice to speaker ID
            voice_map = {
                "alloy": 0, 
                "echo": 1, 
                "fable": 2, 
                "onyx": 3, 
                "nova": 4, 
                "shimmer": 5
            }
            speaker = voice_map[voice_name]
        else:
            # Check if it's a custom voice in voice memories
            from app.voice_memory import VOICE_MEMORIES
            if voice_name in VOICE_MEMORIES:
                speaker = VOICE_MEMORIES[voice_name].speaker_id
            else:
                # Default to speaker 0
                print(f"Unknown voice '{voice_name}', defaulting to speaker 0")
                speaker = 0
                voice_name = "alloy"
    except Exception as e:
        print(f"Error processing voice parameter: {e}")
        speaker = 0
        voice_name = "alloy"
    
    # Handle other parameters
    response_format = body.get("response_format", "mp3")
    speed = float(body.get("speed", 1.0))
    max_audio_length_ms = float(body.get("max_audio_length_ms", 90000))
    
    # Voice consistency parameters
    voice_consistency = float(body.get("voice_consistency", 0.8))  # How strongly to maintain voice characteristics
    max_context_segments = int(body.get("max_context_segments", 2))  # Number of context segments to use
    prioritize_voice_consistency = body.get("prioritize_voice_consistency", False)
    
    # Adjust temperature based on voice consistency priority
    temperature = float(body.get("temperature", 0.9))
    if prioritize_voice_consistency:
        temperature = min(temperature, 0.7)  # Lower temperature for more consistent voice
    
    topk = int(body.get("topk", 50))
    
    # Generate audio
    try:
        print(f"Generating audio for: '{text}' with voice={voice_name} (speaker={speaker})")
        
        # Get context segments for this voice - using the enhanced voice memory system
        from app.voice_memory import get_voice_context, update_voice_memory
        
        context = get_voice_context(voice_name, generator.device, max_segments=max_context_segments)
        if context:
            print(f"Using {len(context)} context segments for voice consistency")
            
            # Adjust context influence based on voice_consistency parameter
            # Higher value = stronger influence of previous voice samples
            if voice_consistency > 0:
                for segment in context:
                    # Scale the audio to increase/decrease its influence on the generation
                    segment.audio = segment.audio * voice_consistency
        
        # Generate audio
        audio = generator.generate(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
        )
        
        # Update voice memory with the newly generated audio
        # Only update if the generation seemed successful (audio not too short or empty)
        if audio is not None and audio.shape[0] > 1000:  # Minimum audio length check
            update_voice_memory(voice_name, audio, text)
        
        # Apply speed adjustment if needed (using resample)
        if speed != 1.0:
            # Calculate new sample rate based on speed
            new_sample_rate = int(generator.sample_rate * speed)
            audio = torchaudio.functional.resample(
                audio, 
                orig_freq=generator.sample_rate, 
                new_freq=new_sample_rate
            )
            # Resample back to original rate to maintain compatibility
            audio = torchaudio.functional.resample(
                audio, 
                orig_freq=new_sample_rate, 
                new_freq=generator.sample_rate
            )
        
        # Create temporary file for audio conversion
        with tempfile.NamedTemporaryFile(suffix=f".{response_format}", delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Save to WAV first (direct format for torchaudio)
        wav_path = f"{temp_path}.wav"
        torchaudio.save(wav_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        # Convert to requested format using ffmpeg
        import ffmpeg
        
        if response_format == "mp3":
            # For MP3, use specific bitrate for better quality
            (
                ffmpeg.input(wav_path)
                .output(temp_path, format='mp3', audio_bitrate='128k')
                .run(quiet=True, overwrite_output=True)
            )
        elif response_format == "opus":
            (
                ffmpeg.input(wav_path)
                .output(temp_path, format='opus')
                .run(quiet=True, overwrite_output=True)
            )
        elif response_format == "aac":
            (
                ffmpeg.input(wav_path)
                .output(temp_path, format='aac')
                .run(quiet=True, overwrite_output=True)
            )
        elif response_format == "flac":
            (
                ffmpeg.input(wav_path)
                .output(temp_path, format='flac')
                .run(quiet=True, overwrite_output=True)
            )
        else:  # wav
            temp_path = wav_path  # Just use the WAV file directly
            response_format = "wav"  # Ensure correct MIME type
        
        # Clean up the temporary WAV file if we created a different format
        if temp_path != wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)
        
        # Return audio file as response
        def iterfile():
            with open(temp_path, 'rb') as f:
                yield from f
            # Clean up temp file after streaming
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return StreamingResponse(
            iterfile(),
            media_type=MIME_TYPES.get(response_format, "application/octet-stream"),
            headers={'Content-Disposition': f'attachment; filename="speech.{response_format}"'}
        )
        
    except Exception as e:
        import traceback
        print(f"Speech generation failed: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")

@router.post("/audio/voices/create", summary="Create a new voice")
async def create_voice(
    request: Request,
    name: str = Body(..., description="Name of the voice to create"),
    initial_text: str = Body(..., description="Text for the initial voice sample"),
    speaker_id: int = Body(0, description="Base speaker ID (0-5)"),
    pitch: Optional[float] = Body(None, description="Base pitch in Hz (optional)"),
    timbre: str = Body("custom", description="Voice quality descriptor")
):
    """Create a new custom voice."""
    from app.voice_memory import create_custom_voice
    
    result = create_custom_voice(
        app_state=request.app.state,
        name=name,
        initial_text=initial_text,
        speaker_id=speaker_id,
        pitch=pitch,
        timbre=timbre
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result
    
@router.post("/audio/conversation", tags=["Conversation API"])
async def conversation_to_speech(
    request: Request,
    text: str = Body(..., description="Text to convert to speech"),
    speaker_id: int = Body(0, description="Speaker ID"),
    context: List[Dict] = Body([], description="Context segments with speaker, text, and audio path"),
):
    """
    Custom endpoint for conversational TTS using CSM-1B.
    
    This is not part of the OpenAI API but provides the unique conversational
    capability of the CSM model.
    """
    # Get generator from app state
    generator = request.app.state.generator
    
    # Validate model availability
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        segments = []
        
        # Process context if provided
        for ctx in context:
            if 'speaker' not in ctx or 'text' not in ctx or 'audio' not in ctx:
                continue
                
            # Audio should be base64-encoded
            audio_data = base64.b64decode(ctx['audio'])
            audio_file = io.BytesIO(audio_data)
            
            # Save to temporary file for torchaudio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                temp.write(audio_file.read())
                temp_path = temp.name
            
            # Load audio
            audio_tensor, sample_rate = torchaudio.load(temp_path)
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0), 
                orig_freq=sample_rate, 
                new_freq=generator.sample_rate
            )
            
            # Clean up
            os.unlink(temp_path)
            
            # Create segment
            segments.append(
                Segment(
                    speaker=ctx['speaker'],
                    text=ctx['text'],
                    audio=audio_tensor
                )
            )
        
        # Generate audio with context
        audio = generator.generate(
            text=text,
            speaker=speaker_id,
            context=segments,
            max_audio_length_ms=10000,  # 10 seconds
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            temp_path = temp.name
        
        # Save audio
        torchaudio.save(temp_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        # Return audio file
        def iterfile():
            with open(temp_path, 'rb') as f:
                yield from f
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return StreamingResponse(
            iterfile(),
            media_type="audio/wav",
            headers={'Content-Disposition': 'attachment; filename="speech.wav"'}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation speech generation failed: {str(e)}")

# Add OpenAI-compatible voice list endpoint
@router.get("/audio/voices", summary="List available voices")
async def list_voices():
    """
    OpenAI compatible endpoint that returns a list of available voices.
    """
    voices = [
        {
            "voice_id": "alloy",
            "name": "Alloy",
            "preview_url": None,
            "description": "CSM Speaker 0",
            "languages": [{"language_code": "en", "name": "English"}]
        },
        {
            "voice_id": "echo",
            "name": "Echo",
            "preview_url": None,
            "description": "CSM Speaker 1",
            "languages": [{"language_code": "en", "name": "English"}]
        },
        {
            "voice_id": "fable",
            "name": "Fable",
            "preview_url": None,
            "description": "CSM Speaker 2",
            "languages": [{"language_code": "en", "name": "English"}]
        },
        {
            "voice_id": "onyx",
            "name": "Onyx",
            "preview_url": None,
            "description": "CSM Speaker 3",
            "languages": [{"language_code": "en", "name": "English"}]
        },
        {
            "voice_id": "nova",
            "name": "Nova",
            "preview_url": None,
            "description": "CSM Speaker 4",
            "languages": [{"language_code": "en", "name": "English"}]
        },
        {
            "voice_id": "shimmer",
            "name": "Shimmer",
            "preview_url": None,
            "description": "CSM Speaker 5",
            "languages": [{"language_code": "en", "name": "English"}]
        }
    ]
    
    return {"voices": voices}

# Add OpenAI-compatible models list endpoint
@router.get("/audio/models", summary="List available audio models")
async def list_models():
    """
    OpenAI compatible endpoint that returns a list of available audio models.
    """
    models = [
        {
            "id": "csm-1b",
            "name": "CSM-1B",
            "description": "Conversational Speech Model 1B from Sesame",
            "created": 1716019200,  # March 13, 2025 (from the example)
            "object": "audio",
            "owned_by": "sesame",
            "capabilities": {
                "tts": True,
                "voice_generation": False,
            },
            "max_input_length": 4096,
            "price": {"text-to-speech": 0.00}
        },
        {
            "id": "tts-1",
            "name": "CSM-1B (Compatibility Mode)",
            "description": "CSM-1B with OpenAI TTS-1 compatibility",
            "created": 1716019200,
            "object": "audio",
            "owned_by": "sesame",
            "capabilities": {
                "tts": True,
                "voice_generation": False,
            },
            "max_input_length": 4096,
            "price": {"text-to-speech": 0.00}
        },
        {
            "id": "tts-1-hd",
            "name": "CSM-1B (HD Mode)",
            "description": "CSM-1B with higher quality settings",
            "created": 1716019200,
            "object": "audio",
            "owned_by": "sesame",
            "capabilities": {
                "tts": True,
                "voice_generation": False,
            },
            "max_input_length": 4096,
            "price": {"text-to-speech": 0.00}
        }
    ]
    
    return {"data": models, "object": "list"}

# Response format options endpoint
@router.get("/audio/speech/response-formats", summary="List available response formats")
async def list_response_formats():
    """List available response formats for speech synthesis."""
    formats = [
        {"name": "mp3", "content_type": "audio/mpeg"},
        {"name": "opus", "content_type": "audio/opus"},
        {"name": "aac", "content_type": "audio/aac"},
        {"name": "flac", "content_type": "audio/flac"},
        {"name": "wav", "content_type": "audio/wav"}
    ]
    
    return {"response_formats": formats}

# Simple test endpoint
@router.get("/test", summary="Test endpoint")
async def test_endpoint():
    """Simple test endpoint that returns a successful response."""
    return {"status": "ok", "message": "API is working"}

# Debug endpoint
@router.get("/debug", summary="Debug endpoint")
async def debug_info(request: Request):
    """Get debug information about the API."""
    generator = request.app.state.generator
    
    debug_info = {
        "model_loaded": generator is not None,
        "device": generator.device if generator is not None else None,
        "sample_rate": generator.sample_rate if generator is not None else None,
        "voices_available": [v.value for v in Voice],
        "response_formats": [f.value for f in ResponseFormat],
    }
    
    return debug_info

# Specialized debugging endpoint for speech generation
@router.post("/debug/speech", summary="Debug speech generation")
async def debug_speech(
    request: Request,
    text: str = Body(..., embed=True),
    speaker: int = Body(0, embed=True)
):
    """Debug endpoint for speech generation."""
    generator = request.app.state.generator
    
    if generator is None:
        return {"error": "Model not loaded"}
    
    try:
        # Generate audio
        audio = generator.generate(
            text=text,
            speaker=speaker,
            context=[],
            max_audio_length_ms=10000,  # Short for testing
            temperature=0.9,
            topk=50,
        )
        
        # Save to temporary WAV file
        temp_path = f"/tmp/debug_speech_{int(time.time())}.wav"
        torchaudio.save(temp_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        return {
            "status": "success",
            "message": f"Audio generated successfully and saved to {temp_path}",
            "audio_shape": list(audio.shape),
            "sample_rate": generator.sample_rate
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return {
            "status": "error",
            "message": str(e),
            "traceback": error_trace
        }