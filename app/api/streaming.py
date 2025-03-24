"""Streaming support for the TTS API."""
import asyncio
import io
import logging
import time
from typing import AsyncGenerator, Optional
import torch
import torchaudio
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from app.api.schemas import SpeechRequest, ResponseFormat

logger = logging.getLogger(__name__)
router = APIRouter()

class AudioChunker:
    """Handle audio chunking for streaming responses."""
    
    def __init__(self, 
                 sample_rate: int, 
                 format: str = "mp3", 
                 chunk_size_ms: int = 500):
        """
        Initialize audio chunker.
        
        Args:
            sample_rate: Audio sample rate in Hz
            format: Output audio format (mp3, opus, etc.)
            chunk_size_ms: Size of each chunk in milliseconds
        """
        self.sample_rate = sample_rate
        self.format = format.lower()
        self.chunk_size_samples = int(sample_rate * (chunk_size_ms / 1000))
        logger.info(f"Audio chunker initialized with {chunk_size_ms}ms chunks ({self.chunk_size_samples} samples)")

    async def chunk_audio(self, 
                    audio: torch.Tensor, 
                    delay_ms: int = 0) -> AsyncGenerator[bytes, None]:
        """
        Convert audio tensor to streaming chunks.
        
        Args:
            audio: Audio tensor to stream
            delay_ms: Artificial delay between chunks (for testing)
            
        Yields:
            Audio chunks as bytes
        """
        # Ensure audio is on CPU
        if audio.is_cuda:
            audio = audio.cpu()
            
        # Calculate number of chunks
        num_samples = audio.shape[0]
        num_chunks = (num_samples + self.chunk_size_samples - 1) // self.chunk_size_samples
        
        logger.info(f"Streaming {num_samples} samples as {num_chunks} chunks")
        
        for i in range(num_chunks):
            start_idx = i * self.chunk_size_samples
            end_idx = min(start_idx + self.chunk_size_samples, num_samples)
            
            # Extract chunk
            chunk = audio[start_idx:end_idx]
            
            # Convert to bytes in requested format
            chunk_bytes = await self._format_chunk(chunk)
            
            # Add artificial delay if requested (for testing)
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000)
                
            yield chunk_bytes
                    
    async def _format_chunk(self, chunk: torch.Tensor) -> bytes:
        """Convert audio chunk to bytes in the specified format."""
        buf = io.BytesIO()
        
        # Ensure chunk is 1D and on CPU
        if len(chunk.shape) == 1:
            chunk = chunk.unsqueeze(0)  # Add channel dimension
        
        # Ensure chunk is on CPU
        if chunk.is_cuda:
            chunk = chunk.cpu()
            
        # Save to buffer in specified format
        if self.format == "mp3":
            torchaudio.save(buf, chunk, self.sample_rate, format="mp3")
        elif self.format == "opus":
            torchaudio.save(buf, chunk, self.sample_rate, format="opus")
        elif self.format == "aac":
            torchaudio.save(buf, chunk, self.sample_rate, format="aac")
        elif self.format == "flac":
            torchaudio.save(buf, chunk, self.sample_rate, format="flac")
        elif self.format == "wav":
            torchaudio.save(buf, chunk, self.sample_rate, format="wav")
        else:
            # Default to mp3
            torchaudio.save(buf, chunk, self.sample_rate, format="mp3")
            
        # Get bytes from buffer
        buf.seek(0)
        return buf.read()

# Helper function to get speaker ID for a voice
def get_speaker_id(app_state, voice):
    """Helper function to get speaker ID from voice name or ID"""
    if hasattr(app_state, "voice_speaker_map") and voice in app_state.voice_speaker_map:
        return app_state.voice_speaker_map[voice]
        
    # Standard voices mapping
    voice_to_speaker = {"alloy": 0, "echo": 1, "fable": 2, "onyx": 3, "nova": 4, "shimmer": 5}
    
    if voice in voice_to_speaker:
        return voice_to_speaker[voice]
    
    # Try parsing as integer
    try:
        speaker_id = int(voice)
        if 0 <= speaker_id < 6:
            return speaker_id
    except (ValueError, TypeError):
        pass
    
    # Check cloned voices if the voice cloner exists
    if hasattr(app_state, "voice_cloner") and app_state.voice_cloner is not None:
        # Check by ID
        if voice in app_state.voice_cloner.cloned_voices:
            return app_state.voice_cloner.cloned_voices[voice].speaker_id
            
        # Check by name
        for v_id, v_info in app_state.voice_cloner.cloned_voices.items():
            if v_info.name.lower() == voice.lower():
                return v_info.speaker_id
    
    # Default to alloy
    return 0

@router.post("/audio/speech/stream", tags=["Audio"])
async def stream_speech(
    request: Request,
    speech_request: SpeechRequest,
):
    """
    Stream audio of text being spoken by a realistic voice.
    
    This endpoint provides an OpenAI-compatible streaming interface for TTS.
    """
    # Check if model is loaded
    if not hasattr(request.app.state, "generator") or request.app.state.generator is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please try again later."
        )
        
    # Get request parameters
    model = speech_request.model
    input_text = speech_request.input
    voice = speech_request.voice
    response_format = speech_request.response_format
    speed = speech_request.speed
    temperature = speech_request.temperature
    max_audio_length_ms = speech_request.max_audio_length_ms
    
    # Log the request
    logger.info(f"Streaming speech from text ({len(input_text)} chars) with voice '{voice}'")
    
    # Check if text is empty
    if not input_text or len(input_text.strip()) == 0:
        raise HTTPException(
            status_code=400, 
            detail="Input text cannot be empty"
        )
        
    # Get speaker ID for the voice
    speaker_id = get_speaker_id(request.app.state, voice)
    if speaker_id is None:
        raise HTTPException(
            status_code=400, 
            detail=f"Voice '{voice}' not found. Available voices: {request.app.state.available_voices}"
        )
        
    # Use voice memory for context
    try:
        # Create media type based on format
        media_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
        }.get(response_format, "audio/mpeg")
        
        # Create the chunker for streaming
        sample_rate = request.app.state.sample_rate
        chunker = AudioChunker(sample_rate, response_format)
        
        # Prepare context from voice memory
        if hasattr(request.app.state, "voice_memory_enabled") and request.app.state.voice_memory_enabled:
            from app.voice_memory import get_voice_context
            context = get_voice_context(voice, request.app.state.device)
        else:
            # Empty context
            context = []
            
        # Check for cloned voice
        voice_info = None
        if hasattr(request.app.state, "voice_cloning_enabled") and request.app.state.voice_cloning_enabled:
            voice_info = request.app.state.get_voice_info(voice)
            if voice_info and voice_info["type"] == "cloned":
                # Use cloned voice context
                from app.voice_cloning import VoiceCloner
                voice_cloner = request.app.state.voice_cloner
                context = voice_cloner.get_voice_context(voice_info["voice_id"])
                
        # Generate audio in a separate task to avoid blocking
        async def generate_streaming_audio():
            try:
                # Apply optional text enhancement
                enhanced_text = input_text
                if hasattr(request.app.state, "prompt_templates"):
                    from app.prompt_engineering import format_text_for_voice
                    enhanced_text = format_text_for_voice(input_text, voice)
                
                # Generate audio
                if voice_info and voice_info["type"] == "cloned":
                    # Generate with cloned voice
                    voice_cloner = request.app.state.voice_cloner
                    audio = voice_cloner.generate_speech(
                        text=enhanced_text,
                        voice_id=voice_info["voice_id"],
                        temperature=temperature,
                        topk=speech_request.topk or 30,
                        max_audio_length_ms=max_audio_length_ms
                    )
                else:
                    # Generate with standard voice
                    audio = request.app.state.generator.generate(
                        text=enhanced_text,
                        speaker=speaker_id,
                        context=context,
                        max_audio_length_ms=max_audio_length_ms,
                        temperature=temperature,
                    )
                
                # Process the audio for better quality
                if hasattr(request.app.state, "voice_enhancement_enabled") and request.app.state.voice_enhancement_enabled:
                    from app.voice_enhancement import process_generated_audio
                    audio = process_generated_audio(
                        audio=audio,
                        voice_name=voice,
                        sample_rate=sample_rate,
                        text=enhanced_text
                    )
                    
                # Update voice memory with the new audio
                if hasattr(request.app.state, "voice_memory_enabled") and request.app.state.voice_memory_enabled:
                    from app.voice_memory import update_voice_memory
                    update_voice_memory(voice, audio, enhanced_text)
                    
                # Handle speed adjustments if not 1.0
                if speed != 1.0 and speed > 0:
                    try:
                        # Adjust speed using torchaudio
                        effects = [
                            ["tempo", str(speed)]
                        ]
                        audio_cpu = audio.cpu()
                        adjusted_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
                            audio_cpu.unsqueeze(0), 
                            sample_rate, 
                            effects
                        )
                        audio = adjusted_audio.squeeze(0)
                        logger.info(f"Adjusted speech speed to {speed}x")
                    except Exception as e:
                        logger.warning(f"Failed to adjust speech speed: {e}")
                    
                # Stream the audio in chunks
                async for chunk in chunker.chunk_audio(audio):
                    yield chunk
                    
            except Exception as e:
                logger.error(f"Error generating streaming audio: {e}")
                # Send an error message - this will cause client to fail,
                # but at least we're sending something back
                error_message = f"Error: {str(e)}".encode()
                yield error_message
        
        # Return streaming response
        return StreamingResponse(
            generate_streaming_audio(),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{response_format}"'
            }
        )
    except Exception as e:
        logger.error(f"Error in stream_speech: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@router.post("/audio/speech/streaming", tags=["Audio"])
async def openai_stream_speech(
    request: Request,
    speech_request: SpeechRequest,
):
    """
    Stream audio in OpenAI-compatible streaming format.
    
    This endpoint is compatible with the OpenAI streaming TTS API.
    """
    # Use the same logic as the stream_speech endpoint but with a different name
    # to maintain the OpenAI API naming convention
    return await stream_speech(request, speech_request)