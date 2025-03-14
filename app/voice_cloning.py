"""
Voice cloning module for CSM-1B TTS API.

This module provides functionality to clone voices from audio samples,
with advanced audio preprocessing and voice adaptation techniques.
"""
import os
import io
import time
import tempfile
import logging
from typing import Dict, List, Optional, Union, Tuple, BinaryIO
from pathlib import Path

import numpy as np
import torch
import torchaudio
from pydantic import BaseModel
from fastapi import UploadFile

from app.models import Segment

# Set up logging
logger = logging.getLogger(__name__)

# Directory for storing cloned voice data
CLONED_VOICES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cloned_voices")
os.makedirs(CLONED_VOICES_DIR, exist_ok=True)

class ClonedVoice(BaseModel):
    """Model representing a cloned voice."""
    id: str
    name: str
    created_at: float
    speaker_id: int
    description: Optional[str] = None
    audio_duration: float
    sample_count: int


class VoiceCloner:
    """Voice cloning utility for CSM-1B model."""

    def __init__(self, generator, device="cuda"):
        """Initialize the voice cloner with a generator instance."""
        self.generator = generator
        self.device = device
        self.sample_rate = generator.sample_rate
        self.cloned_voices = self._load_existing_voices()
        logger.info(f"Voice cloner initialized with {len(self.cloned_voices)} existing voices")

    def _load_existing_voices(self) -> Dict[str, ClonedVoice]:
        """Load existing cloned voices from disk."""
        voices = {}
        if not os.path.exists(CLONED_VOICES_DIR):
            return voices

        for voice_dir in os.listdir(CLONED_VOICES_DIR):
            voice_path = os.path.join(CLONED_VOICES_DIR, voice_dir)
            if not os.path.isdir(voice_path):
                continue

            info_path = os.path.join(voice_path, "info.json")
            if os.path.exists(info_path):
                try:
                    import json
                    with open(info_path, "r") as f:
                        voice_info = json.load(f)
                        voices[voice_dir] = ClonedVoice(**voice_info)
                        logger.info(f"Loaded cloned voice: {voice_dir}")
                except Exception as e:
                    logger.error(f"Error loading voice {voice_dir}: {e}")

        return voices

    async def process_audio_file(
        self, 
        file: Union[UploadFile, BinaryIO, str],
        transcript: Optional[str] = None
    ) -> Tuple[torch.Tensor, Optional[str], float]:
        """
        Process an audio file for voice cloning.
        
        Args:
            file: The audio file (UploadFile, file-like object, or path)
            transcript: Optional transcript of the audio
            
        Returns:
            Tuple of (processed_audio, transcript, duration_seconds)
        """
        # Handle different input types
        if isinstance(file, str):
            # It's a file path
            audio_path = file
        elif hasattr(file, 'read'):
            # It's a file-like object
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                temp.write(file.read() if hasattr(file, 'read') else file)
                audio_path = temp.name
        else:
            # It's a FastAPI UploadFile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                temp.write(await file.read())
                audio_path = temp.name

        try:
            # Load audio
            audio, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Remove first dimension if it's 1
            if audio.shape[0] == 1:
                audio = audio.squeeze(0)
            
            # Resample if necessary
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, orig_freq=sr, new_freq=self.sample_rate
                )
            
            # Get audio duration
            duration_seconds = len(audio) / self.sample_rate
            
            # Process audio for better quality
            processed_audio = self._preprocess_audio(audio)
            processed_duration = len(processed_audio) / self.sample_rate
            
            logger.info(
                f"Processed audio: original duration={duration_seconds:.2f}s, "
                f"processed duration={processed_duration:.2f}s"
            )
            
            # Clean up temp file if we created one
            if isinstance(file, (UploadFile, BinaryIO)) and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {audio_path}: {e}")
            
            return processed_audio, transcript, duration_seconds
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            if isinstance(file, (UploadFile, BinaryIO)) and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass
            raise RuntimeError(f"Failed to process audio file: {e}")

    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Preprocess audio for better voice cloning quality.
        
        Args:
            audio: Raw audio tensor
            
        Returns:
            Processed audio tensor
        """
        # Normalize volume
        if torch.max(torch.abs(audio)) > 0:
            audio = audio / torch.max(torch.abs(audio))
        
        # Remove silence with dynamic threshold
        audio = self._remove_silence(audio)
        
        # Apply additional preprocessing
        audio = self._enhance_speech(audio)
        
        return audio

    def _remove_silence(
        self, 
        audio: torch.Tensor, 
        threshold: float = 0.015, 
        min_silence_duration: float = 0.2
    ) -> torch.Tensor:
        """
        Remove silence from audio while preserving speech rhythm.
        
        Args:
            audio: Input audio tensor
            threshold: Energy threshold for silence detection
            min_silence_duration: Minimum silence duration in seconds
            
        Returns:
            Audio with silence removed
        """
        # Convert to numpy for easier processing
        audio_np = audio.cpu().numpy()
        
        # Calculate energy
        energy = np.abs(audio_np)
        
        # Find regions above threshold (speech)
        is_speech = energy > threshold
        
        # Convert min_silence_duration to samples
        min_silence_samples = int(min_silence_duration * self.sample_rate)
        
        # Find speech segments
        speech_segments = []
        in_speech = False
        speech_start = 0
        
        for i in range(len(is_speech)):
            if is_speech[i] and not in_speech:
                # Start of speech segment
                in_speech = True
                speech_start = i
            elif not is_speech[i] and in_speech:
                # Potential end of speech segment
                # Only end if silence is long enough
                silence_count = 0
                for j in range(i, min(len(is_speech), i + min_silence_samples)):
                    if not is_speech[j]:
                        silence_count += 1
                    else:
                        break
                
                if silence_count >= min_silence_samples:
                    # End of speech segment
                    in_speech = False
                    speech_segments.append((speech_start, i))
        
        # Handle case where audio ends during speech
        if in_speech:
            speech_segments.append((speech_start, len(is_speech)))
        
        # If no speech segments found, return original audio
        if not speech_segments:
            logger.warning("No speech segments detected, returning original audio")
            return audio
        
        # Add small buffer around segments
        buffer_samples = int(0.05 * self.sample_rate)  # 50ms buffer
        processed_segments = []
        
        for start, end in speech_segments:
            buffered_start = max(0, start - buffer_samples)
            buffered_end = min(len(audio_np), end + buffer_samples)
            processed_segments.append(audio_np[buffered_start:buffered_end])
        
        # Concatenate all segments with small pauses between them
        small_pause = np.zeros(int(0.15 * self.sample_rate))  # 150ms pause
        result = processed_segments[0]
        
        for segment in processed_segments[1:]:
            result = np.concatenate([result, small_pause, segment])
        
        return torch.tensor(result, device=audio.device)

    def _enhance_speech(self, audio: torch.Tensor) -> torch.Tensor:
        """Enhance speech quality for better cloning results."""
        # This is a placeholder for more advanced speech enhancement
        # In a production implementation, you could add:
        # - Noise reduction
        # - Equalization for speech frequencies
        # - Gentle compression for better dynamics
        return audio

    async def clone_voice(
        self,
        audio_file: Union[UploadFile, BinaryIO, str],
        voice_name: str,
        transcript: Optional[str] = None,
        description: Optional[str] = None,
        speaker_id: int = 999  # Use a high ID to avoid conflicts
    ) -> ClonedVoice:
        """
        Clone a voice from an audio file.
        
        Args:
            audio_file: Audio file with the voice to clone
            voice_name: Name for the cloned voice
            transcript: Transcript of the audio (optional)
            description: Description of the voice (optional)
            speaker_id: Speaker ID to use (default: 999)
            
        Returns:
            ClonedVoice object with voice information
        """
        # Process the audio file
        processed_audio, provided_transcript, duration = await self.process_audio_file(
            audio_file, transcript
        )
        
        # Generate a unique ID for the voice
        voice_id = f"{int(time.time())}_{voice_name.lower().replace(' ', '_')}"
        
        # Create directory for the voice
        voice_dir = os.path.join(CLONED_VOICES_DIR, voice_id)
        os.makedirs(voice_dir, exist_ok=True)
        
        # Save the processed audio
        audio_path = os.path.join(voice_dir, "reference.wav")
        torchaudio.save(audio_path, processed_audio.unsqueeze(0).cpu(), self.sample_rate)
        
        # Save the transcript if provided
        if provided_transcript:
            transcript_path = os.path.join(voice_dir, "transcript.txt")
            with open(transcript_path, "w") as f:
                f.write(provided_transcript)
        
        # Create and save voice info
        voice_info = ClonedVoice(
            id=voice_id,
            name=voice_name,
            created_at=time.time(),
            speaker_id=speaker_id,
            description=description,
            audio_duration=duration,
            sample_count=len(processed_audio)
        )
        
        # Save voice info as JSON
        import json
        with open(os.path.join(voice_dir, "info.json"), "w") as f:
            f.write(json.dumps(voice_info.dict()))
        
        # Add to cloned voices dictionary
        self.cloned_voices[voice_id] = voice_info
        
        logger.info(f"Voice '{voice_name}' cloned successfully with ID: {voice_id}")
        
        return voice_info

    def get_voice_context(self, voice_id: str) -> List[Segment]:
        """
        Get context segments for a cloned voice.
        
        Args:
            voice_id: ID of the cloned voice
            
        Returns:
            List of context segments for the voice
        """
        if voice_id not in self.cloned_voices:
            logger.warning(f"Voice ID {voice_id} not found")
            return []
        
        voice = self.cloned_voices[voice_id]
        voice_dir = os.path.join(CLONED_VOICES_DIR, voice_id)
        audio_path = os.path.join(voice_dir, "reference.wav")
        
        if not os.path.exists(audio_path):
            logger.error(f"Audio file for voice {voice_id} not found at {audio_path}")
            return []
        
        # Load the audio
        audio, sr = torchaudio.load(audio_path)
        audio = audio.squeeze(0)
        
        # Resample if necessary
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio, orig_freq=sr, new_freq=self.sample_rate
            )
        
        # Load transcript if available
        transcript_path = os.path.join(voice_dir, "transcript.txt")
        transcript = ""
        if os.path.exists(transcript_path):
            with open(transcript_path, "r") as f:
                transcript = f.read()
        else:
            transcript = f"Voice sample for {voice.name}"
        
        # Create context segment
        segment = Segment(
            text=transcript,
            speaker=voice.speaker_id,
            audio=audio.to(self.device)
        )
        
        return [segment]

    def list_voices(self) -> List[ClonedVoice]:
        """List all available cloned voices."""
        return list(self.cloned_voices.values())

    def delete_voice(self, voice_id: str) -> bool:
        """
        Delete a cloned voice.
        
        Args:
            voice_id: ID of the voice to delete
            
        Returns:
            True if successful, False otherwise
        """
        if voice_id not in self.cloned_voices:
            return False
        
        voice_dir = os.path.join(CLONED_VOICES_DIR, voice_id)
        if os.path.exists(voice_dir):
            try:
                import shutil
                shutil.rmtree(voice_dir)
                del self.cloned_voices[voice_id]
                return True
            except Exception as e:
                logger.error(f"Error deleting voice {voice_id}: {e}")
                return False
        
        return False

    async def generate_speech(
        self,
        text: str,
        voice_id: str,
        temperature: float = 0.65,
        topk: int = 30,
        max_audio_length_ms: int = 15000
    ) -> torch.Tensor:
        """
        Generate speech with a cloned voice.
        
        Args:
            text: Text to synthesize
            voice_id: ID of the cloned voice to use
            temperature: Sampling temperature (lower = more stable, higher = more varied)
            topk: Top-k sampling parameter
            max_audio_length_ms: Maximum audio length in milliseconds
            
        Returns:
            Generated audio tensor
        """
        if voice_id not in self.cloned_voices:
            raise ValueError(f"Voice ID {voice_id} not found")
        
        voice = self.cloned_voices[voice_id]
        context = self.get_voice_context(voice_id)
        
        if not context:
            raise ValueError(f"Could not get context for voice {voice_id}")
        
        # Preprocess text for better pronunciation
        processed_text = self._preprocess_text(text)
        
        # Generate audio with the cloned voice
        audio = self.generator.generate(
            text=processed_text,
            speaker=voice.speaker_id,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
        )
        
        return audio

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better pronunciation and voice cloning."""
        # Make sure text ends with punctuation for better phrasing
        text = text.strip()
        if not text.endswith(('.', '?', '!', ';')):
            text = text + '.'
        
        return text
