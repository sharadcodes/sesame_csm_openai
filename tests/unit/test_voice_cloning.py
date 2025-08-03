import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock, mock_open
import torch
import numpy as np
import tempfile
import os
import json
from pathlib import Path


class TestVoiceCloning:
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies for VoiceCloningSystem"""
        with patch('app.voice_cloning.whisper') as mock_whisper, \
             patch('app.voice_cloning.torchaudio') as mock_torchaudio, \
             patch('app.voice_cloning.Path') as mock_path, \
             patch('app.voice_cloning.torch.cuda.is_available', return_value=True):
            
            # Mock whisper model
            mock_model = Mock()
            mock_model.transcribe = Mock(return_value={
                "text": "Sample transcription",
                "segments": [{"start": 0, "end": 2, "text": "Sample"}]
            })
            mock_whisper.load_model.return_value = mock_model
            
            # Mock torchaudio
            mock_torchaudio.load.return_value = (torch.randn(1, 16000), 16000)
            mock_torchaudio.save = Mock()
            
            yield {
                'whisper': mock_whisper,
                'torchaudio': mock_torchaudio,
                'path': mock_path,
                'model': mock_model
            }

    @patch('app.voice_cloning.VoiceCloningSystem.__init__', return_value=None)
    def test_voice_cloning_initialization(self, mock_init):
        """Test VoiceCloningSystem initialization"""
        from app.voice_cloning import VoiceCloningSystem
        
        system = VoiceCloningSystem("/models", Mock())
        mock_init.assert_called_once()

    def test_generate_voice_id(self):
        """Test voice ID generation"""
        from app.voice_cloning import VoiceCloningSystem
        
        with patch('app.voice_cloning.uuid.uuid4', return_value=Mock(hex='test123')):
            voice_id = VoiceCloningSystem._generate_voice_id("test_voice")
            assert voice_id == "test_voice_test123"

    @patch('app.voice_cloning.os.makedirs')
    @patch('builtins.open', mock_open())
    def test_save_voice_metadata(self, mock_makedirs):
        """Test saving voice metadata"""
        from app.voice_cloning import VoiceCloningSystem
        
        # Create a minimal system instance
        with patch.object(VoiceCloningSystem, '__init__', return_value=None):
            system = VoiceCloningSystem("/models", Mock())
            system.voices_dir = Path("/test/voices")
            
            metadata = {
                "voice_id": "test_voice_123",
                "name": "Test Voice",
                "created_at": "2024-01-01"
            }
            
            system._save_voice_metadata("test_voice_123", metadata)
            
            # Verify file was written
            handle = open()
            handle.write.assert_called()
            written_data = ''.join([call.args[0] for call in handle.write.call_args_list])
            assert json.loads(written_data) == metadata

    @patch('app.voice_cloning.ffmpeg.input')
    def test_extract_audio_from_video(self, mock_ffmpeg):
        """Test audio extraction from video files"""
        from app.voice_cloning import VoiceCloningSystem
        
        # Mock ffmpeg chain
        mock_audio = Mock()
        mock_output = Mock()
        mock_ffmpeg.return_value.audio = mock_audio
        mock_audio.output.return_value = mock_output
        
        with patch.object(VoiceCloningSystem, '__init__', return_value=None):
            system = VoiceCloningSystem("/models", Mock())
            
            result = system._extract_audio_from_video("/input/video.mp4", "/output/audio.wav")
            
            # Verify ffmpeg chain
            mock_ffmpeg.assert_called_once_with("/input/video.mp4")
            mock_audio.output.assert_called_once_with("/output/audio.wav", acodec='pcm_s16le', ar=16000, ac=1)
            mock_output.run.assert_called_once_with(overwrite_output=True, quiet=True)
            assert result == "/output/audio.wav"

    @patch('app.voice_cloning.torchaudio.load')
    @patch('app.voice_cloning.torchaudio.functional.resample')
    def test_process_audio_file(self, mock_resample, mock_load):
        """Test audio file processing"""
        from app.voice_cloning import VoiceCloningSystem
        
        # Mock audio loading
        mock_load.return_value = (torch.randn(2, 32000), 32000)  # Stereo, 32kHz
        mock_resample.return_value = torch.randn(1, 16000)  # Mono, 16kHz
        
        with patch.object(VoiceCloningSystem, '__init__', return_value=None):
            system = VoiceCloningSystem("/models", Mock())
            system.logger = Mock()
            
            waveform, sr = system._process_audio_file("/test/audio.wav")
            
            # Verify processing
            assert sr == 16000
            assert waveform.shape[0] == 1  # Mono
            mock_resample.assert_called_once()

    @patch('app.voice_cloning.yt_dlp.YoutubeDL')
    def test_download_youtube_audio(self, mock_ytdl_class):
        """Test YouTube audio download"""
        from app.voice_cloning import VoiceCloningSystem
        
        # Mock YoutubeDL instance
        mock_ytdl = Mock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_ytdl
        mock_ytdl.extract_info.return_value = {"title": "Test Video"}
        
        with patch.object(VoiceCloningSystem, '__init__', return_value=None), \
             patch('os.path.exists', return_value=True):
            system = VoiceCloningSystem("/models", Mock())
            system.logger = Mock()
            
            result = system._download_youtube_audio("https://youtube.com/watch?v=test", "/output")
            
            # Verify download
            mock_ytdl.extract_info.assert_called_once_with("https://youtube.com/watch?v=test", download=True)
            assert result == "/output/audio.wav"

    def test_validate_audio_duration(self):
        """Test audio duration validation"""
        from app.voice_cloning import VoiceCloningSystem
        
        with patch.object(VoiceCloningSystem, '__init__', return_value=None):
            system = VoiceCloningSystem("/models", Mock())
            
            # Test valid durations
            assert system._validate_audio_duration(torch.randn(1, 48000), 16000) == True  # 3 seconds
            assert system._validate_audio_duration(torch.randn(1, 160000), 16000) == True  # 10 seconds
            
            # Test invalid durations
            assert system._validate_audio_duration(torch.randn(1, 16000), 16000) == False  # 1 second
            assert system._validate_audio_duration(torch.randn(1, 480000), 16000) == False  # 30 seconds

    @patch('app.voice_cloning.os.listdir')
    @patch('builtins.open', mock_open(read_data='{"voice_id": "test_123", "name": "Test Voice"}'))
    def test_list_cloned_voices(self, mock_listdir):
        """Test listing cloned voices"""
        from app.voice_cloning import VoiceCloningSystem
        
        mock_listdir.return_value = ['test_123', 'another_456']
        
        with patch.object(VoiceCloningSystem, '__init__', return_value=None), \
             patch('os.path.isdir', return_value=True), \
             patch('os.path.exists', return_value=True):
            system = VoiceCloningSystem("/models", Mock())
            system.voices_dir = Path("/test/voices")
            system.logger = Mock()
            
            voices = system.list_cloned_voices()
            
            assert len(voices) == 2
            assert voices[0]["voice_id"] == "test_123"
            assert voices[0]["name"] == "Test Voice"

    @patch('app.voice_cloning.shutil.rmtree')
    @patch('os.path.exists', return_value=True)
    def test_delete_cloned_voice(self, mock_exists, mock_rmtree):
        """Test deleting a cloned voice"""
        from app.voice_cloning import VoiceCloningSystem
        
        with patch.object(VoiceCloningSystem, '__init__', return_value=None):
            system = VoiceCloningSystem("/models", Mock())
            system.voices_dir = Path("/test/voices")
            system.logger = Mock()
            system.voice_contexts = {"test_123": "context"}
            
            result = system.delete_cloned_voice("test_123")
            
            assert result == True
            mock_rmtree.assert_called_once()
            assert "test_123" not in system.voice_contexts

    def test_get_speaker_id_for_voice(self):
        """Test getting speaker ID for a voice"""
        from app.voice_cloning import VoiceCloningSystem
        
        with patch.object(VoiceCloningSystem, '__init__', return_value=None):
            system = VoiceCloningSystem("/models", Mock())
            system.base_speaker_id = 1000
            system.voice_id_to_index = {"voice_123": 5}
            
            # Test existing voice
            speaker_id = system.get_speaker_id_for_voice("voice_123")
            assert speaker_id == 1005
            
            # Test new voice
            speaker_id = system.get_speaker_id_for_voice("new_voice")
            assert speaker_id == 1006
            assert system.voice_id_to_index["new_voice"] == 6

    @patch('builtins.open', mock_open())
    @patch('torch.load')
    def test_load_voice_contexts(self, mock_torch_load):
        """Test loading voice contexts from disk"""
        from app.voice_cloning import VoiceCloningSystem
        
        mock_torch_load.return_value = torch.randn(10, 512)
        
        with patch.object(VoiceCloningSystem, '__init__', return_value=None), \
             patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['voice_123']):
            system = VoiceCloningSystem("/models", Mock())
            system.voices_dir = Path("/test/voices")
            system.voice_contexts = {}
            system.voice_id_to_index = {}
            system.logger = Mock()
            
            system._load_voice_contexts()
            
            assert "voice_123" in system.voice_contexts
            assert isinstance(system.voice_contexts["voice_123"], torch.Tensor)