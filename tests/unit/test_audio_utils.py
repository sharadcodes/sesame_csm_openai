import pytest
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock, call
from app.utils.audio_utils import convert_audio_format


class TestConvertAudioFormat:
    @pytest.fixture
    def sample_audio_tensor(self):
        """Create a sample audio tensor for testing"""
        # Create a simple sine wave
        sample_rate = 16000
        duration = 0.1  # 100ms
        t = torch.linspace(0, duration, int(sample_rate * duration))
        frequency = 440  # A4 note
        audio = torch.sin(2 * torch.pi * frequency * t)
        return audio, sample_rate

    @patch('app.utils.audio_utils.torchaudio.save')
    @patch('app.utils.audio_utils.ffmpeg.input')
    @patch('builtins.open', create=True)
    @patch('os.unlink')
    def test_convert_to_mp3(self, mock_unlink, mock_open, mock_ffmpeg_input, mock_torchaudio_save, sample_audio_tensor):
        """Test converting audio tensor to MP3 format"""
        audio_tensor, sample_rate = sample_audio_tensor
        
        # Mock file operations
        mock_file = MagicMock()
        mock_file.read.return_value = b"fake_mp3_data"
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock ffmpeg chain
        mock_output = MagicMock()
        mock_ffmpeg_input.return_value.output.return_value = mock_output
        
        # Call the function
        result = convert_audio_format(audio_tensor, sample_rate, format="mp3", bit_rate="192k")
        
        # Verify torchaudio.save was called correctly
        assert mock_torchaudio_save.called
        saved_tensor = mock_torchaudio_save.call_args[0][1]
        assert saved_tensor.shape[0] == 1  # Should add channel dimension
        
        # Verify ffmpeg was called with correct parameters
        mock_ffmpeg_input.assert_called_once()
        mock_ffmpeg_input.return_value.output.assert_called_once()
        output_call_args = mock_ffmpeg_input.return_value.output.call_args
        assert output_call_args[1]['format'] == 'mp3'
        assert output_call_args[1]['audio_bitrate'] == '192k'
        mock_output.run.assert_called_once_with(quiet=True)
        
        # Verify result
        assert result == b"fake_mp3_data"
        
        # Verify cleanup
        assert mock_unlink.call_count == 2  # Both temp files should be deleted

    @patch('app.utils.audio_utils.torchaudio.save')
    @patch('app.utils.audio_utils.ffmpeg.input')
    @patch('builtins.open', create=True)
    @patch('os.unlink')
    def test_convert_to_opus(self, mock_unlink, mock_open, mock_ffmpeg_input, mock_torchaudio_save, sample_audio_tensor):
        """Test converting audio tensor to Opus format"""
        audio_tensor, sample_rate = sample_audio_tensor
        
        # Mock file operations
        mock_file = MagicMock()
        mock_file.read.return_value = b"fake_opus_data"
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock ffmpeg chain
        mock_output = MagicMock()
        mock_ffmpeg_input.return_value.output.return_value = mock_output
        
        # Call the function
        result = convert_audio_format(audio_tensor, sample_rate, format="opus")
        
        # Verify ffmpeg was called correctly for opus
        mock_ffmpeg_input.return_value.output.assert_called_once()
        output_call_args = mock_ffmpeg_input.return_value.output.call_args
        assert output_call_args[1]['format'] == 'opus'
        assert 'audio_bitrate' not in output_call_args[1]  # Opus doesn't use bit_rate parameter
        
        assert result == b"fake_opus_data"

    @patch('app.utils.audio_utils.torchaudio.save')
    @patch('builtins.open', create=True)
    @patch('os.unlink')
    def test_convert_to_wav(self, mock_unlink, mock_open, mock_torchaudio_save, sample_audio_tensor):
        """Test converting audio tensor to WAV format (no ffmpeg needed)"""
        audio_tensor, sample_rate = sample_audio_tensor
        
        # Mock file operations
        mock_file = MagicMock()
        mock_file.read.return_value = b"fake_wav_data"
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Call the function
        result = convert_audio_format(audio_tensor, sample_rate, format="wav")
        
        # Verify torchaudio.save was called
        assert mock_torchaudio_save.called
        
        # For WAV format, ffmpeg should not be used
        assert result == b"fake_wav_data"

    @patch('app.utils.audio_utils.torchaudio.save')
    @patch('app.utils.audio_utils.ffmpeg.input')
    @patch('builtins.open', create=True)
    @patch('os.unlink')
    def test_convert_multi_channel_audio(self, mock_unlink, mock_open, mock_ffmpeg_input, mock_torchaudio_save):
        """Test converting multi-channel audio tensor"""
        # Create stereo audio
        audio_tensor = torch.randn(2, 16000)  # 2 channels, 1 second
        sample_rate = 16000
        
        # Mock file operations
        mock_file = MagicMock()
        mock_file.read.return_value = b"fake_audio_data"
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock ffmpeg chain
        mock_output = MagicMock()
        mock_ffmpeg_input.return_value.output.return_value = mock_output
        
        # Call the function
        result = convert_audio_format(audio_tensor, sample_rate, format="mp3")
        
        # Verify torchaudio.save was called with correct tensor shape
        saved_tensor = mock_torchaudio_save.call_args[0][1]
        assert saved_tensor.shape == (2, 16000)  # Should preserve stereo

    @patch('app.utils.audio_utils.torchaudio.save')
    @patch('app.utils.audio_utils.ffmpeg.input')
    @patch('builtins.open', create=True)
    @patch('os.unlink')
    def test_convert_all_supported_formats(self, mock_unlink, mock_open, mock_ffmpeg_input, mock_torchaudio_save, sample_audio_tensor):
        """Test converting to all supported formats"""
        audio_tensor, sample_rate = sample_audio_tensor
        formats = ["mp3", "opus", "aac", "flac", "wav"]
        
        for format_type in formats:
            # Reset mocks
            mock_ffmpeg_input.reset_mock()
            mock_torchaudio_save.reset_mock()
            
            # Mock file operations
            mock_file = MagicMock()
            mock_file.read.return_value = f"fake_{format_type}_data".encode()
            mock_open.return_value.__enter__.return_value = mock_file
            
            # Mock ffmpeg chain
            mock_output = MagicMock()
            mock_ffmpeg_input.return_value.output.return_value = mock_output
            
            # Call the function
            result = convert_audio_format(audio_tensor, sample_rate, format=format_type)
            
            # Verify result
            assert result == f"fake_{format_type}_data".encode()
            
            # Verify ffmpeg usage (except for WAV)
            if format_type != "wav":
                mock_ffmpeg_input.return_value.output.assert_called_once()

    @patch('app.utils.audio_utils.torchaudio.save')
    @patch('app.utils.audio_utils.ffmpeg.input')
    @patch('builtins.open', create=True)
    @patch('os.unlink')
    @patch('os.path.exists')
    def test_cleanup_on_exception(self, mock_exists, mock_unlink, mock_open, mock_ffmpeg_input, mock_torchaudio_save, sample_audio_tensor):
        """Test that temporary files are cleaned up even on exception"""
        audio_tensor, sample_rate = sample_audio_tensor
        
        # Make ffmpeg raise an exception
        mock_ffmpeg_input.side_effect = Exception("FFmpeg error")
        mock_exists.return_value = True
        
        # Call should raise exception
        with pytest.raises(Exception, match="FFmpeg error"):
            convert_audio_format(audio_tensor, sample_rate, format="mp3")
        
        # Verify cleanup was still called
        assert mock_unlink.call_count == 2  # Both temp files

    @patch('app.utils.audio_utils.torchaudio.save')
    @patch('app.utils.audio_utils.ffmpeg.input')
    @patch('builtins.open', create=True)
    @patch('os.unlink')
    def test_default_bit_rate(self, mock_unlink, mock_open, mock_ffmpeg_input, mock_torchaudio_save, sample_audio_tensor):
        """Test default bit rate is used when not specified"""
        audio_tensor, sample_rate = sample_audio_tensor
        
        # Mock file operations
        mock_file = MagicMock()
        mock_file.read.return_value = b"fake_mp3_data"
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock ffmpeg chain
        mock_output = MagicMock()
        mock_ffmpeg_input.return_value.output.return_value = mock_output
        
        # Call without specifying bit_rate
        result = convert_audio_format(audio_tensor, sample_rate, format="mp3")
        
        # Verify default bit rate was used
        output_call_args = mock_ffmpeg_input.return_value.output.call_args
        assert output_call_args[1]['audio_bitrate'] == '128k'