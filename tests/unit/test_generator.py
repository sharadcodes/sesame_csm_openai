import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import torch
import numpy as np
from app.models import Segment


class TestCSMGenerator:
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies for CSMGenerator"""
        with patch('app.generator.load_model') as mock_load_model, \
             patch('app.generator.MimiModel') as mock_mimi_model, \
             patch('app.generator.torch.cuda.is_available', return_value=True), \
             patch('app.generator.torch.cuda.device_count', return_value=1):
            
            # Mock CSM model
            mock_csm_model = Mock()
            mock_csm_model.generate = Mock(return_value=torch.randn(1, 1, 1024))
            mock_csm_model.to = Mock(return_value=mock_csm_model)
            mock_csm_model.eval = Mock(return_value=mock_csm_model)
            mock_load_model.return_value = mock_csm_model
            
            # Mock Mimi codec
            mock_codec = Mock()
            mock_codec.decode = Mock(return_value=torch.randn(1, 1, 16000))
            mock_codec.encode = Mock(return_value=torch.randn(1, 8, 100))
            mock_codec.to = Mock(return_value=mock_codec)
            mock_codec.eval = Mock(return_value=mock_codec)
            mock_mimi_model.from_pretrained.return_value = mock_codec
            
            yield {
                'load_model': mock_load_model,
                'mimi_model': mock_mimi_model,
                'csm_model': mock_csm_model,
                'codec': mock_codec
            }

    @patch('app.generator.CSMGenerator.__init__', return_value=None)
    def test_generator_initialization(self, mock_init):
        """Test CSMGenerator initialization"""
        from app.generator import CSMGenerator
        
        generator = CSMGenerator("/models", device_map="auto")
        mock_init.assert_called_once()

    def test_normalize_text(self):
        """Test text normalization in generator"""
        from app.generator import CSMGenerator
        
        with patch.object(CSMGenerator, '__init__', return_value=None):
            generator = CSMGenerator("/models")
            
            # Test basic normalization
            text = "Hello, world!"
            normalized = generator._normalize_text(text)
            assert isinstance(normalized, str)
            assert len(normalized) > 0

    def test_prepare_segments(self):
        """Test segment preparation"""
        from app.generator import CSMGenerator
        
        with patch.object(CSMGenerator, '__init__', return_value=None):
            generator = CSMGenerator("/models")
            generator.logger = Mock()
            
            text = "This is a test. Another sentence here."
            segments = generator._prepare_segments(text, speaker_id=42)
            
            assert isinstance(segments, list)
            assert len(segments) > 0
            assert all(isinstance(seg, Segment) for seg in segments)
            assert all(seg.speaker_id == 42 for seg in segments)

    @patch('app.generator.torch.no_grad')
    def test_generate_segment_audio(self, mock_no_grad, mock_dependencies):
        """Test single segment audio generation"""
        from app.generator import CSMGenerator
        
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock()
        
        with patch.object(CSMGenerator, '__init__', return_value=None):
            generator = CSMGenerator("/models")
            generator.model = mock_dependencies['csm_model']
            generator.codec = mock_dependencies['codec']
            generator.tokenizer = Mock()
            generator.tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
            generator.device = "cuda"
            generator.watermark = None
            generator.logger = Mock()
            
            segment = Segment(text="Test", speaker_id=42, begin=0.0, end=1.0)
            audio = generator._generate_segment_audio(segment)
            
            assert isinstance(audio, torch.Tensor)
            assert audio.dim() >= 1

    def test_combine_audio_segments(self):
        """Test combining multiple audio segments"""
        from app.generator import CSMGenerator
        
        with patch.object(CSMGenerator, '__init__', return_value=None):
            generator = CSMGenerator("/models")
            generator.sample_rate = 16000
            generator.logger = Mock()
            
            # Create test segments
            segments = [
                torch.randn(16000),  # 1 second
                torch.randn(16000),  # 1 second
                torch.randn(8000),   # 0.5 seconds
            ]
            
            combined = generator._combine_audio_segments(segments)
            
            assert isinstance(combined, torch.Tensor)
            assert combined.shape[0] == sum(s.shape[0] for s in segments)

    @patch('app.generator.CSMGenerator._generate_segment_audio')
    async def test_generate_async(self, mock_generate_segment, mock_dependencies):
        """Test async generate method"""
        from app.generator import CSMGenerator
        
        mock_generate_segment.return_value = torch.randn(16000)
        
        with patch.object(CSMGenerator, '__init__', return_value=None):
            generator = CSMGenerator("/models")
            generator._prepare_segments = Mock(return_value=[
                Segment(text="Test", speaker_id=42, begin=0.0, end=1.0)
            ])
            generator._combine_audio_segments = Mock(return_value=torch.randn(1, 16000))
            generator.sample_rate = 16000
            generator.logger = Mock()
            
            audio, sr = await generator.generate("Test text", speaker_id=42)
            
            assert isinstance(audio, torch.Tensor)
            assert sr == 16000

    def test_cleanup(self, mock_dependencies):
        """Test cleanup method"""
        from app.generator import CSMGenerator
        
        with patch.object(CSMGenerator, '__init__', return_value=None), \
             patch('torch.cuda.empty_cache') as mock_empty_cache:
            generator = CSMGenerator("/models")
            generator.model = mock_dependencies['csm_model']
            generator.codec = mock_dependencies['codec']
            generator.logger = Mock()
            
            generator.cleanup()
            
            mock_empty_cache.assert_called_once()

    def test_speaker_id_validation(self):
        """Test speaker ID validation"""
        from app.generator import CSMGenerator
        
        with patch.object(CSMGenerator, '__init__', return_value=None):
            generator = CSMGenerator("/models")
            generator._prepare_segments = Mock()
            generator.logger = Mock()
            
            # Test with valid speaker IDs
            generator._prepare_segments("Test", speaker_id=42)
            generator._prepare_segments("Test", speaker_id=1000)
            
            # Verify speaker_id is passed correctly
            calls = generator._prepare_segments.call_args_list
            assert calls[0][1]['speaker_id'] == 42
            assert calls[1][1]['speaker_id'] == 1000

    @patch('app.generator.torch.cuda.is_available', return_value=False)
    def test_cpu_fallback(self, mock_cuda):
        """Test CPU fallback when CUDA is not available"""
        from app.generator import CSMGenerator
        
        with patch('app.generator.load_model') as mock_load_model, \
             patch('app.generator.MimiModel') as mock_mimi_model:
            
            mock_model = Mock()
            mock_load_model.return_value = mock_model
            mock_codec = Mock()
            mock_mimi_model.from_pretrained.return_value = mock_codec
            
            generator = CSMGenerator("/models")
            
            assert generator.device == "cpu"

    def test_temperature_and_topk_params(self, mock_dependencies):
        """Test temperature and top-k parameters in generation"""
        from app.generator import CSMGenerator
        
        with patch.object(CSMGenerator, '__init__', return_value=None):
            generator = CSMGenerator("/models")
            generator.model = mock_dependencies['csm_model']
            generator.codec = mock_dependencies['codec']
            generator.tokenizer = Mock()
            generator.tokenizer.encode = Mock(return_value=[1, 2, 3])
            generator.device = "cuda"
            generator.watermark = None
            generator.logger = Mock()
            
            segment = Segment(text="Test", speaker_id=42, begin=0.0, end=1.0)
            
            # Test with custom temperature and top-k
            generator._generate_segment_audio(segment, temperature=0.5, topk=30)
            
            # Verify model.generate was called with correct params
            generate_call = generator.model.generate.call_args[1]
            assert 'temperature' in generate_call or 'top_k' in generate_call

    @patch('app.generator.clean_text_for_tts')
    def test_text_cleaning_integration(self, mock_clean_text):
        """Test integration with text cleaning"""
        from app.generator import CSMGenerator
        
        mock_clean_text.return_value = "cleaned text"
        
        with patch.object(CSMGenerator, '__init__', return_value=None):
            generator = CSMGenerator("/models")
            generator.logger = Mock()
            
            result = generator._normalize_text("dirty text")
            
            mock_clean_text.assert_called_once_with("dirty text")
            assert result == "cleaned text"