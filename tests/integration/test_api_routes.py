import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import torch
import numpy as np
from app.api.schemas import ResponseFormat


@pytest.fixture
def mock_app_state():
    """Mock app state with necessary components"""
    state = Mock()
    
    # Mock generator
    generator = Mock()
    generator.generate = AsyncMock(return_value=(torch.randn(1, 16000), 16000))
    generator.cleanup = Mock()
    state.generator = generator
    
    # Mock voice systems
    state.voice_cloning_system = Mock()
    state.voice_enhancement_system = Mock()
    state.voice_memory = Mock()
    
    # Mock model info
    state.model_name = "csm-1b"
    state.voices = {
        "alloy": 42,
        "echo": 43,
        "fable": 44,
        "onyx": 45,
        "nova": 46,
        "shimmer": 47
    }
    
    return state


@pytest.fixture
def mock_app(mock_app_state):
    """Create a mock app with test client"""
    with patch('app.main.app') as mock_app_instance:
        # Configure the mock app
        mock_app_instance.state = mock_app_state
        
        # Import the actual app to get routes
        from app.main import app
        app.state = mock_app_state
        
        yield app


@pytest.fixture
def client(mock_app):
    """Create test client"""
    return TestClient(mock_app)


class TestHealthEndpoint:
    def test_health_check(self, client):
        """Test /health endpoint returns correct status"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "voices_loaded" in data
        assert "version" in data


class TestSpeechGeneration:
    @patch('app.api.routes.convert_audio_format')
    async def test_generate_speech_minimal(self, mock_convert, client):
        """Test speech generation with minimal parameters"""
        mock_convert.return_value = b"fake_audio_data"
        
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Hello, world!"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == b"fake_audio_data"

    @patch('app.api.routes.convert_audio_format')
    async def test_generate_speech_all_params(self, mock_convert, client):
        """Test speech generation with all parameters"""
        mock_convert.return_value = b"fake_audio_data"
        
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "csm-1b",
                "input": "Test text",
                "voice": "nova",
                "response_format": "wav",
                "speed": 1.5,
                "temperature": 0.8,
                "topk": 40
            }
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

    def test_generate_speech_empty_input(self, client):
        """Test speech generation with empty input"""
        response = client.post(
            "/v1/audio/speech",
            json={"input": ""}
        )
        
        assert response.status_code == 400
        assert "Empty input text" in response.json()["detail"]

    def test_generate_speech_invalid_voice(self, client, mock_app_state):
        """Test speech generation with invalid voice"""
        # Make voice lookup fail
        mock_app_state.voices = {"alloy": 42}  # Missing requested voice
        
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Test", "voice": "invalid_voice"}
        )
        
        # Should still work as it falls back to speaker_id 42
        assert response.status_code == 200

    @patch('app.api.routes.convert_audio_format')
    async def test_generate_speech_all_formats(self, mock_convert, client):
        """Test all supported audio formats"""
        formats = ["mp3", "opus", "aac", "flac", "wav"]
        content_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/ogg;codecs=opus", 
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav"
        }
        
        for format_type in formats:
            mock_convert.return_value = f"fake_{format_type}_data".encode()
            
            response = client.post(
                "/v1/audio/speech",
                json={
                    "input": "Test",
                    "response_format": format_type
                }
            )
            
            assert response.status_code == 200
            assert response.headers["content-type"] == content_types[format_type]

    @patch('app.api.routes.convert_audio_format')
    async def test_generate_speech_with_speed(self, mock_convert, client):
        """Test speech generation with speed adjustment"""
        mock_convert.return_value = b"fake_audio_data"
        
        # Test various speeds
        speeds = [0.25, 1.0, 2.0, 4.0]
        for speed in speeds:
            response = client.post(
                "/v1/audio/speech",
                json={"input": "Test", "speed": speed}
            )
            assert response.status_code == 200


class TestVoiceEndpoints:
    def test_list_voices(self, client, mock_app_state):
        """Test /v1/audio/voices endpoint"""
        response = client.get("/v1/audio/voices")
        assert response.status_code == 200
        data = response.json()
        
        assert "voices" in data
        voices = data["voices"]
        assert len(voices) >= 6  # At least the standard voices
        
        # Check voice structure
        for voice in voices:
            assert "id" in voice
            assert "name" in voice
            assert "description" in voice

    def test_list_models(self, client):
        """Test /v1/audio/models endpoint"""
        response = client.get("/v1/audio/models")
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        models = data["models"]
        assert len(models) > 0
        
        # Check model structure
        for model in models:
            assert "id" in model
            assert "name" in model
            assert "description" in model


class TestStreamingEndpoint:
    @patch('app.api.routes.StreamingResponse')
    async def test_streaming_speech(self, mock_streaming_response, client):
        """Test streaming speech generation"""
        # Mock the streaming response
        mock_streaming_response.return_value = Mock(status_code=200)
        
        response = client.post(
            "/v1/audio/speech/streaming",
            json={"input": "Stream this text"}
        )
        
        # Verify StreamingResponse was called
        assert mock_streaming_response.called


class TestErrorHandling:
    def test_missing_input_field(self, client):
        """Test request without required input field"""
        response = client.post(
            "/v1/audio/speech",
            json={"voice": "alloy"}  # Missing 'input'
        )
        
        assert response.status_code == 422  # Validation error

    def test_invalid_speed_value(self, client):
        """Test request with invalid speed value"""
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Test", "speed": 5.0}  # Speed > 4.0
        )
        
        assert response.status_code == 422

    def test_invalid_temperature_value(self, client):
        """Test request with invalid temperature value"""
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Test", "temperature": 3.0}  # Temperature > 2.0
        )
        
        assert response.status_code == 422

    @patch('app.api.routes.convert_audio_format')
    async def test_generator_exception(self, mock_convert, client, mock_app_state):
        """Test handling of generator exceptions"""
        # Make generator raise an exception
        mock_app_state.generator.generate.side_effect = Exception("Generator error")
        
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Test"}
        )
        
        assert response.status_code == 500
        assert "Audio generation failed" in response.json()["detail"]


class TestCompatibility:
    def test_openai_compatible_request(self, client):
        """Test that OpenAI-style requests work correctly"""
        # OpenAI-style request
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",  # OpenAI model name
                "input": "Hello from OpenAI format",
                "voice": "alloy"
            }
        )
        
        # Should work even with different model name
        assert response.status_code == 200

    def test_extra_fields_ignored(self, client):
        """Test that extra fields in request are ignored"""
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "Test",
                "voice": "alloy",
                "extra_field": "should be ignored",
                "another_extra": 123
            }
        )
        
        assert response.status_code == 200