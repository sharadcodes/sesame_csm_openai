import pytest
from pydantic import ValidationError
from app.api.schemas import SpeechRequest, TTSRequest, ResponseFormat, Voice


class TestSpeechRequest:
    def test_minimal_valid_request(self):
        """Test creating a speech request with minimal required fields"""
        request = SpeechRequest(input="Hello, world!")
        assert request.input == "Hello, world!"
        assert request.model == "csm-1b"
        assert request.voice == "alloy"
        assert request.response_format == ResponseFormat.mp3
        assert request.speed == 1.0
        assert request.max_audio_length_ms == 90000
        assert request.temperature == 0.9
        assert request.topk == 50

    def test_full_valid_request(self):
        """Test creating a speech request with all fields"""
        request = SpeechRequest(
            model="csm-1b",
            input="Test text",
            voice="echo",
            response_format=ResponseFormat.wav,
            speed=1.5,
            max_audio_length_ms=60000,
            temperature=0.7,
            topk=30
        )
        assert request.model == "csm-1b"
        assert request.input == "Test text"
        assert request.voice == "echo"
        assert request.response_format == ResponseFormat.wav
        assert request.speed == 1.5
        assert request.max_audio_length_ms == 60000
        assert request.temperature == 0.7
        assert request.topk == 30

    def test_missing_required_input(self):
        """Test that missing input field raises validation error"""
        with pytest.raises(ValidationError) as exc_info:
            SpeechRequest()
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('input',)
        assert errors[0]['type'] == 'missing'

    def test_speed_validation(self):
        """Test speed parameter validation bounds"""
        # Valid speeds
        valid_speeds = [0.25, 1.0, 2.0, 4.0]
        for speed in valid_speeds:
            request = SpeechRequest(input="test", speed=speed)
            assert request.speed == speed

        # Invalid speeds
        invalid_speeds = [0.1, 4.1, -1.0, 5.0]
        for speed in invalid_speeds:
            with pytest.raises(ValidationError):
                SpeechRequest(input="test", speed=speed)

    def test_temperature_validation(self):
        """Test temperature parameter validation bounds"""
        # Valid temperatures
        valid_temps = [0.0, 0.5, 1.0, 2.0]
        for temp in valid_temps:
            request = SpeechRequest(input="test", temperature=temp)
            assert request.temperature == temp

        # Invalid temperatures
        invalid_temps = [-0.1, 2.1, 3.0]
        for temp in invalid_temps:
            with pytest.raises(ValidationError):
                SpeechRequest(input="test", temperature=temp)

    def test_topk_validation(self):
        """Test topk parameter validation bounds"""
        # Valid topk values
        valid_topks = [1, 50, 100]
        for topk in valid_topks:
            request = SpeechRequest(input="test", topk=topk)
            assert request.topk == topk

        # Invalid topk values
        invalid_topks = [0, -1, 101, 200]
        for topk in invalid_topks:
            with pytest.raises(ValidationError):
                SpeechRequest(input="test", topk=topk)

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored due to extra='ignore' config"""
        request = SpeechRequest(
            input="test",
            extra_field="should be ignored",
            another_extra=123
        )
        assert request.input == "test"
        assert not hasattr(request, 'extra_field')
        assert not hasattr(request, 'another_extra')

    def test_voice_accepts_any_string(self):
        """Test that voice parameter accepts any string value"""
        voices = ["alloy", "echo", "custom_voice", "cloned_voice_123", "any-string-here"]
        for voice in voices:
            request = SpeechRequest(input="test", voice=voice)
            assert request.voice == voice

    def test_response_format_enum(self):
        """Test all valid response format options"""
        formats = [ResponseFormat.mp3, ResponseFormat.opus, ResponseFormat.aac, 
                   ResponseFormat.flac, ResponseFormat.wav]
        for fmt in formats:
            request = SpeechRequest(input="test", response_format=fmt)
            assert request.response_format == fmt

    def test_response_format_string_conversion(self):
        """Test response format string conversion"""
        request = SpeechRequest(input="test", response_format="wav")
        assert request.response_format == ResponseFormat.wav

    def test_invalid_response_format(self):
        """Test invalid response format raises validation error"""
        with pytest.raises(ValidationError):
            SpeechRequest(input="test", response_format="invalid_format")


class TestTTSRequest:
    def test_tts_request_inheritance(self):
        """Test that TTSRequest is properly inherited from SpeechRequest"""
        request = TTSRequest(input="Test text")
        assert isinstance(request, SpeechRequest)
        assert request.input == "Test text"
        assert request.model == "csm-1b"

    def test_tts_request_backwards_compatibility(self):
        """Test TTSRequest maintains all SpeechRequest functionality"""
        request = TTSRequest(
            input="Test",
            voice="nova",
            speed=2.0,
            temperature=1.5
        )
        assert request.input == "Test"
        assert request.voice == "nova"
        assert request.speed == 2.0
        assert request.temperature == 1.5


class TestVoice:
    def test_voice_is_string_type(self):
        """Test that Voice is essentially a string type"""
        voice = Voice("test_voice")
        assert isinstance(voice, str)
        assert voice == "test_voice"

    def test_voice_allows_any_value(self):
        """Test that Voice accepts any string value without restriction"""
        test_values = [
            "alloy",
            "echo",
            "custom_voice_123",
            "user-uploaded-voice",
            "any-string-at-all",
            "12345",
            "voice-with-special-chars!@#"
        ]
        for value in test_values:
            voice = Voice(value)
            assert voice == value