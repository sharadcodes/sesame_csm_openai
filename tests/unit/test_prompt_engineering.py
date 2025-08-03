import pytest
from app.prompt_engineering import (
    VOICE_STYLES,
    initialize_templates,
    split_into_segments,
    format_text_for_voice
)


class TestVoiceStyles:
    def test_voice_styles_structure(self):
        """Test that all voice styles have required fields"""
        required_fields = ["adjectives", "characteristics", "speaking_style"]
        
        for voice, style in VOICE_STYLES.items():
            for field in required_fields:
                assert field in style, f"Voice '{voice}' missing field '{field}'"
                assert style[field], f"Voice '{voice}' has empty '{field}'"
            
            # Check that adjectives and characteristics are lists
            assert isinstance(style["adjectives"], list)
            assert isinstance(style["characteristics"], list)
            assert len(style["adjectives"]) >= 3
            assert len(style["characteristics"]) >= 3
            
            # Check speaking_style is a string
            assert isinstance(style["speaking_style"], str)

    def test_all_standard_voices_present(self):
        """Test that all standard OpenAI voices are defined"""
        standard_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        for voice in standard_voices:
            assert voice in VOICE_STYLES

    def test_custom_voice_present(self):
        """Test that custom voice template is present"""
        assert "custom" in VOICE_STYLES


class TestInitializeTemplates:
    def test_initialize_templates(self):
        """Test template initialization"""
        result = initialize_templates()
        assert result == VOICE_STYLES
        assert isinstance(result, dict)
        assert len(result) > 0


class TestSplitIntoSegments:
    def test_split_empty_text(self):
        """Test splitting empty text"""
        assert split_into_segments("") == [""]
        assert split_into_segments(None) == [None]

    def test_split_short_text(self):
        """Test text shorter than max_chars returns as single segment"""
        text = "This is a short text."
        segments = split_into_segments(text, max_chars=100)
        assert len(segments) == 1
        assert segments[0] == text

    def test_split_by_sentences(self):
        """Test splitting text by sentences"""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        segments = split_into_segments(text, max_chars=30)
        assert len(segments) >= 2
        # Each segment should be <= max_chars
        for segment in segments:
            assert len(segment) <= 30

    def test_split_long_sentence(self):
        """Test splitting a single long sentence"""
        text = "This is a very long sentence that contains many words and will definitely exceed our maximum character limit for a single segment."
        segments = split_into_segments(text, max_chars=50)
        assert len(segments) > 1
        for segment in segments:
            assert len(segment) <= 50

    def test_split_by_phrases(self):
        """Test splitting by phrases when sentences are too long"""
        text = "This is the first part, and this is the second part; finally this is the third part: all together now."
        segments = split_into_segments(text, max_chars=40)
        assert len(segments) >= 2
        for segment in segments:
            assert len(segment) <= 40

    def test_split_preserves_punctuation(self):
        """Test that splitting preserves punctuation"""
        text = "Question? Exclamation! Statement."
        segments = split_into_segments(text, max_chars=15)
        # Check that punctuation is preserved
        all_text = " ".join(segments)
        assert "?" in all_text
        assert "!" in all_text
        assert "." in all_text

    def test_split_handles_multiple_spaces(self):
        """Test handling of multiple spaces"""
        text = "Text  with   multiple    spaces."
        segments = split_into_segments(text, max_chars=50)
        assert len(segments) == 1
        # Should normalize spaces
        assert "  " not in segments[0]

    def test_split_very_long_word(self):
        """Test handling of words longer than max_chars"""
        text = "This has a verylongwordthatexceedsthemaximumcharacterlimitforsegmentation in it."
        segments = split_into_segments(text, max_chars=20)
        # Should handle gracefully without breaking
        assert len(segments) > 1

    def test_split_complex_text(self):
        """Test splitting complex multi-sentence text"""
        text = ("This is the first sentence. Here's another one that's a bit longer. "
                "Now we have a really long sentence that will definitely need to be "
                "split because it exceeds our character limit. Short one. And finally, "
                "this is the last sentence of our test text.")
        segments = split_into_segments(text, max_chars=80)
        
        # Should create multiple segments
        assert len(segments) > 2
        
        # Each segment should respect the limit
        for segment in segments:
            assert len(segment) <= 80
            
        # All text should be preserved
        combined = " ".join(segments)
        # Account for potential space normalization
        assert len(combined.replace("  ", " ")) >= len(text.replace("  ", " ")) - 10


class TestFormatTextForVoice:
    def test_format_text_returns_unchanged(self):
        """Test that format_text_for_voice returns text unchanged"""
        # Based on the code, this function now just returns the text as-is
        text = "This is test text."
        for voice in VOICE_STYLES.keys():
            result = format_text_for_voice(text, voice)
            assert result == text

    def test_format_text_with_segments(self):
        """Test formatting with segment information"""
        text = "Segment text"
        result = format_text_for_voice(text, "alloy", segment_index=1, total_segments=3)
        # Should still return unchanged text
        assert result == text

    def test_format_text_all_voices(self):
        """Test formatting works for all defined voices"""
        text = "Test all voices"
        for voice_name in VOICE_STYLES.keys():
            result = format_text_for_voice(text, voice_name)
            assert result == text
            assert isinstance(result, str)

    def test_format_text_unknown_voice(self):
        """Test formatting with unknown voice name"""
        text = "Unknown voice test"
        # Should handle gracefully and return text unchanged
        result = format_text_for_voice(text, "unknown_voice")
        assert result == text

    def test_format_text_special_characters(self):
        """Test formatting preserves special characters"""
        text = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        result = format_text_for_voice(text, "nova")
        assert result == text