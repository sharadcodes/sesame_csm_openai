import pytest
from app.text_normalizer import TextNormalizer, clean_text_for_tts


class TestTextNormalizer:
    def test_normalize_empty_text(self):
        """Test normalizing empty or None text"""
        assert TextNormalizer.normalize_text("") == ""
        assert TextNormalizer.normalize_text(None) == None

    def test_normalize_contractions(self):
        """Test contraction normalization"""
        test_cases = [
            ("I don't know", "I dont know."),
            ("She's happy", "Shes happy."),
            ("They're coming", "Theyre coming."),
            ("It's a beautiful day", "Its a beautiful day."),
            ("We won't go", "We wont go."),
            ("I've seen it", "Ive seen it."),
            ("You'll be fine", "Youll be fine."),
        ]
        for input_text, expected in test_cases:
            assert TextNormalizer.normalize_text(input_text) == expected

    def test_normalize_contractions_case_insensitive(self):
        """Test that contractions are normalized regardless of case"""
        test_cases = [
            ("I DON'T KNOW", "I DONT KNOW."),
            ("SHE'S happy", "SHES happy."),
            ("THEY'RE coming", "THEYRE coming."),
        ]
        for input_text, expected in test_cases:
            assert TextNormalizer.normalize_text(input_text) == expected

    def test_normalize_abbreviations(self):
        """Test abbreviation expansion"""
        test_cases = [
            ("Mr. Smith", "Mister Smith."),
            ("Dr. Jones", "Doctor Jones."),
            ("123 Main St.", "123 Main Street."),
            ("First Ave.", "First Avenue."),
            ("e.g. this example", "for example this example."),
            ("i.e. that is correct", "that is that is correct."),
        ]
        for input_text, expected in test_cases:
            assert TextNormalizer.normalize_text(input_text) == expected

    def test_normalize_numbers(self):
        """Test number to word conversion"""
        test_cases = [
            ("I have 3 apples", "I have three apples."),
            ("She is 20 years old", "She is twenty years old."),
            ("The year 2024", "The year 2024."),  # Year numbers not converted
            ("10 items in stock", "ten items in stock."),
            ("100 percent", "one hundred percent."),
        ]
        for input_text, expected in test_cases:
            assert TextNormalizer.normalize_text(input_text) == expected

    def test_normalize_symbols(self):
        """Test symbol replacement"""
        test_cases = [
            ("Tom & Jerry", "Tom and Jerry."),
            ("50% off", "fifty percent off."),
            ("Email me @ test", "Email me at test."),
            ("Item #5", "Item number five."),
            ("$10 dollars", "dollar ten dollars."),
            ("€50 euros", "euro fifty euros."),
            ("£20 pounds", "pound twenty pounds."),
            ("¥100 yen", "yen one hundred yen."),
        ]
        for input_text, expected in test_cases:
            assert TextNormalizer.normalize_text(input_text) == expected

    def test_normalize_dates(self):
        """Test date format normalization"""
        test_cases = [
            ("Meeting on 12/25/2024", "Meeting on 12 25 2024."),
            ("Born 01/01/2000", "Born 01 01 2000."),
            ("Due by 6/7/2023", "Due by 6 7 2023."),
        ]
        for input_text, expected in test_cases:
            assert TextNormalizer.normalize_text(input_text) == expected

    def test_remove_voice_instructions(self):
        """Test removal of voice instructions in square brackets"""
        test_cases = [
            ("[Whispered] Hello there", "Hello there."),
            ("Say [loudly] this word", "Say this word."),
            ("[Happy tone] Great job!", "Great job!"),
            ("Normal text [instruction] more text", "Normal text more text."),
        ]
        for input_text, expected in test_cases:
            assert TextNormalizer.normalize_text(input_text) == expected

    def test_fix_excessive_spaces(self):
        """Test fixing multiple spaces"""
        test_cases = [
            ("Hello    world", "Hello world."),
            ("Too  many   spaces", "Too many spaces."),
            ("  Leading and trailing  ", "Leading and trailing."),
            ("Tab\ttab\tspaces", "Tab tab spaces."),
        ]
        for input_text, expected in test_cases:
            assert TextNormalizer.normalize_text(input_text) == expected

    def test_ensure_ending_punctuation(self):
        """Test that text ends with proper punctuation"""
        test_cases = [
            ("Hello world", "Hello world."),
            ("Already has period.", "Already has period."),
            ("Question?", "Question?"),
            ("Exclamation!", "Exclamation!"),
            ("Semicolon;", "Semicolon;"),
            ("Colon:", "Colon:"),
            ("Comma,", "Comma,"),
        ]
        for input_text, expected in test_cases:
            assert TextNormalizer.normalize_text(input_text) == expected

    def test_complex_normalization(self):
        """Test complex text with multiple normalization needs"""
        input_text = "[Excited] I don't think Mr. Johnson & his team achieved 90% of their $1000 goal on 12/25/2023"
        expected = "I dont think Mister Johnson and his team achieved ninety percent of their dollar one thousand goal on 12 25 2023."
        assert TextNormalizer.normalize_text(input_text) == expected

    def test_split_into_sentences(self):
        """Test splitting text into sentences"""
        text = "This is sentence one. This is sentence two! Is this sentence three? Yes it is."
        sentences = TextNormalizer.split_into_sentences(text)
        assert len(sentences) == 4
        assert sentences[0] == "This is sentence one."
        assert sentences[1] == "This is sentence two!"
        assert sentences[2] == "Is this sentence three?"
        assert sentences[3] == "Yes it is."

    def test_split_into_sentences_empty(self):
        """Test splitting empty text"""
        assert TextNormalizer.split_into_sentences("") == []

    def test_split_into_sentences_single(self):
        """Test splitting single sentence"""
        sentences = TextNormalizer.split_into_sentences("Just one sentence")
        assert len(sentences) == 1
        assert sentences[0] == "Just one sentence."

    def test_split_into_sentences_with_normalization(self):
        """Test that split_into_sentences also normalizes text"""
        text = "I don't know. She's here!"
        sentences = TextNormalizer.split_into_sentences(text)
        assert sentences[0] == "I dont know."
        assert sentences[1] == "Shes here!"


class TestCleanTextForTTS:
    def test_clean_text_for_tts(self):
        """Test the clean_text_for_tts convenience function"""
        input_text = "I don't think it's ready"
        expected = "I dont think its ready."
        assert clean_text_for_tts(input_text) == expected

    def test_clean_text_for_tts_empty(self):
        """Test cleaning empty text"""
        assert clean_text_for_tts("") == ""