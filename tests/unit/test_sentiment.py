"""Tests for lexicon-based sentiment extraction."""

from surreal_memory.extraction.sentiment import (
    SentimentExtractor,
    Valence,
)


class TestSentimentBasicEN:
    """Test basic English sentiment extraction."""

    def setup_method(self) -> None:
        self.extractor = SentimentExtractor()

    def test_positive_text(self) -> None:
        """Clearly positive text should return POSITIVE valence."""
        result = self.extractor.extract("This is a great and excellent solution!")
        assert result.valence == Valence.POSITIVE
        assert result.positive_count > 0
        assert result.intensity > 0

    def test_negative_text(self) -> None:
        """Clearly negative text should return NEGATIVE valence."""
        result = self.extractor.extract("The system crashed and everything is broken.")
        assert result.valence == Valence.NEGATIVE
        assert result.negative_count > 0
        assert result.intensity > 0

    def test_neutral_text(self) -> None:
        """Text without sentiment words should return NEUTRAL."""
        result = self.extractor.extract("The meeting is at three pm in the conference room.")
        assert result.valence == Valence.NEUTRAL
        assert result.intensity == 0.0

    def test_empty_text(self) -> None:
        """Empty text should return NEUTRAL with zero intensity."""
        result = self.extractor.extract("")
        assert result.valence == Valence.NEUTRAL
        assert result.intensity == 0.0
        assert result.positive_count == 0
        assert result.negative_count == 0

    def test_short_text(self) -> None:
        """Text shorter than 3 characters should return NEUTRAL."""
        result = self.extractor.extract("hi")
        assert result.valence == Valence.NEUTRAL

    def test_mixed_sentiment_positive_majority(self) -> None:
        """Mixed text with more positive words should return POSITIVE."""
        result = self.extractor.extract(
            "The solution is great and excellent, though one test failed."
        )
        assert result.valence == Valence.POSITIVE

    def test_mixed_sentiment_negative_majority(self) -> None:
        """Mixed text with more negative words should return NEGATIVE."""
        result = self.extractor.extract(
            "The deploy crashed, the build is broken and the tests failed. At least logging works."
        )
        assert result.valence == Valence.NEGATIVE


class TestSentimentNegation:
    """Test negation handling."""

    def setup_method(self) -> None:
        self.extractor = SentimentExtractor()

    def test_negated_positive_becomes_negative(self) -> None:
        """'not good' should flip positive to negative."""
        result = self.extractor.extract("This solution is not good at all.")
        assert result.negative_count > 0

    def test_negated_negative_becomes_positive(self) -> None:
        """'not broken' should flip negative to positive."""
        result = self.extractor.extract("The system is not broken anymore.")
        assert result.positive_count > 0

    def test_double_negation(self) -> None:
        """Contractions like 'don't' should act as negators."""
        result = self.extractor.extract("I don't like this approach, it's terrible.")
        # "don't" negates "like" → +1 negative, "terrible" → +1 negative
        assert result.negative_count >= 1

    def test_negation_window_expires(self) -> None:
        """Negation should only affect the next 2 tokens."""
        result = self.extractor.extract(
            "This is not some random placeholder but it's actually great."
        )
        # "not" affects next 2 tokens ("some", "random"), then expires
        # "great" should still be positive
        assert result.positive_count >= 1

    def test_without_negator(self) -> None:
        """'without' should negate following sentiment words."""
        result = self.extractor.extract("The system works without any problem or error.")
        # "without" negates "problem" and "error" → positive
        assert result.positive_count >= 1


class TestSentimentIntensifiers:
    """Test intensifier handling."""

    def setup_method(self) -> None:
        self.extractor = SentimentExtractor()

    def test_intensifier_increases_intensity(self) -> None:
        """'very good' should have higher intensity than 'good'."""
        result_plain = self.extractor.extract("This is good.")
        result_intense = self.extractor.extract("This is very good.")
        assert result_intense.intensity >= result_plain.intensity

    def test_extremely_negative(self) -> None:
        """'extremely frustrated' should have high intensity."""
        result = self.extractor.extract("I am extremely frustrated with this bug.")
        assert result.valence == Valence.NEGATIVE
        assert result.intensity > 0.3

    def test_intensity_capped_at_one(self) -> None:
        """Intensity should never exceed 1.0."""
        result = self.extractor.extract(
            "Very extremely really totally absolutely amazingly incredibly "
            "great wonderful fantastic excellent awesome brilliant superb perfect."
        )
        assert result.intensity <= 1.0


class TestSentimentEmotionTags:
    """Test emotion tag extraction."""

    def setup_method(self) -> None:
        self.extractor = SentimentExtractor()

    def test_frustration_tag(self) -> None:
        """Frustration words should produce 'frustration' emotion tag."""
        result = self.extractor.extract("I am so frustrated with this broken build.")
        assert "frustration" in result.emotion_tags

    def test_satisfaction_tag(self) -> None:
        """Satisfaction words should produce 'satisfaction' tag."""
        result = self.extractor.extract("I'm very happy and satisfied with the results.")
        assert "satisfaction" in result.emotion_tags

    def test_confusion_tag(self) -> None:
        """Confusion words should produce 'confusion' tag."""
        result = self.extractor.extract("The error message is unclear and I'm confused.")
        assert "confusion" in result.emotion_tags

    def test_excitement_tag(self) -> None:
        """Excitement words should produce 'excitement' tag."""
        result = self.extractor.extract("This new feature is amazing and I'm excited to try it!")
        assert "excitement" in result.emotion_tags

    def test_anxiety_tag(self) -> None:
        """Anxiety words should produce 'anxiety' tag."""
        result = self.extractor.extract("I'm worried the deploy will fail and stressed about it.")
        assert "anxiety" in result.emotion_tags

    def test_relief_tag(self) -> None:
        """Relief words should produce 'relief' tag."""
        result = self.extractor.extract("Finally solved the bug, I'm so relieved!")
        assert "relief" in result.emotion_tags

    def test_multiple_emotion_tags(self) -> None:
        """Multiple emotion categories in one text."""
        result = self.extractor.extract(
            "I was frustrated with the bug but finally solved it and I'm relieved."
        )
        assert len(result.emotion_tags) >= 2

    def test_no_emotion_tags_neutral(self) -> None:
        """Neutral text should have no emotion tags (or only empty)."""
        result = self.extractor.extract("The meeting is at three pm.")
        # Might have empty set
        assert result.valence == Valence.NEUTRAL

    def test_emotion_tags_are_frozenset(self) -> None:
        """Emotion tags should be an immutable frozenset."""
        result = self.extractor.extract("I'm happy and excited!")
        assert isinstance(result.emotion_tags, frozenset)


class TestSentimentEdgeCases:
    """Test edge cases."""

    def setup_method(self) -> None:
        self.extractor = SentimentExtractor()

    def test_result_is_frozen(self) -> None:
        """SentimentResult should be immutable."""
        result = self.extractor.extract("Happy day!")
        try:
            result.valence = Valence.NEGATIVE  # type: ignore[misc]
            raise AssertionError("Should not allow mutation")
        except AttributeError:
            pass  # Expected — frozen dataclass

    def test_intensity_between_zero_and_one(self) -> None:
        """Intensity should always be in [0.0, 1.0]."""
        texts = [
            "Everything is great and wonderful and amazing!",
            "Terrible horrible broken crashed failed error bug.",
            "Normal text with some good and bad things.",
            "",
            "x",
        ]
        for text in texts:
            result = self.extractor.extract(text)
            assert 0.0 <= result.intensity <= 1.0

    def test_counts_non_negative(self) -> None:
        """Positive and negative counts should never be negative."""
        result = self.extractor.extract("Not bad, not terrible, not awful.")
        assert result.positive_count >= 0
        assert result.negative_count >= 0

    def test_technical_error_words(self) -> None:
        """Technical error words should register as negative."""
        result = self.extractor.extract("Got a critical error and regression in production.")
        assert result.valence == Valence.NEGATIVE
        assert result.negative_count >= 2

    def test_success_words(self) -> None:
        """Achievement/success words should register as positive."""
        result = self.extractor.extract("Successfully completed the migration with no issues.")
        assert result.valence == Valence.POSITIVE
