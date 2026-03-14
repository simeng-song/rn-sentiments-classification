from src.amazon_sentiment.preprocessing import clean_text


def test_clean_text_basic_normalization():
    cleaned = clean_text("This BOOK is, honestly, very good!")
    assert "book" in cleaned
    assert "good" in cleaned
    assert "," not in cleaned


def test_clean_text_handles_none():
    assert clean_text(None) == ""
