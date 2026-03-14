from src.amazon_sentiment.labeling import label_from_rating


def test_label_from_rating_with_neutral():
    assert label_from_rating(1.0, include_neutral=True) == "negative"
    assert label_from_rating(3.0, include_neutral=True) == "neutre"
    assert label_from_rating(5.0, include_neutral=True) == "positive"


def test_label_from_rating_without_neutral():
    assert label_from_rating(3.0, include_neutral=False) is None
