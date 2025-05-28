import pytest
from app.utils.text_cleaner import TextCleaner

@pytest.fixture
def text_cleaner():
    return TextCleaner()

def test_clean_text_basic(text_cleaner):
    text = "Hello! This is a test #hashtag @mention http://example.com"
    cleaned = text_cleaner.clean_text(text)
    assert "http" not in cleaned
    assert "#" not in cleaned
    assert "@" not in cleaned
    assert cleaned.islower()

def test_clean_text_empty(text_cleaner):
    assert text_cleaner.clean_text("") == ""
    assert text_cleaner.clean_text(None) == ""

def test_remove_stopwords(text_cleaner):
    text = "the quick brown fox jumps over the lazy dog"
    cleaned = text_cleaner.remove_stopwords(text)
    assert "the" not in cleaned
    assert "over" not in cleaned

def test_lemmatize_text(text_cleaner):
    text = "running jumping swimming"
    lemmatized = text_cleaner.lemmatize_text(text)
    assert "run" in lemmatized
    assert "jump" in lemmatized
    assert "swim" in lemmatized

def test_preprocess_text_full(text_cleaner):
    text = "RT @user: The quick brown fox #jumps over the lazy dog! http://example.com"
    processed = text_cleaner.preprocess_text(text)
    assert "http" not in processed
    assert "#" not in processed
    assert "@" not in processed
    assert "the" not in processed
    assert processed.islower()

def test_preprocess_text_special_characters(text_cleaner):
    text = "Hello! This is a test with émojis 😊 and special chars @#$%"
    processed = text_cleaner.preprocess_text(text)
    assert "é" not in processed
    assert "��" not in processed
    assert "@" not in processed
    assert "#" not in processed
