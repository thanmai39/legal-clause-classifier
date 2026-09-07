"""
Unit tests for the pure text-processing logic in src/data/preprocess.py.

These don't need the dataset, the model, or any external resource -- they
test small, isolated functions directly. This is what a "unit test" means:
verifying one small unit of logic in isolation, fast enough to run on
every commit.
"""

from src.data.preprocess import clean_text, extract_category


def test_clean_text_collapses_whitespace():
    assert clean_text("Hello   \n\n  world  ") == "Hello world"


def test_clean_text_strips_leading_trailing_whitespace():
    assert clean_text("  Governing Law clause  ") == "Governing Law clause"


def test_clean_text_handles_already_clean_text():
    assert clean_text("Simple clause text.") == "Simple clause text."


def test_extract_category_finds_quoted_category_name():
    question = (
        'Highlight the parts (if any) of this contract related to '
        '"Governing Law" that should be reviewed by a lawyer. Details: ...'
    )
    assert extract_category(question) == "Governing Law"


def test_extract_category_handles_missing_quotes():
    assert extract_category("no quoted category here") == "UNKNOWN"
