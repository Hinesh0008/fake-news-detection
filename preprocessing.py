"""Text cleaning helpers shared by model training and the Streamlit app."""

import re
from typing import Final

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Keep negation words because removing them can reverse a news statement.
STOP_WORDS: Final[frozenset[str]] = frozenset(
    ENGLISH_STOP_WORDS - {"no", "nor", "not"}
)


def clean_text(text: str) -> str:
    """Lowercase text, remove links/symbols, and filter English stopwords."""
    if not isinstance(text, str):
        raise TypeError("News text must be a string.")

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [word for word in text.split() if word not in STOP_WORDS]
    return " ".join(words)
