import re


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Basic normalization suitable for transcripts.

    - Normalize whitespace
    - Remove bracketed artifacts like [Music], [Applause]
    - Strip
    """
    text = re.sub(r"\[(?:music|applause|laughter|silence)\]", " ", text, flags=re.I)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()



