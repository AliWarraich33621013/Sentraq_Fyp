from context_fetch.preprocessing.cleaner import normalize_text
from context_fetch.preprocessing.segmenter import sentence_split, semantic_chunk


def test_clean_and_segment():
    raw = "[Music] Hello there. This is a test! We test chunking."
    cleaned = normalize_text(raw)
    sentences = sentence_split(cleaned)
    chunks = semantic_chunk(sentences, max_tokens=80, overlap_tokens=20)
    assert len(sentences) >= 2
    assert len(chunks) >= 1




