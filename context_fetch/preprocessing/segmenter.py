from typing import List
import nltk

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


_SENT_MODEL = None


def _get_sentence_model():
    global _SENT_MODEL
    if SentenceTransformer is None:
        return None
    if _SENT_MODEL is None:
        _SENT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SENT_MODEL


def sentence_split(text: str) -> List[str]:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:  # pragma: no cover
        nltk.download("punkt")
    return nltk.sent_tokenize(text)


def semantic_chunk(sentences: List[str], max_tokens: int = 800, overlap_tokens: int = 150) -> List[str]:
    """Group sentences into chunks based on semantic similarity and size budget.

    Uses sentence-transformers embeddings to group adjacent sentences.
    `max_tokens`/`overlap_tokens` are approximate and use character counts as proxy.
    """
    if not sentences:
        return []

    # If sentence-transformers is unavailable, use a simple size-based chunker
    model = _get_sentence_model()

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current and current_len + sent_len > max_tokens:
            chunks.append(" ".join(current))
            if overlap_tokens > 0:
                carry = []
                carry_len = 0
                for s in reversed(current):
                    if carry_len + len(s) > overlap_tokens:
                        break
                    carry.append(s)
                    carry_len += len(s)
                current = list(reversed(carry))
                current_len = sum(len(s) for s in current)
            else:
                current = []
                current_len = 0
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks



