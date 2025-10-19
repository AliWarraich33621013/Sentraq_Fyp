from typing import List


def _hf_summarize(texts: List[str], model_name: str) -> str:
    try:
        from transformers import pipeline
    except Exception:
        return _extractive_fallback(texts)

    try:
        pipe = pipeline("summarization", model=model_name)
    except Exception:
        return _extractive_fallback(texts)

    outputs = []
    for chunk in texts:
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            result = pipe(chunk, max_length=220, min_length=60, do_sample=False)
            outputs.append(result[0]["summary_text"])  # type: ignore
        except Exception:
            outputs.append(_lead_k_sentences(chunk, k=3))
    
    # Remove duplicate sentences from all outputs
    def deduplicate_sentences(text: str) -> str:
        sentences = text.split('. ')
        unique_sentences = []
        seen = set()
        for sent in sentences:
            sent_clean = sent.strip()
            if sent_clean and sent_clean not in seen:
                unique_sentences.append(sent_clean)
                seen.add(sent_clean)
        return '. '.join(unique_sentences)
    
    # Deduplicate each output first
    outputs = [deduplicate_sentences(output) for output in outputs]
    combined = " ".join(outputs)
    # Always deduplicate the final combined result
    combined = deduplicate_sentences(combined)
    
    if len(combined.split()) > 300:
        try:
            result = pipe(combined, max_length=250, min_length=80, do_sample=False)
            summary = result[0]["summary_text"]
            return deduplicate_sentences(summary)
        except Exception:
            return _lead_k_sentences(combined, k=5)
    return combined


def _openai_summarize(texts: List[str]) -> str:
    import os
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    prompt = (
        "You are a helpful assistant. Summarize the following transcript chunks into a concise, context-aware summary. "
        "Highlight key points, arguments, and conclusions. Avoid verbatim extraction.\n\n"
    )
    content = "\n\n".join(f"Chunk {i+1}:\n{t}" for i, t in enumerate(texts))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You summarize transcripts."},
            {"role": "user", "content": prompt + content},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return resp.choices[0].message.content or ""


def _lead_k_sentences(text: str, k: int = 3) -> str:
    from context_fetch.preprocessing.segmenter import sentence_split

    sents = sentence_split(text)
    return " ".join(sents[:k])


def _extractive_fallback(chunks: List[str]) -> str:
    parts = []
    for ch in chunks:
        parts.append(_lead_k_sentences(ch, k=2))
    return " ".join(parts)


def summarize_chunks(chunks: List[str], *, use_openai: bool = False, model_name: str = "facebook/bart-large-cnn") -> str:
    if not chunks:
        return ""
    if use_openai:
        return _openai_summarize(chunks)
    return _hf_summarize(chunks, model_name)



