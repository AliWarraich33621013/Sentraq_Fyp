import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from context_fetch.fetchers.youtube_fetcher import fetch_youtube_transcript
from context_fetch.fetchers.file_fetcher import read_transcript_file
from context_fetch.preprocessing.cleaner import normalize_text
from context_fetch.preprocessing.segmenter import (
    sentence_split,
    semantic_chunk,
)
from context_fetch.analysis.summarizer import summarize_chunks
from context_fetch.analysis.abuse_detector import detect_abuse
from context_fetch.utils.io_utils import ensure_parent_dir


@dataclass
class Segment:
    text: str
    start: Optional[float] = None
    end: Optional[float] = None


def run_pipeline(
    *,
    text: Optional[str],
    timed_segments: Optional[List[Dict[str, Any]]],
    use_openai: bool = False,
    model_name: str = "facebook/bart-large-cnn",
    classifier_name: str = "facebook/bart-large-mnli",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> Dict[str, Any]:
    """
    Execute the end-to-end pipeline.

    - `text` is used when there are no timestamps
    - `timed_segments` is a list of {"text", "start", "end"}
    """

    # 1) Normalize
    if timed_segments is not None:
        raw_text = " ".join(seg["text"] for seg in timed_segments)
    else:
        raw_text = text or ""

    cleaned = normalize_text(raw_text)

    # 2) Sentence split
    sentences = sentence_split(cleaned)

    # 3) Semantic chunk into larger blocks for summarization
    chunks = semantic_chunk(sentences, max_tokens=chunk_size, overlap_tokens=chunk_overlap)

    # 4) Summarize
    summary = summarize_chunks(chunks, use_openai=use_openai, model_name=model_name)

    # 5) Abuse detection (attempt to align back to timestamps if provided)
    abuse = detect_abuse(
        sentences=sentences,
        timed_segments=timed_segments,
        classifier_name=classifier_name,
    )

    return {
        "summary": summary,
        "abuse_detection": abuse,
        "stats": {
            "num_sentences": len(sentences),
            "num_chunks": len(chunks),
            "total_chars": len(cleaned),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Context-aware transcript summarizer and abuse detector")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--youtube-url", type=str, help="YouTube video URL")
    src.add_argument("--input-file", type=str, help="Path to local transcript file (plain text)")

    parser.add_argument("--output", type=str, default="out.json", help="Output JSON path")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI for summarization/classification")
    parser.add_argument("--model-name", type=str, default="facebook/bart-large-cnn", help="HF model for summarization")
    parser.add_argument("--classifier-name", type=str, default="facebook/bart-large-mnli", help="HF model for zero-shot classification")
    parser.add_argument("--chunk-size", type=int, default=800, help="Approx token limit per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Approx token overlap between chunks")
    parser.add_argument("--language", type=str, default="en", help="Preferred transcript language code")
    parser.add_argument("--whisper-audio", action="store_true", help="Fallback: download audio and transcribe with Whisper (optional)")

    args = parser.parse_args()

    metadata: Dict[str, Any] = {}
    text_input: Optional[str] = None
    timed_segments: Optional[List[Dict[str, Any]]] = None

    if args.youtube_url:
        result = fetch_youtube_transcript(
            args.youtube_url, preferred_language=args.language, whisper_fallback=args.whisper_audio
        )
        metadata = {
            "source": "youtube",
            "video_id": result.get("video_id"),
            "language": result.get("language"),
        }
        timed_segments = result["segments"]
    else:
        text_input = read_transcript_file(args.input_file)
        metadata = {
            "source": "file",
            "path": os.path.abspath(args.input_file),
        }

    output = run_pipeline(
        text=text_input,
        timed_segments=timed_segments,
        use_openai=args.use_openai,
        model_name=args.model_name,
        classifier_name=args.classifier_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    final = {
        "metadata": metadata,
        **output,
    }

    ensure_parent_dir(args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()




