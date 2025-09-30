from typing import Dict, Any, List, Optional
import re

try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
except Exception:  # pragma: no cover
    YouTubeTranscriptApi = None  # type: ignore
    TranscriptsDisabled = Exception  # type: ignore


YOUTUBE_ID_RE = re.compile(r"(?:v=|be/)([A-Za-z0-9_-]{6,})")


def _extract_video_id(url: str) -> Optional[str]:
    m = YOUTUBE_ID_RE.search(url)
    return m.group(1) if m else None


def fetch_youtube_transcript(url: str, preferred_language: str = "en", whisper_fallback: bool = False) -> Dict[str, Any]:
    """Fetch transcript with timestamps from YouTube.

    Returns: {"video_id", "language", "segments": [{"text","start","end"}...]}
    """
    video_id = _extract_video_id(url) or url
    segments: List[Dict[str, Any]] = []
    lang_used = preferred_language

    if YouTubeTranscriptApi is not None:
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = None
            # 1) Try manually created in preferred language
            try:
                transcript = transcripts.find_manually_created_transcript([preferred_language])
            except Exception:
                pass
            # 2) Try generated in preferred language
            if transcript is None:
                try:
                    transcript = transcripts.find_generated_transcript([preferred_language])
                except Exception:
                    pass
            # 3) Try any English variant
            if transcript is None:
                for code in ["en", "en-US", "en-GB"]:
                    try:
                        transcript = transcripts.find_transcript([code])
                        break
                    except Exception:
                        continue
            # 4) Fallback to first available transcript
            if transcript is None:
                try:
                    transcript = next(iter(transcripts))
                except Exception:
                    transcript = None

            if transcript is not None:
                lang_used = getattr(transcript, "language_code", preferred_language)
                data = transcript.fetch()
                for item in data:
                    start = float(item.get("start", 0.0))
                    duration = float(item.get("duration", 0.0))
                    segments.append({
                        "text": item.get("text", ""),
                        "start": start,
                        "end": start + duration,
                    })
                return {"video_id": video_id, "language": lang_used, "segments": segments}
        except Exception:
            pass

    if whisper_fallback:
        # Lazy import to keep optional
        try:
            import subprocess
            import tempfile
            import os
            import json
            # Requires: ffmpeg, openai-whisper
            with tempfile.TemporaryDirectory() as d:
                audio_path = os.path.join(d, "audio.mp3")
                # Download audio via yt-dlp if available
                yt_dlp = ["yt-dlp", "-x", "--audio-format", "mp3", "-o", audio_path, url]
                subprocess.run(yt_dlp, check=True)
                # Transcribe
                import whisper  # type: ignore
                model = whisper.load_model("base")
                result = model.transcribe(audio_path, language=preferred_language)
                # Convert segments
                for seg in result.get("segments", []):
                    segments.append({
                        "text": seg.get("text", ""),
                        "start": float(seg.get("start", 0.0)),
                        "end": float(seg.get("end", 0.0)),
                    })
                return {"video_id": video_id, "language": preferred_language, "segments": segments}
        except Exception:
            pass

    # Fallback: empty with no segments
    return {"video_id": video_id, "language": preferred_language, "segments": segments}



