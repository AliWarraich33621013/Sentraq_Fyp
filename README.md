Context Fetch: YouTube Transcript Summarizer and Abuse Detector

This project fetches a transcript from a YouTube video URL or reads a raw text file, cleans and segments it, generates a context-aware summary, and detects abusive/violent speech with severity and examples. It’s modular and beginner-friendly, with optional notes for using paid models.

Features
- Fetch transcript from YouTube via youtube-transcript-api, with a Whisper fallback option (local) if needed
- Accept raw transcript/text files
- Clean and tokenize text with NLTK (or spaCy optional)
- Context-aware summarization using Transformer models (local), with optional OpenAI
- Abuse/violence detection with timestamps/segments, severity scoring, and examples
- CLI usage with clear flags
- Sample data and expected outputs
- Simple unit tests

Project Structure
```
context_fetch/
  __init__.py
  fetchers/
    __init__.py
    youtube_fetcher.py
    file_fetcher.py
  preprocessing/
    __init__.py
    cleaner.py
    segmenter.py
  analysis/
    __init__.py
    summarizer.py
    abuse_detector.py
  utils/
    __init__.py
    io_utils.py
    time_utils.py
data/
  samples/
    sample_transcript.txt
    expected_output_sample.json
tests/
  test_fetchers.py
  test_pipeline.py
main.py
requirements.txt
README.md
```

Prerequisites
- Python 3.10+
- Windows, macOS, or Linux
- Internet access for downloading models the first time

Quickstart
1) Create and activate a virtual environment
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader punkt
```

3) Run with a YouTube URL
```bash
python main.py --youtube-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --output out_youtube.json
```

4) Run with a local transcript file
```bash
python main.py --input-file data/samples/sample_transcript.txt --output out_file.json
```

5) View results
Open the generated JSON file (summary, abuse flags with timestamps, and metadata).

CLI Usage
```bash
python main.py --help | cat
```

Optional: Better Models
- OpenAI: set `OPENAI_API_KEY` and use `--use-openai` to enable GPT-based summarization and classification. Costs may apply.
- Whisper (local): install `openai-whisper` and `ffmpeg`, then use `--whisper-audio` to transcribe audio if `youtube-transcript-api` fails.

Design Overview
- Fetchers: download transcript from YouTube (`youtube_fetcher.py`) or read from a file (`file_fetcher.py`). YouTube transcripts include timestamps when available.
- Preprocessing: `cleaner.py` normalizes text; `segmenter.py` splits into sentences and semantic chunks using `sentence-transformers` for context-aware grouping.
- Analysis: `summarizer.py` uses a local Transformer summarization pipeline (e.g., `facebook/bart-large-cnn`) on chunks and combines them; optional OpenAI path. `abuse_detector.py` uses a lightweight classifier with a small vocabulary and a zero-shot/sequence classification transformer for robustness.
- Utils: helpers for IO and timestamp alignment.

Example Outputs
- YouTube URL input: see `data/samples/expected_output_sample.json` for format.
- Raw transcript file input: same JSON schema, different `source` metadata.

Testing
```bash
pytest -q
```

Troubleshooting
- If `youtube-transcript-api` cannot fetch transcripts, try `--whisper-audio` (requires `ffmpeg` and downloads the audio automatically).
- If model downloads are slow, pre-download by running once with internet access.
- For spaCy users, install `spacy` and a model: `python -m spacy download en_core_web_sm`, then set `--use-spacy`.

License
MIT




