from typing import Optional


def read_transcript_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()




