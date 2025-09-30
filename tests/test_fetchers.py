from context_fetch.fetchers.file_fetcher import read_transcript_file


def test_read_transcript_file():
    text = read_transcript_file("data/samples/sample_transcript.txt")
    assert "conflict" in text.lower()




