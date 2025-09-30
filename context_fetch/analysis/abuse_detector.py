from typing import List, Dict, Any, Optional


ABUSE_LEXICON = {
    "violence": [
        "kill", "murder", "shoot", "stab", "attack", "beat", "bomb", "execute",
        "destroy", "war", "genocide", "slaughter",
    ],
    "abuse": [
        "idiot", "stupid", "moron", "dumb", "trash", "hate", "loser", "nazi",
        "slur", "racist", "sexist", "harass", "threat",
    ],
}

LABEL_TEMPLATES = [
    "violent threat or instruction",
    "abusive or harassing language",
    "self-harm or suicide",
    "neutral or safe",
]


def _zero_shot_scores(sentences: List[str], classifier_name: str) -> List[Dict[str, float]]:
    try:
        from transformers import pipeline
    except Exception:
        return [{lab: 0.0 for lab in LABEL_TEMPLATES} for _ in sentences]

    try:
        clf = pipeline("zero-shot-classification", model=classifier_name)
        results = clf(sentences, candidate_labels=LABEL_TEMPLATES, multi_label=True)
        if isinstance(results, dict):
            results = [results]
        scores: List[Dict[str, float]] = []
        for r in results:  # type: ignore
            labels = r["labels"]
            s = r["scores"]
            scores.append({lab: float(val) for lab, val in zip(labels, s)})
        return scores
    except Exception:
        return [{lab: 0.0 for lab in LABEL_TEMPLATES} for _ in sentences]


def _lexicon_hits(text: str) -> Dict[str, List[str]]:
    text_low = text.lower()
    hits: Dict[str, List[str]] = {"violence": [], "abuse": []}
    for cat, words in ABUSE_LEXICON.items():
        for w in words:
            if w in text_low:
                hits[cat].append(w)
    return hits


def _severity(score_dict: Dict[str, float], lexicon: Dict[str, List[str]]) -> float:
    z = max(score_dict.get("violent threat or instruction", 0.0), score_dict.get("abusive or harassing language", 0.0))
    l = 0.0
    if lexicon["violence"]:
        l += 0.4
    if lexicon["abuse"]:
        l += 0.3
    return min(1.0, z * 0.7 + l)


def detect_abuse(
    *,
    sentences: List[str],
    timed_segments: Optional[List[Dict[str, Any]]],
    classifier_name: str = "facebook/bart-large-mnli",
) -> Dict[str, Any]:
    """Detect abusive/violent speech and attempt to align segments to timestamps.

    Returns a dict with overall severity and flagged examples with timestamps where possible.
    """
    if not sentences:
        return {"overall_severity": 0.0, "flags": []}

    # Zero-shot scores per sentence
    scores = _zero_shot_scores(sentences, classifier_name)

    flags = []
    severities = []
    for idx, sent in enumerate(sentences):
        scr = scores[idx]
        hits = _lexicon_hits(sent)
        sev = _severity(scr, hits)
        severities.append(sev)
        if sev >= 0.5:
            # Try to align to timestamp by naive mapping: sentence index to segment index
            ts = None
            if timed_segments and idx < len(timed_segments):
                ts = {
                    "start": float(timed_segments[idx].get("start", 0.0)),
                    "end": float(timed_segments[idx].get("end", 0.0)),
                }
            flags.append({
                "sentence_index": idx,
                "text": sent,
                "severity": round(sev, 3),
                "labels": {k: round(v, 3) for k, v in scr.items()},
                "lexicon_hits": hits,
                "timestamp": ts,
            })

    overall = round(float(sum(severities) / max(1, len(severities))), 3)
    return {"overall_severity": overall, "flags": flags}



