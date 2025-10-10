from typing import List, Dict, Any, Optional


ABUSE_LEXICON = {
    "violence": [
        "kill", "murder", "shoot", "stab", "attack", "beat", "bomb", "execute",
        "destroy", "war", "genocide", "slaughter", "violence", "violent",
    ],
    "abuse": [
        "idiot", "stupid", "moron", "dumb", "trash", "hate", "loser", "nazi",
        "slur", "racist", "sexist", "harass", "threat", "abuse", "abusive",
    ],
    "self_harm": [
        "suicide", "kill myself", "killing myself", "end it all", "ending it all",
        "hurt myself", "hurting myself", "self harm", "self-harm", "self harm",
        "take my life", "taking my life", "end my life", "ending my life",
        "die", "dying", "death", "dead", "overdose", "overdosing",
        "cut myself", "cutting myself", "harm myself", "harming myself",
        "want to die", "wanna die", "don't want to live", "don't wanna live",
        "not worth living", "better off dead", "better off without me",
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
    except Exception as e:
        print(f"Transformers import failed: {e}")
        return [{lab: 0.0 for lab in LABEL_TEMPLATES} for _ in sentences]

    try:
        clf = pipeline("zero-shot-classification", model=classifier_name)
        # Test with a simple sentence first
        test_result = clf("I want to hurt myself", candidate_labels=LABEL_TEMPLATES, multi_label=True)
        print(f"Test classification result: {test_result}")
        
        results = clf(sentences, candidate_labels=LABEL_TEMPLATES, multi_label=True)
        if isinstance(results, dict):
            results = [results]
        scores: List[Dict[str, float]] = []
        for r in results:  # type: ignore
            labels = r["labels"]
            s = r["scores"]
            # Ensure all labels are present in the result
            score_dict = {lab: 0.0 for lab in LABEL_TEMPLATES}
            for lab, val in zip(labels, s):
                if lab in score_dict:
                    score_dict[lab] = float(val)
            scores.append(score_dict)
        return scores
    except Exception as e:
        print(f"Zero-shot classification failed: {e}")
        return [{lab: 0.0 for lab in LABEL_TEMPLATES} for _ in sentences]


def _lexicon_hits(text: str) -> Dict[str, List[str]]:
    text_low = text.lower()
    hits: Dict[str, List[str]] = {"violence": [], "abuse": [], "self_harm": []}
    for cat, words in ABUSE_LEXICON.items():
        for w in words:
            if w in text_low:
                hits[cat].append(w)
    return hits


def _severity(score_dict: Dict[str, float], lexicon: Dict[str, List[str]]) -> float:
    # Get zero-shot scores for all categories
    z_violence = score_dict.get("violent threat or instruction", 0.0)
    z_abuse = score_dict.get("abusive or harassing language", 0.0)
    z_self_harm = score_dict.get("self-harm or suicide", 0.0)
    
    # Check if zero-shot classifier is working (all scores are 0.0 indicates failure)
    z_max = max(z_violence, z_abuse, z_self_harm)
    zero_shot_working = z_max > 0.0
    
    # Lexicon-based scoring with higher weights for self-harm
    l = 0.0
    if lexicon["violence"]:
        l += 0.4
    if lexicon["abuse"]:
        l += 0.3
    if lexicon["self_harm"]:
        l += 0.6  # Higher weight for self-harm detection
    
    # If zero-shot is working, combine both; otherwise rely primarily on lexicon
    if zero_shot_working:
        combined = z_max * 0.6 + l * 0.4
    else:
        # When zero-shot fails, rely more heavily on lexicon patterns
        combined = l * 0.8 + 0.2  # Add small base score when patterns are detected
    
    # Lower threshold for self-harm detection
    if z_self_harm > 0.3 or lexicon["self_harm"]:
        combined = max(combined, 0.5)
    
    return min(1.0, combined)


def _enhanced_self_harm_detection(sentences: List[str]) -> List[Dict[str, Any]]:
    """Enhanced self-harm detection using pattern matching and context analysis."""
    flags = []
    
    for idx, sent in enumerate(sentences):
        sent_lower = sent.lower()
        
        # Direct self-harm patterns
        self_harm_patterns = [
            "hurt myself", "hurting myself", "kill myself", "killing myself",
            "end it all", "ending it all", "end my life", "ending my life",
            "take my life", "taking my life", "want to die", "wanna die",
            "don't want to live", "don't wanna live", "better off dead",
            "not worth living", "better off without me", "self harm", "self-harm"
        ]
        
        # Suicide-related terms
        suicide_terms = ["suicide", "suicidal", "overdose", "overdosing"]
        
        # Pain/emotional distress indicators
        pain_indicators = ["overwhelmed", "pain", "suffering", "hopeless", "hopelessness"]
        
        severity = 0.0
        detected_patterns = []
        
        # Check for direct self-harm patterns
        for pattern in self_harm_patterns:
            if pattern in sent_lower:
                severity = max(severity, 0.8)
                detected_patterns.append(pattern)
        
        # Check for suicide terms
        for term in suicide_terms:
            if term in sent_lower:
                severity = max(severity, 0.7)
                detected_patterns.append(term)
        
        # Check for pain indicators combined with self-harm context
        pain_count = sum(1 for indicator in pain_indicators if indicator in sent_lower)
        if pain_count > 0 and any(term in sent_lower for term in ["myself", "my life", "die", "end"]):
            severity = max(severity, 0.6)
            detected_patterns.extend([ind for ind in pain_indicators if ind in sent_lower])
        
        if severity > 0.5:
            flags.append({
                "sentence_index": idx,
                "text": sent,
                "severity": round(severity, 3),
                "labels": {"self-harm or suicide": severity, "enhanced_detection": True},
                "lexicon_hits": {"self_harm": detected_patterns},
                "timestamp": None,
                "detection_method": "enhanced_pattern_matching"
            })
    
    return flags


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
        if sev >= 0.3:  # Lower threshold for better detection
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

    # Add enhanced self-harm detection
    enhanced_flags = _enhanced_self_harm_detection(sentences)
    flags.extend(enhanced_flags)
    
    # Remove duplicates based on sentence index
    seen_indices = set()
    unique_flags = []
    for flag in flags:
        if flag["sentence_index"] not in seen_indices:
            unique_flags.append(flag)
            seen_indices.add(flag["sentence_index"])
    
    # Calculate overall severity including enhanced detection
    all_severities = severities + [flag["severity"] for flag in enhanced_flags]
    overall = round(float(sum(all_severities) / max(1, len(all_severities))), 3)
    
    return {"overall_severity": overall, "flags": unique_flags}



