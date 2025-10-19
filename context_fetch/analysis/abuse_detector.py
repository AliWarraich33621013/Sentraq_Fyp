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


def _context_adjust_scores(sentence: str, score_dict: Dict[str, float], lexicon: Dict[str, List[str]]) -> Dict[str, float]:
    """Adjust zero-shot scores using simple context rules to reduce false positives.

    - Violence: downweight if no violent lexicon/patterns and sentence lacks intent/action cues
    - Abuse: downweight if no abuse lexicon and no second-person target
    - Self-harm: upweight when strong self-harm intent patterns are present
    - Neutral descriptive: heavily downweight when purely descriptive without harmful intent
    """
    s = sentence.lower()

    adjusted = dict(score_dict)

    # Heuristics for intent/action cues
    intent_verbs = ["kill", "stab", "shoot", "attack", "hurt", "harm", "bomb", "punch", "beat"]
    has_intent = any(v in s for v in intent_verbs)

    # Second-person targeting often present in harassment
    has_second_person = any(p in s for p in [" you ", " you're", " you.", " you,", " your "]) or s.startswith("you ")

    # Check for purely descriptive/neutral content
    descriptive_patterns = [
        "his eyes were", "her eyes were", "the room was", "it was", "he was", "she was",
        "the light", "the sound", "the air", "the floor", "the wall", "the door",
        "like a", "as if", "seemed to", "appeared to", "looked like"
    ]
    is_purely_descriptive = any(pattern in s for pattern in descriptive_patterns) and not has_intent

    # Check for suicidal context (should not be tagged as violent threat)
    suicidal_context = any(phrase in s for phrase in [
        "dying was", "death would", "kill myself", "end it all", "better off dead",
        "not worth living", "overdose", "jump from", "fall alone"
    ])

    # Downweight violence false positives when descriptive only
    if adjusted.get("violent threat or instruction", 0.0) > 0:
        if not lexicon["violence"] and not has_intent:
            adjusted["violent threat or instruction"] *= 0.4
        
        # Heavily downweight violence in suicidal contexts (should be self-harm, not violence)
        if suicidal_context:
            adjusted["violent threat or instruction"] *= 0.2

    # Downweight abuse when no insults and no second-person target
    if adjusted.get("abusive or harassing language", 0.0) > 0:
        if not lexicon["abuse"] and not has_second_person:
            adjusted["abusive or harassing language"] *= 0.35

    # Heavily downweight all categories for purely descriptive content
    if is_purely_descriptive and not lexicon["violence"] and not lexicon["abuse"] and not lexicon["self_harm"]:
        for key in ["violent threat or instruction", "abusive or harassing language", "self-harm or suicide"]:
            if adjusted.get(key, 0.0) > 0:
                adjusted[key] *= 0.2

    # Upweight self-harm for clear intent patterns (including third-person narrative of self)
    self_harm_cues = [
        "kill myself", "killing myself", "hurt myself", "hurting myself", "end it all",
        "end my life", "take my life", "want to die", "wanna die", "better off dead",
        "not worth living", "overdose", "cut myself", "his death would", "my death would",
        "would help everyone"  # often appears with self-referential death
    ]
    if any(cue in s for cue in self_harm_cues) or lexicon["self_harm"]:
        adjusted["self-harm or suicide"] = max(adjusted.get("self-harm or suicide", 0.0) * 1.2, 0.6)

    # Clamp to [0,1]
    for k in list(adjusted.keys()):
        v = adjusted[k]
        adjusted[k] = 1.0 if v > 1.0 else (0.0 if v < 0.0 else v)

    return adjusted


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
        # Contextually adjust zero-shot scores to reduce false positives
        scr_raw = scores[idx]
        hits = _lexicon_hits(sent)
        scr = _context_adjust_scores(sent, scr_raw, hits)
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
    
    # Calculate overall severity with risk-aware scaling: emphasize top risks
    all_severities = severities + [flag["severity"] for flag in enhanced_flags]
    if all_severities:
        top_sorted = sorted(all_severities, reverse=True)
        top_max = top_sorted[0]
        top_k_mean = sum(top_sorted[: min(5, len(top_sorted))]) / min(5, len(top_sorted))
        overall_val = 0.7 * top_max + 0.3 * top_k_mean
    else:
        overall_val = 0.0
    overall = round(float(min(1.0, max(0.0, overall_val))), 3)
    
    return {"overall_severity": overall, "flags": unique_flags}



