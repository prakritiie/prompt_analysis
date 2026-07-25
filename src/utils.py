"""
Text feature helpers shared across pipeline stages.

Kept deliberately dependency-light: pandas, textstat, stdlib only.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
import textstat

from src.config import (
    CODE_MARKERS,
    CODE_MARKER_MIN_HITS,
    REPEAT_NGRAM_N,
    TERMINAL_PUNCT,
)

_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_WHITESPACE = re.compile(r"\s+")


def normalise(text) -> str:
    """Lowercase, strip punctuation, collapse whitespace. For wordclouds/TF-IDF."""
    text = _NON_ALPHA.sub(" ", str(text).lower())
    return _WHITESPACE.sub(" ", text).strip()


def word_count(text) -> int:
    return len(str(text).split())


def sentence_count(text) -> int:
    """
    Number of sentences, minimum 1.

    Returning 1 rather than 0 for punctuation-free outputs is what keeps
    words_per_sentence from producing inf. A single unpunctuated blob really
    is one run-on sentence, so this is the honest floor, not a fudge.
    """
    parts = [s for s in _SENTENCE_SPLIT.split(str(text)) if s.strip()]
    return max(len(parts), 1)


def looks_like_code(text) -> bool:
    """
    Heuristic: does this output contain enough code/markup markers that a
    prose readability score would be meaningless?

    Requires 2+ distinct markers so a single stray brace or colon in ordinary
    prose doesn't trip it.
    """
    s = str(text)
    hits = sum(1 for marker in CODE_MARKERS if marker in s)
    return hits >= CODE_MARKER_MIN_HITS


def flesch(text) -> float:
    """Flesch reading ease, NaN on failure rather than crashing the pipeline."""
    try:
        return float(textstat.flesch_reading_ease(str(text)))
    except Exception:
        return np.nan


def readability_level(score) -> str:
    if pd.isna(score):
        return "not scored"
    if score >= 90:
        return "very easy"
    if score >= 60:
        return "easy"
    if score >= 30:
        return "medium"
    if score >= 10:
        return "hard"
    if score >= 0:
        return "very hard"
    return "extremely complex"


def is_truncated(text) -> bool:
    """Output that doesn't end on terminal punctuation is probably cut off."""
    s = str(text).rstrip()
    return bool(s) and not s.endswith(TERMINAL_PUNCT)


def has_repeated_ngram(text, n: int = REPEAT_NGRAM_N, min_count: int = 3) -> bool:
    """
    Detect degenerate repetition: the same n-gram appearing min_count+ times.

    This is the classic failure mode of a generator that fell into a loop.
    """
    tokens = str(text).lower().split()
    if len(tokens) < n * min_count:
        return False
    grams = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return grams.most_common(1)[0][1] >= min_count


def categorize_prompt(text) -> str:
    """
    Rule-based task classifier, matching on leading verb.

    Retained ONLY as the baseline that clustering is measured against. Its
    'Other' rate is a headline number in the audit, not a bug to hide.
    """
    t = str(text).lower().lstrip()

    if t.startswith((
        "can you", "could you", "would you", "do you", "did you",
        "is it", "are there", "should i", "what", "why", "how", "when", "who",
    )):
        return "Question"
    if t.startswith(("write", "create", "generate", "compose", "draft")):
        return "Creative Task"
    if t.startswith(("explain", "describe", "define", "clarify", "elaborate")):
        return "Explanation"
    if t.startswith(("calculate", "solve", "compute", "find the value", "evaluate")):
        return "Problem Solving"
    if t.startswith(("suggest", "recommend", "advise", "tips for", "ways to")):
        return "Advice"
    if t.startswith(("give", "list", "provide", "name", "mention", "outline")):
        return "Listing Task"
    if t.startswith(("rewrite", "rephrase", "improve", "edit", "correct", "fix")):
        return "Editing/Rewriting"
    return "Other"
