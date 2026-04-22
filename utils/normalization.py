"""
utils/normalization.py — Entity name normalization for GraphNet.

Single source of truth for turning raw entity names into canonical keys
so that "Company A", "company a", "CompanyA", "COMPANY A", and
"  Company   A  " all resolve to the same graph node.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Tuple, Optional


# ── Articles / noise words stripped from the FRONT of a name ─────────────────
_LEADING_ARTICLES = re.compile(
    r"^(the|a|an|la|le|el|los|las|les|die|der|das)\s+", re.IGNORECASE
)

# ── Characters we collapse (anything not alphanumeric or space) ──────────────
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE = re.compile(r"\s+")


def canonical_key(name: str) -> str:
    """
    Produce a stable, lowercase, whitespace-collapsed key from *name*.

    Pipeline:
        1. Unicode → ASCII (café → cafe)
        2. Strip outer whitespace
        3. Remove leading articles ("The ", "A ", …)
        4. Lowercase
        5. Strip punctuation & special chars
        6. Collapse runs of whitespace to a single space

    >>> canonical_key("  The  Company A  ")
    'company a'
    >>> canonical_key("CompanyA")
    'companya'
    >>> canonical_key("COMPANY  A")
    'company a'
    >>> canonical_key("Côte d'Ivoire")
    'cote divoire'
    """
    # 1. Unicode normalisation → strip accents
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # 2. Strip
    text = text.strip()

    # 3. Remove leading article
    text = _LEADING_ARTICLES.sub("", text).strip()

    # 4. Lowercase
    text = text.lower()

    # 5. Remove non-alphanumeric (keep spaces)
    text = _NON_ALNUM.sub("", text)

    # 6. Collapse whitespace
    text = _MULTI_SPACE.sub(" ", text).strip()

    return text


def display_name(name: str) -> str:
    """
    Clean a raw entity name for *display* purposes (not for keying).

    - Strips whitespace
    - Removes leading articles
    - Title-cases the result
    """
    text = name.strip()
    text = _LEADING_ARTICLES.sub("", text).strip()
    return text.title()


# ── Fuzzy matching helpers ───────────────────────────────────────────────────

def similarity(a: str, b: str) -> float:
    """
    Return a 0-1 similarity score between two strings.
    Uses canonical keys so casing/punctuation don't matter.
    """
    ka, kb = canonical_key(a), canonical_key(b)
    if ka == kb:
        return 1.0
    return SequenceMatcher(None, ka, kb).ratio()


def fuzzy_match(query: str, candidates: List[str],
                threshold: float = 0.55,
                top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Return candidates whose canonical key is similar to *query*,
    sorted best-first.

    Uses a three-tier strategy:
        1. Exact canonical match  (score = 1.0)
        2. Substring containment  (score = 0.90)
        3. SequenceMatcher ratio  (score = ratio)

    Args:
        query:      The user's search string.
        candidates: All entity names in the graph.
        threshold:  Minimum score to include.
        top_k:      Maximum results returned.

    Returns:
        List of (candidate, score) tuples, descending by score.
    """
    qkey = canonical_key(query)
    if not qkey:
        return []

    scored: List[Tuple[str, float]] = []

    for cand in candidates:
        ckey = canonical_key(cand)
        if not ckey:
            continue

        # Tier 1 — exact canonical match
        if ckey == qkey:
            scored.append((cand, 1.0))
            continue

        # Tier 2 — substring containment (either direction)
        if qkey in ckey or ckey in qkey:
            scored.append((cand, 0.90))
            continue

        # Tier 2b — word-level overlap
        qwords = set(qkey.split())
        cwords = set(ckey.split())
        if qwords and cwords:
            overlap = len(qwords & cwords) / max(len(qwords), len(cwords))
            if overlap >= 0.5:
                scored.append((cand, 0.80 + overlap * 0.1))
                continue

        # Tier 3 — character-level similarity
        ratio = SequenceMatcher(None, qkey, ckey).ratio()
        if ratio >= threshold:
            scored.append((cand, ratio))

    # Sort descending by score, then alphabetically for ties
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored[:top_k]


def find_best_match(query: str, candidates: List[str],
                    threshold: float = 0.55) -> Optional[str]:
    """
    Return the single best-matching candidate, or None if nothing
    exceeds *threshold*.
    """
    matches = fuzzy_match(query, candidates, threshold=threshold, top_k=1)
    return matches[0][0] if matches else None