"""
test_normalization.py — Tests for utils/normalization.py

Covers canonical_key, display_name, similarity, fuzzy_match, and
find_best_match. These are the foundation of GraphNet's entity
deduplication, so correctness here prevents duplicate nodes in the graph.
"""

import pytest
from utils.normalization import (
    canonical_key, display_name, similarity, fuzzy_match, find_best_match
)


# ═══════════════════════════════════════════════════════════════════════════
# canonical_key
# ═══════════════════════════════════════════════════════════════════════════

class TestCanonicalKey:
    """Ensure that variant spellings of the same entity collapse to one key."""

    def test_basic_lowercasing(self):
        assert canonical_key("Company A") == "company a"

    def test_all_caps(self):
        assert canonical_key("COMPANY A") == "company a"

    def test_mixed_case_collapses(self):
        assert canonical_key("CompanyA") == "companya"

    def test_extra_whitespace_collapsed(self):
        assert canonical_key("  Company   A  ") == "company a"

    def test_leading_article_stripped(self):
        assert canonical_key("The Company A") == "company a"
        assert canonical_key("A Company") == "company"
        assert canonical_key("An Entity") == "entity"

    def test_non_english_articles(self):
        assert canonical_key("La Compagnie") == "compagnie"
        assert canonical_key("El Banco") == "banco"
        assert canonical_key("Die Firma") == "firma"

    def test_unicode_accents_stripped(self):
        assert canonical_key("Côte d'Ivoire") == "cote divoire"
        assert canonical_key("café") == "cafe"
        assert canonical_key("São Paulo") == "sao paulo"

    def test_punctuation_removed(self):
        assert canonical_key("McKinsey & Company") == "mckinsey company"
        assert canonical_key("AT&T") == "att"

    def test_empty_string(self):
        assert canonical_key("") == ""

    def test_whitespace_only(self):
        assert canonical_key("   ") == ""

    def test_same_entity_different_forms_match(self):
        """Core deduplication scenario: all forms produce identical keys."""
        forms = ["HSBC Holdings", "hsbc holdings", "  HSBC  Holdings  ",
                 "The HSBC Holdings"]
        keys = [canonical_key(f) for f in forms]
        assert len(set(keys)) == 1


# ═══════════════════════════════════════════════════════════════════════════
# display_name
# ═══════════════════════════════════════════════════════════════════════════

class TestDisplayName:
    """Ensure display names are clean and title-cased."""

    def test_basic_title_case(self):
        assert display_name("john smith") == "John Smith"

    def test_strips_whitespace(self):
        assert display_name("  alice chen  ") == "Alice Chen"

    def test_removes_leading_article(self):
        assert display_name("The Company") == "Company"

    def test_preserves_internal_capitalisation_via_title(self):
        # title() will produce "Mckinsey" from "mckinsey"; that's expected
        result = display_name("mckinsey & company")
        assert result.startswith("Mckinsey")


# ═══════════════════════════════════════════════════════════════════════════
# similarity
# ═══════════════════════════════════════════════════════════════════════════

class TestSimilarity:
    """Verify the similarity function for entity matching."""

    def test_identical_strings(self):
        assert similarity("Alice", "Alice") == 1.0

    def test_case_insensitive(self):
        assert similarity("alice", "ALICE") == 1.0

    def test_completely_different(self):
        score = similarity("Apple", "Zebra")
        assert score < 0.5

    def test_partial_overlap(self):
        score = similarity("HSBC Holdings", "HSBC Bank")
        assert 0.4 < score < 1.0

    def test_empty_strings(self):
        assert similarity("", "") == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# fuzzy_match
# ═══════════════════════════════════════════════════════════════════════════

class TestFuzzyMatch:
    """Verify fuzzy matching used by search_entities."""

    @pytest.fixture
    def candidates(self):
        return [
            "HSBC Holdings",
            "McKinsey & Company",
            "Goldman Sachs",
            "Hong Kong",
            "HSBC Bank UK",
        ]

    def test_exact_match_returns_score_1(self, candidates):
        results = fuzzy_match("HSBC Holdings", candidates)
        assert len(results) > 0
        assert results[0][0] == "HSBC Holdings"
        assert results[0][1] == 1.0

    def test_case_insensitive_match(self, candidates):
        results = fuzzy_match("hsbc holdings", candidates)
        assert results[0][0] == "HSBC Holdings"
        assert results[0][1] == 1.0

    def test_substring_match(self, candidates):
        results = fuzzy_match("HSBC", candidates)
        # Should match both HSBC entries
        matched_names = [r[0] for r in results]
        assert "HSBC Holdings" in matched_names
        assert "HSBC Bank UK" in matched_names

    def test_no_match_below_threshold(self, candidates):
        results = fuzzy_match("Completely Unrelated XYZ", candidates,
                              threshold=0.8)
        assert len(results) == 0

    def test_top_k_limits_results(self, candidates):
        results = fuzzy_match("HSBC", candidates, top_k=1)
        assert len(results) <= 1

    def test_empty_query(self, candidates):
        results = fuzzy_match("", candidates)
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════
# find_best_match
# ═══════════════════════════════════════════════════════════════════════════

class TestFindBestMatch:
    def test_returns_best(self):
        candidates = ["Alice Chen", "Bob Kumar", "Alice Wong"]
        result = find_best_match("Alice Chen", candidates)
        assert result == "Alice Chen"

    def test_returns_none_when_no_match(self):
        candidates = ["Alice Chen", "Bob Kumar"]
        result = find_best_match("Completely Different Name", candidates,
                                 threshold=0.9)
        assert result is None