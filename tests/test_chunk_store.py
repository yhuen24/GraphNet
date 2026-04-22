"""
test_chunk_store.py — Tests for ai/chunk_store.py

Covers: deterministic chunk ID generation, storage and deduplication,
semantic search retrieval, source-filtered retrieval, chunk ordering,
source listing, and clear/count operations.

These tests use the TF-IDF fallback so they run without a Google API key.
"""

import pytest
from ai.chunk_store import ChunkStore


@pytest.fixture
def chunk_store(tmp_dir):
    """Provide a ChunkStore backed by a temp ChromaDB directory."""
    cs = ChunkStore(chroma_dir=str(tmp_dir / "chroma_test"))
    # Initialize with no API key → TF-IDF fallback
    success = cs.initialize(google_api_key="")
    if not success:
        pytest.skip("ChunkStore could not initialize (chromadb or sklearn missing)")
    return cs


@pytest.fixture
def sample_chunks():
    """Three distinct text chunks from a hypothetical corporate document."""
    return [
        "HSBC Holdings plc reported revenue of $51.7 billion for fiscal year 2024. "
        "The company operates in over 60 countries with major presences in Hong Kong, "
        "London, and New York. CEO Georges Elhedery announced a strategic restructuring.",

        "The Global Banking and Markets division is being reorganised into two units: "
        "Eastern Markets and Western Markets. This restructuring aims to reduce "
        "operational overlap and improve regional accountability.",

        "McKinsey & Company was engaged as an external consultant for the restructuring. "
        "The estimated cost of the programme is $1.2 billion over three years, with "
        "expected annual savings of $600 million once fully implemented.",
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Chunk ID generation
# ═══════════════════════════════════════════════════════════════════════════

class TestChunkIdGeneration:
    def test_deterministic_ids(self):
        """Same source + index always produces the same ID."""
        id1 = ChunkStore._make_chunk_id("report.pdf", 0)
        id2 = ChunkStore._make_chunk_id("report.pdf", 0)
        assert id1 == id2

    def test_different_index_different_id(self):
        id1 = ChunkStore._make_chunk_id("report.pdf", 0)
        id2 = ChunkStore._make_chunk_id("report.pdf", 1)
        assert id1 != id2

    def test_different_source_different_id(self):
        id1 = ChunkStore._make_chunk_id("report.pdf", 0)
        id2 = ChunkStore._make_chunk_id("other.pdf", 0)
        assert id1 != id2

    def test_id_is_hex_string(self):
        cid = ChunkStore._make_chunk_id("test.pdf", 5)
        assert len(cid) == 16
        assert all(c in "0123456789abcdef" for c in cid)


# ═══════════════════════════════════════════════════════════════════════════
# Storage
# ═══════════════════════════════════════════════════════════════════════════

class TestChunkStorage:
    def test_store_chunks(self, chunk_store, sample_chunks):
        stored = chunk_store.store_chunks(sample_chunks, source="report.pdf")
        assert stored == 3

    def test_count_after_storage(self, chunk_store, sample_chunks):
        chunk_store.store_chunks(sample_chunks, source="report.pdf")
        assert chunk_store.count() == 3

    def test_duplicate_storage_skips(self, chunk_store, sample_chunks):
        """Re-storing the same chunks should not create duplicates."""
        first = chunk_store.store_chunks(sample_chunks, source="report.pdf")
        second = chunk_store.store_chunks(sample_chunks, source="report.pdf")
        assert first == 3
        assert second == 0  # all already stored
        assert chunk_store.count() == 3

    def test_store_empty_list(self, chunk_store):
        stored = chunk_store.store_chunks([], source="empty.pdf")
        assert stored == 0

    def test_store_from_multiple_sources(self, chunk_store, sample_chunks):
        chunk_store.store_chunks(sample_chunks[:2], source="doc1.pdf")
        chunk_store.store_chunks(sample_chunks[1:], source="doc2.pdf")
        # 2 from doc1 + 2 from doc2 (chunk at index 1 has different source so not a duplicate)
        assert chunk_store.count() == 4


# ═══════════════════════════════════════════════════════════════════════════
# Search / retrieval
# ═══════════════════════════════════════════════════════════════════════════

class TestChunkSearch:
    def test_search_returns_relevant_chunks(self, chunk_store, sample_chunks):
        chunk_store.store_chunks(sample_chunks, source="report.pdf")
        results = chunk_store.search("HSBC revenue", top_k=3)
        assert len(results) > 0
        # The first chunk mentions revenue — it should rank high
        top_text = results[0]["text"]
        assert "revenue" in top_text.lower() or "HSBC" in top_text

    def test_search_returns_score(self, chunk_store, sample_chunks):
        chunk_store.store_chunks(sample_chunks, source="report.pdf")
        results = chunk_store.search("restructuring plan", top_k=2)
        for r in results:
            assert "score" in r
            assert 0 <= r["score"] <= 1

    def test_search_returns_metadata(self, chunk_store, sample_chunks):
        chunk_store.store_chunks(sample_chunks, source="report.pdf")
        results = chunk_store.search("McKinsey consulting", top_k=1)
        assert len(results) > 0
        assert results[0]["source"] == "report.pdf"
        assert "chunk_index" in results[0]

    def test_search_empty_store(self, chunk_store):
        results = chunk_store.search("anything")
        assert results == []

    def test_search_top_k_limits_results(self, chunk_store, sample_chunks):
        chunk_store.store_chunks(sample_chunks, source="report.pdf")
        results = chunk_store.search("HSBC", top_k=1)
        assert len(results) <= 1


# ═══════════════════════════════════════════════════════════════════════════
# Source-based retrieval
# ═══════════════════════════════════════════════════════════════════════════

class TestSourceRetrieval:
    def test_get_chunks_by_source(self, chunk_store, sample_chunks):
        chunk_store.store_chunks(sample_chunks, source="report.pdf")
        chunks = chunk_store.get_chunks_by_source("report.pdf")
        assert len(chunks) == 3

    def test_chunks_sorted_by_index(self, chunk_store, sample_chunks):
        chunk_store.store_chunks(sample_chunks, source="report.pdf")
        chunks = chunk_store.get_chunks_by_source("report.pdf")
        indices = [c["chunk_index"] for c in chunks]
        assert indices == sorted(indices)

    def test_get_chunks_unknown_source(self, chunk_store, sample_chunks):
        chunk_store.store_chunks(sample_chunks, source="report.pdf")
        chunks = chunk_store.get_chunks_by_source("nonexistent.pdf")
        assert len(chunks) == 0

    def test_list_sources(self, chunk_store, sample_chunks):
        chunk_store.store_chunks(sample_chunks[:1], source="doc1.pdf")
        chunk_store.store_chunks(sample_chunks[1:2], source="doc2.pdf")
        sources = chunk_store.list_sources()
        assert "doc1.pdf" in sources
        assert "doc2.pdf" in sources

    def test_list_sources_empty(self, chunk_store):
        assert chunk_store.list_sources() == []


# ═══════════════════════════════════════════════════════════════════════════
# Clear and lifecycle
# ═══════════════════════════════════════════════════════════════════════════

class TestClearAndLifecycle:
    def test_clear_removes_all(self, chunk_store, sample_chunks):
        chunk_store.store_chunks(sample_chunks, source="report.pdf")
        assert chunk_store.count() == 3
        chunk_store.clear()
        assert chunk_store.count() == 0

    def test_save_is_noop(self, chunk_store):
        """save() should not raise — it's a no-op for ChromaDB."""
        chunk_store.save()

    def test_count_empty_store(self, chunk_store):
        assert chunk_store.count() == 0