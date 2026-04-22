"""
test_document_processor.py — Tests for ai/document_processor.py

Covers: file format dispatch, text extraction for each supported format,
chunking logic (size, overlap, sentence-boundary splitting), and
edge cases (empty files, unsupported formats, malformed input).
"""

import pytest
import json
import os
from io import BytesIO

from ai.document_processor import (
    DocumentProcessor,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_REASONABLE_CHUNK_SIZE,
)


# ═══════════════════════════════════════════════════════════════════════════
# Text file processing
# ═══════════════════════════════════════════════════════════════════════════

class TestTextProcessing:
    """Tests for .txt and .md file extraction."""

    def test_process_text_file_from_path(self, sample_txt_file):
        result = DocumentProcessor.process_file(file_path=sample_txt_file)
        assert result["success"] is True
        assert "John Smith" in result["text"]
        assert "Acme Corporation" in result["text"]
        assert result["metadata"]["format"] == "text"

    def test_process_text_file_from_bytes(self):
        content = b"Hello World. This is a test document."
        result = DocumentProcessor.process_file(
            file_bytes=content,
            file_extension=".txt",
            filename="test.txt"
        )
        assert result["success"] is True
        assert "Hello World" in result["text"]
        assert result["metadata"]["character_count"] == len(content.decode("utf-8"))

    def test_text_word_count_is_accurate(self, sample_txt_file):
        result = DocumentProcessor.process_file(file_path=sample_txt_file)
        # Word count in metadata should match actual split
        expected_words = len(result["text"].split())
        assert result["metadata"]["word_count"] == expected_words

    def test_markdown_treated_as_text(self, tmp_dir):
        md_path = tmp_dir / "notes.md"
        md_path.write_text("# Heading\n\nSome *bold* text.", encoding="utf-8")
        result = DocumentProcessor.process_file(file_path=str(md_path))
        assert result["success"] is True
        assert result["metadata"]["format"] == "text"


# ═══════════════════════════════════════════════════════════════════════════
# CSV processing
# ═══════════════════════════════════════════════════════════════════════════

class TestCSVProcessing:
    """Tests for .csv file extraction."""

    def test_process_csv_file(self, sample_csv_file):
        result = DocumentProcessor.process_file(file_path=sample_csv_file)
        assert result["success"] is True
        assert "Alice Chen" in result["text"]
        assert "Bob Kumar" in result["text"]
        assert result["metadata"]["format"] == "csv"
        assert result["metadata"]["row_count"] == 3
        assert result["metadata"]["column_count"] == 4

    def test_csv_from_bytes(self):
        csv_bytes = b"Name,Age\nAlice,30\nBob,25\n"
        result = DocumentProcessor.process_file(
            file_bytes=csv_bytes,
            file_extension=".csv",
            filename="data.csv"
        )
        assert result["success"] is True
        assert result["metadata"]["row_count"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# JSON processing
# ═══════════════════════════════════════════════════════════════════════════

class TestJSONProcessing:
    """Tests for .json file extraction."""

    def test_process_json_file(self, sample_json_file):
        result = DocumentProcessor.process_file(file_path=sample_json_file)
        assert result["success"] is True
        assert "Project Alpha" in result["text"]
        assert "John Smith" in result["text"]
        assert result["metadata"]["format"] == "json"

    def test_json_from_bytes(self):
        data = {"key": "value", "nested": {"a": 1}}
        json_bytes = json.dumps(data).encode("utf-8")
        result = DocumentProcessor.process_file(
            file_bytes=json_bytes,
            file_extension=".json",
            filename="data.json"
        )
        assert result["success"] is True
        assert "value" in result["text"]

    def test_malformed_json_raises(self, tmp_dir):
        """process_file catches exceptions from _process_json and returns
        a failure dict rather than letting JSONDecodeError propagate."""
        path = tmp_dir / "bad.json"
        path.write_text("{invalid json content", encoding="utf-8")
        result = DocumentProcessor.process_file(file_path=str(path))
        assert result["success"] is False
        assert result["error"]  # should contain the decode error message


# ═══════════════════════════════════════════════════════════════════════════
# Unsupported formats and edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Tests for error handling and boundary conditions."""

    def test_unsupported_format_returns_error(self, tmp_dir):
        path = tmp_dir / "data.xyz"
        path.write_text("some content", encoding="utf-8")
        result = DocumentProcessor.process_file(file_path=str(path))
        assert result["success"] is False
        assert "Unsupported" in result.get("error", "")

    def test_no_extension_raises_error(self):
        result = DocumentProcessor.process_file(
            file_bytes=b"content",
            file_extension=None,
            filename="noext"
        )
        assert result["success"] is False

    def test_empty_text_file(self, tmp_dir):
        path = tmp_dir / "empty.txt"
        path.write_text("", encoding="utf-8")
        result = DocumentProcessor.process_file(file_path=str(path))
        assert result["success"] is True
        assert result["text"] == ""
        assert result["metadata"]["character_count"] == 0

    def test_unicode_content_handled(self, tmp_dir):
        path = tmp_dir / "unicode.txt"
        content = "日本語テスト — Ünïcödé — café — 中文"
        path.write_text(content, encoding="utf-8")
        result = DocumentProcessor.process_file(file_path=str(path))
        assert result["success"] is True
        assert "café" in result["text"]
        assert "中文" in result["text"]


# ═══════════════════════════════════════════════════════════════════════════
# Chunking logic
# ═══════════════════════════════════════════════════════════════════════════

class TestChunking:
    """Tests for DocumentProcessor.chunk_text() — the sentence-boundary-aware
    text splitter that feeds chunks to the LLM extraction pipeline."""

    def test_short_text_returns_single_chunk(self):
        text = "This is a short paragraph."
        chunks = DocumentProcessor.chunk_text(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_text_split_into_multiple_chunks(self, sample_long_text):
        chunks = DocumentProcessor.chunk_text(
            sample_long_text, chunk_size=2000, chunk_overlap=200
        )
        assert len(chunks) > 1
        # Every chunk should be non-empty
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_chunk_size_respected(self, sample_long_text):
        chunk_size = 3000
        chunks = DocumentProcessor.chunk_text(
            sample_long_text, chunk_size=chunk_size, chunk_overlap=300
        )
        for chunk in chunks:
            # Allow small overshoot for sentence-boundary alignment
            assert len(chunk) <= chunk_size + 200

    def test_overlap_creates_shared_content(self, sample_long_text):
        chunks = DocumentProcessor.chunk_text(
            sample_long_text, chunk_size=2000, chunk_overlap=500
        )
        if len(chunks) >= 2:
            # The end of chunk 0 should overlap with the start of chunk 1
            tail_of_first = chunks[0][-200:]
            assert tail_of_first in chunks[1] or chunks[1][:200] in chunks[0]

    def test_no_content_lost_during_chunking(self, sample_long_text):
        """Every character of the original text should appear in at least
        one chunk (accounting for overlap and whitespace stripping)."""
        chunks = DocumentProcessor.chunk_text(
            sample_long_text, chunk_size=2000, chunk_overlap=200
        )
        reassembled = " ".join(chunks)
        # Check a sample of words from the original
        original_words = sample_long_text.split()
        sample_words = original_words[::50]  # every 50th word
        for word in sample_words:
            assert word in reassembled, f"Word '{word}' lost during chunking"

    def test_empty_text_returns_empty_list(self):
        chunks = DocumentProcessor.chunk_text("")
        # Either empty list or a list with one empty/whitespace string
        assert len(chunks) == 0 or all(c.strip() == "" for c in chunks)

    def test_sentence_boundary_splitting(self):
        """Chunks should preferably break at sentence boundaries.
        This is best-effort — the splitter falls back to hard cuts when
        no sentence break is available within the target window."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = DocumentProcessor.chunk_text(text, chunk_size=40, chunk_overlap=5)
        # Count how many chunks end at a sentence boundary (period)
        boundary_count = sum(
            1 for c in chunks if c.rstrip().endswith(".")
        )
        # At least half should respect sentence boundaries
        assert boundary_count >= len(chunks) // 2

    def test_default_chunk_size_is_reasonable(self):
        """Default chunk size should be >= MIN_REASONABLE_CHUNK_SIZE."""
        assert DEFAULT_CHUNK_SIZE >= MIN_REASONABLE_CHUNK_SIZE

    def test_default_overlap_is_smaller_than_chunk_size(self):
        assert DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE