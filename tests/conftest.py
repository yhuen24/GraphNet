"""
conftest.py — Shared fixtures for GraphNet test suite.

Provides reusable test data, temporary directories, and mock objects
used across multiple test modules.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path


# ── Temporary directory fixture ──────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a clean temporary directory for each test."""
    return tmp_path


# ── Sample text fixtures ────────────────────────────────────────────────────

@pytest.fixture
def sample_corporate_text():
    """A short corporate paragraph with known entities and relationships."""
    return (
        "HSBC Holdings plc, headquartered in London, reported revenue of "
        "$51.7 billion for the fiscal year 2024. The company's CEO, "
        "Georges Elhedery, announced a strategic restructuring of the "
        "Global Banking and Markets division. HSBC operates in over "
        "60 countries and territories, with significant operations in "
        "Hong Kong, Shanghai, and New York. The restructuring plan was "
        "developed in consultation with McKinsey & Company."
    )


@pytest.fixture
def sample_short_text():
    """A minimal text for basic extraction tests."""
    return "Alice works at Acme Corporation in London."


@pytest.fixture
def sample_long_text():
    """A text long enough to require chunking (~10,000 chars)."""
    paragraph = (
        "The global financial services industry continues to undergo "
        "significant transformation driven by technological innovation, "
        "regulatory change, and evolving customer expectations. "
        "Major institutions are investing heavily in digital capabilities "
        "while managing legacy infrastructure constraints. "
    )
    # Repeat to exceed default chunk size
    return paragraph * 60


# ── Sample file fixtures ────────────────────────────────────────────────────

@pytest.fixture
def sample_txt_file(tmp_dir):
    """Create a temporary .txt file with known content."""
    path = tmp_dir / "test_document.txt"
    content = (
        "Project Alpha was led by John Smith from Engineering.\n"
        "The project involved collaboration with Sarah Jones from Marketing.\n"
        "It was completed in Q3 2024 with a budget of $500,000.\n"
        "The client was Acme Corporation based in New York.\n"
    )
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture
def sample_csv_file(tmp_dir):
    """Create a temporary .csv file with tabular data."""
    path = tmp_dir / "employees.csv"
    content = (
        "Name,Department,Role,Location\n"
        "Alice Chen,Engineering,Lead Developer,London\n"
        "Bob Kumar,Marketing,Director,New York\n"
        "Carol White,Finance,Analyst,Singapore\n"
    )
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture
def sample_json_file(tmp_dir):
    """Create a temporary .json file with structured data."""
    path = tmp_dir / "projects.json"
    data = {
        "projects": [
            {
                "name": "Project Alpha",
                "lead": "John Smith",
                "department": "Engineering",
                "budget": 500000
            },
            {
                "name": "Project Beta",
                "lead": "Sarah Jones",
                "department": "Marketing",
                "budget": 250000
            }
        ]
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


@pytest.fixture
def empty_file(tmp_dir):
    """Create a 0-byte file for edge case testing."""
    path = tmp_dir / "empty.txt"
    path.write_text("", encoding="utf-8")
    return str(path)


@pytest.fixture
def sample_entities():
    """A list of extracted entities in the format returned by EntityExtractor."""
    return [
        {"name": "HSBC Holdings", "type": "Organization",
         "description": "Global banking corporation", "properties": {}},
        {"name": "Georges Elhedery", "type": "Person",
         "description": "CEO of HSBC", "properties": {}},
        {"name": "London", "type": "Location",
         "description": "City in the United Kingdom", "properties": {}},
        {"name": "McKinsey & Company", "type": "Organization",
         "description": "Management consulting firm", "properties": {}},
    ]


@pytest.fixture
def sample_relationships():
    """A list of extracted relationships matching sample_entities."""
    return [
        {"source": "Georges Elhedery", "target": "HSBC Holdings",
         "type": "CEO_OF", "description": "Serves as CEO",
         "properties": {}},
        {"source": "HSBC Holdings", "target": "London",
         "type": "HEADQUARTERED_IN", "description": "Head office location",
         "properties": {}},
        {"source": "HSBC Holdings", "target": "McKinsey & Company",
         "type": "CONSULTED_WITH", "description": "Strategic consulting",
         "properties": {}},
    ]