"""
Configuration module for GraphNet application.
Manages environment variables and application settings.
Supports both local .env files and Streamlit Cloud secrets.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables (local development)
load_dotenv()

# Try to access Streamlit secrets (cloud deployment)
try:
    import streamlit as st
    _secrets = st.secrets
except Exception:
    _secrets = {}


def _get(key: str, default: str = "") -> str:
    """Read a config value from Streamlit secrets first, then os.environ."""
    try:
        return str(_secrets[key])
    except Exception:
        return os.getenv(key, default)


class Config:
    """Configuration class for GraphNet"""

    # Graph Database Mode: 'neo4j' or 'embedded'
    GRAPH_MODE = _get("GRAPH_MODE", "embedded")

    # Neo4j Configuration
    NEO4J_URI = _get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = _get("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = _get("NEO4J_PASSWORD", "password")

    GOOGLE_API_KEY = _get("GOOGLE_API_KEY", "")
    AI_MODEL = _get("AI_MODEL", "gemini-2.5-flash")

    # Application Settings
    MAX_FILE_SIZE_MB = int(_get("MAX_FILE_SIZE_MB", "10"))
    CHUNK_SIZE = int(_get("CHUNK_SIZE", "40000"))
    CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "2000"))

    # Supported file formats
    SUPPORTED_FORMATS = [
        '.txt', '.pdf', '.docx', '.xlsx', '.pptx',
        '.csv', '.json', '.md'
    ]

    # Entity extraction settings
    MAX_ENTITIES_PER_CHUNK = int(_get("MAX_ENTITIES_PER_CHUNK", "20"))
    CONFIDENCE_THRESHOLD = float(_get("CONFIDENCE_THRESHOLD", "0.7"))

    # Visualization settings
    GRAPH_HEIGHT = "700px"
    GRAPH_WIDTH = "100%"

    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []

        if not cls.GOOGLE_API_KEY:
            issues.append("⚠️ GOOGLE_API_KEY is not set")

        if not cls.NEO4J_PASSWORD or cls.NEO4J_PASSWORD == "password":
            issues.append("⚠️ NEO4J_PASSWORD should be changed from default")

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Return configuration as dictionary (excluding sensitive data)"""
        return {
            "neo4j_uri": cls.NEO4J_URI,
            "neo4j_username": cls.NEO4J_USERNAME,
            "ai_model": cls.AI_MODEL,
            "max_file_size_mb": cls.MAX_FILE_SIZE_MB,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "supported_formats": cls.SUPPORTED_FORMATS
        }


# Create a global config instance
config = Config()