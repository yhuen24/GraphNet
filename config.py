"""
Configuration module for GraphNet application.
Manages environment variables and application settings.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for GraphNet"""

    # Graph Database Mode: 'neo4j' or 'embedded'
    GRAPH_MODE = os.getenv("GRAPH_MODE", "embedded")  # Default to embedded

    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Application Settings
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Supported file formats
    SUPPORTED_FORMATS = [
        '.txt', '.pdf', '.docx', '.xlsx', '.pptx', 
        '.csv', '.json', '.md'
    ]
    
    # Entity extraction settings
    MAX_ENTITIES_PER_CHUNK = int(os.getenv("MAX_ENTITIES_PER_CHUNK", "20"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    
    # Visualization settings
    GRAPH_HEIGHT = "700px"
    GRAPH_WIDTH = "100%"
    
    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        
        if not cls.OPENAI_API_KEY:
            issues.append("⚠️ OPENAI_API_KEY is not set")
        
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
            "openai_model": cls.OPENAI_MODEL,
            "max_file_size_mb": cls.MAX_FILE_SIZE_MB,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "supported_formats": cls.SUPPORTED_FORMATS
        }


# Create a global config instance
config = Config()
