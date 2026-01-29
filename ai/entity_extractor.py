"""
Entity Extractor module for GraphNet.
Uses LangChain and OpenAI to extract entities and relationships from text.
"""

import logging
import json
from typing import List, Dict, Any, Tuple
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Entity(BaseModel):
    """Entity model"""
    name: str = Field(description="Name of the entity")
    type: str = Field(description="Type/category of the entity (e.g., Person, Organization, Location, Concept, Product, Date)")
    description: str = Field(description="Brief description of the entity")


class Relationship(BaseModel):
    """Relationship model"""
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    type: str = Field(description="Type of relationship (e.g., WORKS_FOR, LOCATED_IN, RELATED_TO, OWNS)")
    description: str = Field(description="Description of the relationship")


class ExtractionResult(BaseModel):
    """Result of entity extraction"""
    entities: List[Entity] = Field(description="List of extracted entities")
    relationships: List[Relationship] = Field(description="List of extracted relationships")


class EntityExtractor:
    """Extract entities and relationships from text using LangChain and LLMs"""
    
    def __init__(self):
        """Initialize the entity extractor with LangChain"""
        self.llm = None
        self.parser = PydanticOutputParser(pydantic_object=ExtractionResult)
        self.initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize the LLM
        
        Returns:
            Boolean indicating initialization success
        """
        try:
            if not config.OPENAI_API_KEY:
                logger.error("OpenAI API key not configured")
                return False
            
            self.llm = ChatOpenAI(
                model=config.OPENAI_MODEL,
                temperature=0,
                openai_api_key=config.OPENAI_API_KEY
            )
            
            self.initialized = True
            logger.info("Entity extractor initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize entity extractor: {str(e)}")
            self.initialized = False
            return False
    
    def extract_entities_and_relationships(self, text: str, context: str = None) -> Dict[str, Any]:
        """
        Extract entities and relationships from text
        
        Args:
            text: Text to process
            context: Optional context (e.g., filename, source)
        
        Returns:
            Dictionary containing entities and relationships
        """
        if not self.initialized:
            logger.error("Entity extractor not initialized")
            return {"entities": [], "relationships": [], "error": "Not initialized"}
        
        try:
            # Create the extraction prompt
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at extracting entities and relationships from text.
                
Extract all relevant entities (people, organizations, locations, concepts, products, dates, etc.) 
and their relationships from the given text.

Entity types should be one of: Person, Organization, Location, Concept, Product, Date, Event, Technology, or Other.

Relationship types should be descriptive (e.g., WORKS_FOR, LOCATED_IN, RELATED_TO, OWNS, CREATED, MANAGES, PARTICIPATED_IN).

{format_instructions}

Be thorough but precise. Only extract entities and relationships that are clearly mentioned or strongly implied in the text.
"""),
                ("user", "Context: {context}\n\nText to analyze:\n{text}")
            ])
            
            # Format the prompt
            formatted_prompt = prompt_template.format_messages(
                format_instructions=self.parser.get_format_instructions(),
                context=context or "Unknown source",
                text=text[:4000]  # Limit text length for API
            )
            
            # Get response from LLM
            response = self.llm.invoke(formatted_prompt)
            
            # Parse the response
            result = self.parser.parse(response.content)
            
            return {
                "entities": [
                    {
                        "name": e.name,
                        "type": e.type,
                        "description": e.description
                    } for e in result.entities
                ],
                "relationships": [
                    {
                        "source": r.source,
                        "target": r.target,
                        "type": r.type,
                        "description": r.description
                    } for r in result.relationships
                ],
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {
                "entities": [],
                "relationships": [],
                "success": False,
                "error": str(e)
            }
    
    def extract_from_chunks(self, chunks: List[str], context: str = None) -> Dict[str, Any]:
        """
        Extract entities and relationships from multiple text chunks
        
        Args:
            chunks: List of text chunks
            context: Optional context
        
        Returns:
            Aggregated extraction results
        """
        all_entities = {}
        all_relationships = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            result = self.extract_entities_and_relationships(chunk, context)
            
            if result.get("success"):
                # Aggregate entities (avoid duplicates)
                for entity in result.get("entities", []):
                    entity_key = f"{entity['name']}_{entity['type']}"
                    if entity_key not in all_entities:
                        all_entities[entity_key] = entity
                
                # Collect relationships
                all_relationships.extend(result.get("relationships", []))
        
        return {
            "entities": list(all_entities.values()),
            "relationships": all_relationships,
            "success": True,
            "chunks_processed": len(chunks)
        }
    
    def simple_extract(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Simplified extraction that returns entities and relationships directly
        
        Args:
            text: Text to process
        
        Returns:
            Tuple of (entities, relationships)
        """
        result = self.extract_entities_and_relationships(text)
        return result.get("entities", []), result.get("relationships", [])


class SimpleEntityExtractor:
    """
    Fallback entity extractor using simple heuristics (when LLM is not available)
    """
    
    @staticmethod
    def extract_basic_entities(text: str) -> Dict[str, Any]:
        """
        Extract basic entities using simple pattern matching
        This is a fallback when LLM is not available
        
        Args:
            text: Text to process
        
        Returns:
            Dictionary with basic entities
        """
        import re
        
        entities = []
        
        # Extract potential organization names (capitalized words followed by Inc, Corp, etc.)
        org_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company)\b'
        orgs = re.findall(org_pattern, text)
        for org in set(orgs):
            entities.append({
                "name": org,
                "type": "Organization",
                "description": f"Organization mentioned in text"
            })
        
        # Extract potential person names (Title + Capitalized Name)
        person_pattern = r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        persons = re.findall(person_pattern, text)
        for person in set(persons):
            entities.append({
                "name": person,
                "type": "Person",
                "description": f"Person mentioned in text"
            })
        
        # Extract dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        dates = re.findall(date_pattern, text)
        for date in set(dates):
            entities.append({
                "name": date,
                "type": "Date",
                "description": f"Date mentioned in text"
            })
        
        return {
            "entities": entities,
            "relationships": [],
            "success": True,
            "method": "simple_heuristic"
        }
