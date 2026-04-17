"""
Entity Extractor module for GraphNet.
Uses LangChain and Google Gemini to extract entities and relationships from text.
"""

import logging
import json
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants - Constrained Types
# =============================================================================

ENTITY_TYPES = [
    "Person", "Organization", "Location", "Concept", "Product", "Date",
    "Event", "Technology", "Document", "Project", "Skill", "Policy",
    "Department", "Course", "Qualification", "Task", "Role", "Metric",
    "Risk", "Decision", "Regulation", "Contract", "Resource",
    "Deliverable"
]

RELATIONSHIP_TYPES = [
    # Hierarchy & Structure
    "WORKS_FOR", "REPORTS_TO", "LOCATED_IN", "PART_OF", "BELONGS_TO", "HOLDS_ROLE",
    # Identity & Skills
    "HAS_SKILL", "TEACHES", "STUDIES",
    # Execution & Projects
    "MANAGES", "PARTICIPATED_IN", "ASSIGNED", "CREATED", "DELIVERED", "OCCURRED_ON",
    # Tasks & Dependencies
    "REQUIRES", "DEPENDS_ON", "PRECEDED_BY", "RESULTED_IN", "MITIGATES",
    # Goals & Alignment
    "ALIGNS_WITH", "TARGETS",
    # Products & Tools
    "USES_TOOL", "INTEGRATES_WITH", "OWNS",
    # External
    "SERVES", "PARTNERS_WITH",
    # Documents & Auditing
    "AUTHORED", "APPROVED", "REVIEWED", "CONTAINS", "REFERENCES",
    "SUPERSEDES", "COMPLIES_WITH", "CERTIFIES",
]

# Properties that contain date values and should be normalized
DATE_PROPERTY_KEYS = {
    "deadline", "start_date", "end_date", "date_created", "date_modified",
    "effective_date", "expiry_date", "due_date", "submission_date",
    "review_date", "approval_date", "completion_date"
}

# Default chunk configuration
DEFAULT_CHUNK_SIZE = 40000
DEFAULT_CHUNK_OVERLAP = 2000
MAX_SINGLE_EXTRACT_LENGTH = 50000


# =============================================================================
# Pydantic Models
# =============================================================================

class Entity(BaseModel):
    """Entity model with properties and confidence"""
    name: str = Field(description="Name of the entity")
    type: str = Field(description=f"Type/category of the entity. Must be one of: {', '.join(ENTITY_TYPES)}")
    description: str = Field(description="Brief description of the entity")
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.8,
        description="Confidence score for this extraction (0.0 to 1.0)"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional properties (e.g., deadline, status, priority, email, url)"
    )


class Relationship(BaseModel):
    """Relationship model with confidence"""
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    type: str = Field(description=f"Type of relationship. Must be one of: {', '.join(RELATIONSHIP_TYPES)}")
    description: str = Field(description="Description of the relationship")
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.8,
        description="Confidence score for this extraction (0.0 to 1.0)"
    )


class ExtractionResult(BaseModel):
    """Result of entity extraction"""
    entities: List[Entity] = Field(description="List of extracted entities")
    relationships: List[Relationship] = Field(description="List of extracted relationships")


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert at extracting entities and relationships from text for a knowledge graph.

Extract all relevant entities and their relationships from the given text.

## Entity Types
Entity type MUST be one of: {entity_types}

## Relationship Types
Relationship type MUST be one of: {relationship_types}

Relationship categories for reference:
- Hierarchy & Structure: WORKS_FOR, REPORTS_TO, LOCATED_IN, PART_OF, BELONGS_TO, HOLDS_ROLE
- Identity & Skills: HAS_SKILL, TEACHES, STUDIES
- Execution & Projects: MANAGES, PARTICIPATED_IN, ASSIGNED, CREATED, DELIVERED
- Tasks & Dependencies: REQUIRES, DEPENDS_ON, PRECEDED_BY, RESULTED_IN, MITIGATES
- Goals & Alignment: ALIGNS_WITH, TARGETS
- Products & Tools: USES, INTEGRATES, OWNS
- External: SERVES, PARTNERS_WITH
- Documents & Auditing: AUTHORED, APPROVED, REVIEWED, CONTAINS, REFERENCES, SUPERSEDES, COMPLIES_WITH, CERTIFIES

## Entity Properties
When extracting entities, include relevant properties as key-value pairs.
Common properties by entity type:

- Task / Deliverable: deadline, status, priority, assigned_to
- Event: start_date, end_date, location, organizer
- Person: email, role, department, phone, responsibilities
- Document: version, author, date_created, date_modified, format
- Project: start_date, end_date, budget, status, phase
- Contract: effective_date, expiry_date, value, parties
- Metric: value, unit, period, target
- Risk: severity, likelihood, impact, mitigation_status
- Policy / Regulation / Standard: version, effective_date, issuing_body
- Qualification: institution, date_awarded, expiry_date, level
- Organization / Department: size, location, head

## Date Handling
- Normalize ALL dates to ISO 8601 format (YYYY-MM-DD) in properties.
- Interpret phrases like "due by", "no later than", "submit before", "deadline:",
  parenthetical dates, and "by [date]" as deadline properties on the relevant entity.
- Relative dates like "next Friday" or "end of semester" should be kept as-is if
  the exact date cannot be determined.
- Deadlines, due dates, start/end dates MUST be stored as PROPERTIES on the entity,
  NOT as separate Date nodes with relationships.

## Confidence Scoring
- Assign a confidence score (0.0 to 1.0) to each entity and relationship.
- 0.9-1.0: Explicitly stated in text
- 0.7-0.89: Strongly implied
- 0.5-0.69: Reasonably inferred
- Below 0.5: Do not extract

## Output Format
Return your response as a JSON object with this EXACT structure:
{{
    "entities": [
        {{
            "name": "Task A",
            "type": "Task",
            "description": "Brief description",
            "confidence": 0.95,
            "properties": {{
                "deadline": "2026-05-15",
                "status": "In Progress",
                "priority": "High"
            }}
        }}
    ],
    "relationships": [
        {{
            "source": "Entity1",
            "target": "Entity2",
            "type": "ASSIGNED",
            "description": "Brief description",
            "confidence": 0.9
        }}
    ]
}}

## Rules
- Only extract what is clearly stated or strongly implied (confidence >= 0.5).
- Do NOT invent properties that are not mentioned or implied in the text.
- Normalize all dates to YYYY-MM-DD where possible.
- Use the MOST SPECIFIC entity type available (e.g., "Department" not "Organization" for a department).
- Use the MOST SPECIFIC relationship type available (e.g., "ASSIGNED" not "REFERENCES" for task assignments).
- Return ONLY the JSON object, no other text."""


# =============================================================================
# Main Extractor
# =============================================================================

class EntityExtractor:
    """Extract entities and relationships from text using LangChain and LLMs"""

    def __init__(self):
        """Initialize the entity extractor with LangChain"""
        self.llm = None
        self.initialized = False
        self.parser = JsonOutputParser(pydantic_object=ExtractionResult)
        self._max_retries = 3
        self._retry_base_delay = 2  # seconds

    def initialize(self) -> bool:
        """
        Initialize the LLM

        Returns:
            Boolean indicating initialization success
        """
        try:
            if not config.GOOGLE_API_KEY:
                logger.error("Google API key not configured")
                return False

            self.llm = ChatGoogleGenerativeAI(
                model=config.AI_MODEL,
                temperature=0,
                google_api_key=config.GOOGLE_API_KEY
            )

            self.initialized = True
            logger.info("Entity extractor initialized with Gemini")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            return False

    # -------------------------------------------------------------------------
    # Text Chunking
    # -------------------------------------------------------------------------

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> List[str]:
        """
        Split text into overlapping chunks so entities at boundaries aren't lost.
        Attempts to break at sentence boundaries for cleaner extraction.

        Args:
            text: Full text to split
            chunk_size: Maximum characters per chunk
            overlap: Characters of overlap between consecutive chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Try to break at a sentence boundary within the last 20% of the chunk
            if end < len(text):
                search_start = max(start, end - int(chunk_size * 0.2))
                last_period = text.rfind('. ', search_start, end)
                last_newline = text.rfind('\n', search_start, end)
                break_point = max(last_period, last_newline)
                if break_point > search_start:
                    end = break_point + 1

            chunks.append(text[start:end].strip())
            start = end - overlap

            # Avoid infinite loop on tiny overlap
            if start >= len(text):
                break

        logger.info(f"Split text ({len(text)} chars) into {len(chunks)} chunks")
        return chunks

    # -------------------------------------------------------------------------
    # LLM Call with Retry
    # -------------------------------------------------------------------------

    def _invoke_llm_with_retry(self, messages) -> Optional[Any]:
        """
        Call the LLM with exponential backoff retry logic.

        Args:
            messages: Formatted prompt messages

        Returns:
            LLM response or None on failure
        """
        for attempt in range(self._max_retries):
            try:
                response = self.llm.invoke(messages)
                return response
            except Exception as e:
                delay = self._retry_base_delay * (2 ** attempt)
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self._max_retries}): {e}. "
                    f"Retrying in {delay}s..."
                )
                if attempt < self._max_retries - 1:
                    time.sleep(delay)
                else:
                    logger.error(f"LLM call failed after {self._max_retries} attempts")
                    raise

    # -------------------------------------------------------------------------
    # Response Parsing
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_llm_response(response) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM response into a dictionary, handling markdown fences.

        Args:
            response: Raw LLM response

        Returns:
            Parsed dictionary or None
        """
        # Handle Gemini 3 Flash list-of-blocks responses
        content = response.content
        if isinstance(content, str):
            response_text = content.strip()
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text" or "text" in block:
                        parts.append(block.get("text", ""))
            response_text = "".join(parts).strip()
        else:
            response_text = str(content).strip()

        # Remove markdown code blocks if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = response_text.strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response_text[:500]}")
            return None

    # -------------------------------------------------------------------------
    # Post-Processing: Date Normalization
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_date(date_string: str) -> str:
        """
        Attempt to normalize a date string to ISO 8601 (YYYY-MM-DD).
        Falls back to the original string if parsing fails.

        Args:
            date_string: Raw date string

        Returns:
            Normalized date string or original
        """
        if not date_string or not isinstance(date_string, str):
            return date_string

        # Already in ISO format
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_string):
            return date_string

        try:
            from dateutil import parser as date_parser
            parsed = date_parser.parse(date_string, fuzzy=True, dayfirst=True)
            return parsed.strftime("%Y-%m-%d")
        except (ValueError, TypeError, ImportError):
            logger.debug(f"Could not normalize date: '{date_string}'")
            return date_string

    def _normalize_entity_properties(self, entity: Dict) -> Dict:
        """
        Normalize all date properties on an entity to ISO format.

        Args:
            entity: Entity dictionary

        Returns:
            Entity with normalized date properties
        """
        props = entity.get("properties", {})
        for key, value in props.items():
            if key in DATE_PROPERTY_KEYS and isinstance(value, str):
                normalized = self._normalize_date(value)
                if normalized != value:
                    logger.debug(f"Normalized {entity['name']}.{key}: '{value}' -> '{normalized}'")
                props[key] = normalized
        entity["properties"] = props
        return entity

    # -------------------------------------------------------------------------
    # Post-Processing: Validation
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_entity_type(entity: Dict) -> Dict:
        """
        Ensure entity type is in the allowed list.
        Maps unknown types to the closest match or 'Concept' as fallback.
        """
        entity_type = entity.get("type", "").strip()

        # Direct match (case-insensitive)
        for allowed in ENTITY_TYPES:
            if entity_type.lower() == allowed.lower():
                entity["type"] = allowed
                return entity

        # Fuzzy fallback mappings for common LLM drift
        type_mappings = {
            "company": "Organization", "firm": "Organization", "agency": "Organization",
            "university": "Organization", "school": "Organization", "institute": "Organization",
            "tool": "Technology", "software": "Technology", "framework": "Technology",
            "place": "Location", "city": "Location", "country": "Location",
            "deadline": "Date", "time": "Date", "period": "Date",
            "job": "Role", "position": "Role", "title": "Role",
            "requirement": "Standard", "rule": "Policy",
            "goal": "Deliverable", "objective": "Deliverable",
            "certification": "Qualification", "degree": "Qualification",
            "hazard": "Risk", "threat": "Risk",
            "agreement": "Contract",
            "other": "Concept",
        }

        mapped = type_mappings.get(entity_type.lower())
        if mapped:
            logger.debug(f"Mapped entity type '{entity_type}' -> '{mapped}' for '{entity.get('name')}'")
            entity["type"] = mapped
        else:
            logger.warning(f"Unknown entity type '{entity_type}' for '{entity.get('name')}', defaulting to 'Concept'")
            entity["type"] = "Concept"

        return entity

    @staticmethod
    def _validate_relationship_type(relationship: Dict) -> Dict:
        """
        Ensure relationship type is in the allowed list.
        Maps unknown types to the closest match or 'REFERENCES' as a generic fallback.
        """
        rel_type = relationship.get("type", "").strip().upper()

        # Direct match
        if rel_type in RELATIONSHIP_TYPES:
            relationship["type"] = rel_type
            return relationship

        # ---- Hierarchy & Structure ----
        rel_mappings = {
            # -> WORKS_FOR
            "EMPLOYED_BY": "WORKS_FOR", "WORKS_AT": "WORKS_FOR",
            "IS_EMPLOYEE_OF": "WORKS_FOR", "HIRED_BY": "WORKS_FOR",
            # -> REPORTS_TO
            "SUPERVISED_BY": "REPORTS_TO", "ANSWERS_TO": "REPORTS_TO",
            # -> LOCATED_IN
            "BASED_IN": "LOCATED_IN", "SITUATED_IN": "LOCATED_IN",
            "HEADQUARTERED_IN": "LOCATED_IN",
            # -> PART_OF
            "MEMBER_OF": "PART_OF", "SUBSET_OF": "PART_OF",
            "DIVISION_OF": "PART_OF", "INCLUDED_IN": "PART_OF",
            # -> BELONGS_TO
            "IN_TEAM": "BELONGS_TO", "AFFILIATED_WITH": "BELONGS_TO",
            # -> HOLDS_ROLE
            "HAS_ROLE": "HOLDS_ROLE", "SERVES_AS": "HOLDS_ROLE",
            "ACTS_AS": "HOLDS_ROLE",

            # ---- Identity & Skills ----
            # -> HAS_SKILL
            "KNOWS": "HAS_SKILL", "PROFICIENT_IN": "HAS_SKILL",
            "SKILLED_IN": "HAS_SKILL", "EXPERT_IN": "HAS_SKILL",
            # -> TEACHES
            "INSTRUCTS": "TEACHES", "TRAINS": "TEACHES",
            # -> STUDIES
            "ENROLLED_IN": "STUDIES", "TAKING": "STUDIES",
            "LEARNING": "STUDIES",

            # ---- Execution & Projects ----
            # -> MANAGES
            "LEADS": "MANAGES", "SUPERVISES": "MANAGES",
            "OVERSEES": "MANAGES", "HEADS": "MANAGES",
            # -> PARTICIPATED_IN
            "ATTENDED": "PARTICIPATED_IN", "INVOLVED_IN": "PARTICIPATED_IN",
            "TOOK_PART_IN": "PARTICIPATED_IN",
            # -> ASSIGNED
            "ASSIGNED_TO": "ASSIGNED", "DELEGATED_TO": "ASSIGNED",
            "TASKED_WITH": "ASSIGNED",
            # -> CREATED
            "MADE_BY": "CREATED", "BUILT_BY": "CREATED",
            "DEVELOPED_BY": "CREATED", "DESIGNED_BY": "CREATED",
            "BUILT": "CREATED", "DEVELOPED": "CREATED",
            # -> DELIVERED
            "PRODUCED": "DELIVERED", "COMPLETED": "DELIVERED",
            "SUBMITTED": "DELIVERED", "SHIPPED": "DELIVERED",

            # ---- Tasks & Dependencies ----
            # -> REQUIRES
            "NEEDS": "REQUIRES", "DEMANDS": "REQUIRES",
            "PREREQUISITE_FOR": "REQUIRES",
            # -> DEPENDS_ON
            "BLOCKED_BY": "DEPENDS_ON", "RELIANT_ON": "DEPENDS_ON",
            "CONTINGENT_ON": "DEPENDS_ON",
            # -> PRECEDED_BY
            "FOLLOWED_BY": "PRECEDED_BY", "COMES_AFTER": "PRECEDED_BY",
            # -> RESULTED_IN
            "LED_TO": "RESULTED_IN", "CAUSED": "RESULTED_IN",
            "PRODUCED_OUTCOME": "RESULTED_IN",
            # -> MITIGATES
            "ADDRESSES": "MITIGATES", "REDUCES": "MITIGATES",
            "CONTROLS": "MITIGATES",

            # ---- Goals & Alignment ----
            # -> ALIGNS_WITH
            "SUPPORTS": "ALIGNS_WITH", "CONTRIBUTES_TO": "ALIGNS_WITH",
            "IN_LINE_WITH": "ALIGNS_WITH",
            # -> TARGETS
            "AIMS_FOR": "TARGETS", "GOALS": "TARGETS",
            "OBJECTIVES": "TARGETS",

            # ---- Products & Tools ----
            # -> USES_TOOL
            "USES": "USES_TOOL", "UTILIZES": "USES_TOOL",
            "EMPLOYS_TOOL": "USES_TOOL", "RUNS_ON": "USES_TOOL",
            # -> INTEGRATES_WITH
            "CONNECTS_TO": "INTEGRATES_WITH", "INTERFACES_WITH": "INTEGRATES_WITH",
            "COMPATIBLE_WITH": "INTEGRATES_WITH",
            # -> OWNS
            "PURCHASED": "OWNS", "ACQUIRED": "OWNS",
            "HAS_OWNERSHIP": "OWNS",

            # ---- External ----
            # -> SERVES
            "PROVIDES_SERVICE_TO": "SERVES", "SUPPORTS_CLIENT": "SERVES",
            "CATERS_TO": "SERVES",
            # -> PARTNERS_WITH
            "COLLABORATES_WITH": "PARTNERS_WITH", "ALLIED_WITH": "PARTNERS_WITH",
            "JOINT_VENTURE_WITH": "PARTNERS_WITH", "CONTRACTED_WITH": "PARTNERS_WITH",

            # ---- Documents & Auditing ----
            # -> AUTHORED
            "AUTHORED_BY": "AUTHORED", "WROTE": "AUTHORED",
            "WRITTEN_BY": "AUTHORED", "DRAFTED_BY": "AUTHORED",
            "DRAFTED": "AUTHORED",
            # -> APPROVED
            "APPROVED_BY": "APPROVED", "SIGNED_OFF_BY": "APPROVED",
            "AUTHORIZED_BY": "APPROVED", "AUTHORIZED": "APPROVED",
            # -> REVIEWED
            "REVIEWED_BY": "REVIEWED", "ASSESSED_BY": "REVIEWED",
            "EVALUATED_BY": "REVIEWED", "AUDITED_BY": "REVIEWED",
            "EVALUATED": "REVIEWED", "ASSESSED": "REVIEWED",
            # -> CONTAINS
            "HAS": "CONTAINS", "INCLUDES": "CONTAINS",
            "ENCOMPASSES": "CONTAINS",
            # -> REFERENCES
            "CITES": "REFERENCES", "REFERS_TO": "REFERENCES",
            "MENTIONS": "REFERENCES", "LINKS_TO": "REFERENCES",
            "RELATED_TO": "REFERENCES", "ASSOCIATED_WITH": "REFERENCES",
            "IS_ABOUT": "REFERENCES", "CONCERNS": "REFERENCES",
            # -> SUPERSEDES
            "REPLACES": "SUPERSEDES", "OVERRIDES": "SUPERSEDES",
            "UPDATES": "SUPERSEDES", "SUCCEEDS": "SUPERSEDES",
            # -> COMPLIES_WITH
            "ADHERES_TO": "COMPLIES_WITH", "CONFORMS_TO": "COMPLIES_WITH",
            "FOLLOWS": "COMPLIES_WITH", "GOVERNED_BY": "COMPLIES_WITH",
            "IMPLEMENTS": "COMPLIES_WITH",
            # -> CERTIFIES
            "ACCREDITS": "CERTIFIES", "VALIDATES": "CERTIFIES",
            "QUALIFIES": "CERTIFIES",
        }

        mapped = rel_mappings.get(rel_type)
        if mapped:
            logger.debug(f"Mapped relationship type '{rel_type}' -> '{mapped}'")
            relationship["type"] = mapped
        else:
            logger.warning(f"Unknown relationship type '{rel_type}', defaulting to 'REFERENCES'")
            relationship["type"] = "REFERENCES"

        return relationship

    # -------------------------------------------------------------------------
    # Deduplication and Merging
    # -------------------------------------------------------------------------

    @staticmethod
    def _make_entity_key(entity: Dict) -> str:
        """Generate a case-insensitive deduplication key for an entity."""
        name = entity.get("name", "").strip().lower()
        etype = entity.get("type", "").strip().lower()
        return f"{name}||{etype}"

    @staticmethod
    def _merge_entity_properties(existing: Dict, new: Dict) -> Dict:
        """
        Merge properties from a duplicate entity extraction into the existing one.
        Keeps the more informative value for each property.
        """
        existing_props = existing.get("properties", {})
        new_props = new.get("properties", {})

        for key, value in new_props.items():
            if value is None or value == "":
                continue
            if key not in existing_props or not existing_props[key]:
                existing_props[key] = value

        existing["properties"] = existing_props

        # Merge descriptions if the new one adds info
        new_desc = new.get("description", "")
        existing_desc = existing.get("description", "")
        if new_desc and new_desc.lower() not in existing_desc.lower():
            existing["description"] = f"{existing_desc}; {new_desc}" if existing_desc else new_desc

        # Keep the higher confidence
        existing["confidence"] = max(
            existing.get("confidence", 0.5),
            new.get("confidence", 0.5)
        )

        # Track all sources
        existing_sources = existing.get("sources", [])
        new_sources = new.get("sources", [])
        existing["sources"] = list(set(existing_sources + new_sources))

        return existing

    @staticmethod
    def _make_relationship_key(rel: Dict) -> str:
        """Generate a case-insensitive deduplication key for a relationship."""
        source = rel.get("source", "").strip().lower()
        target = rel.get("target", "").strip().lower()
        rtype = rel.get("type", "").strip().upper()
        return f"{source}||{rtype}||{target}"

    # -------------------------------------------------------------------------
    # Core Extraction
    # -------------------------------------------------------------------------

    def extract_entities_and_relationships(
        self, text: str, context: str = None, chunk_index: int = None
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from text.
        Auto-chunks if text exceeds the maximum length.

        Args:
            text: Text to process
            context: Optional context (e.g., filename, source)
            chunk_index: Optional index if this is part of a chunked extraction

        Returns:
            Dictionary containing entities, relationships, and metadata
        """
        if not self.initialized:
            logger.error("Entity extractor not initialized")
            return {"entities": [], "relationships": [], "success": False, "error": "Not initialized"}

        # Auto-chunk if text is too long
        if len(text) > MAX_SINGLE_EXTRACT_LENGTH:
            logger.info(
                f"Text length ({len(text)}) exceeds {MAX_SINGLE_EXTRACT_LENGTH}, auto-chunking..."
            )
            chunks = self.chunk_text(text)
            return self.extract_from_chunks(chunks, context)

        try:
            # Build the prompt with constrained types
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("user", "Context: {context}\n\nText to analyze:\n{text}")
            ])

            messages = prompt_template.format_messages(
                entity_types=", ".join(ENTITY_TYPES),
                relationship_types=", ".join(RELATIONSHIP_TYPES),
                context=context or "Unknown source",
                text=text
            )

            # Call LLM with retry
            response = self._invoke_llm_with_retry(messages)

            # Parse response
            result_dict = self._parse_llm_response(response)
            if result_dict is None:
                return {
                    "entities": [],
                    "relationships": [],
                    "success": False,
                    "error": "Failed to parse LLM response"
                }

            # Post-process entities
            entities = []
            for entity_data in result_dict.get("entities", []):
                # Ensure properties dict exists
                if "properties" not in entity_data:
                    entity_data["properties"] = {}
                if "confidence" not in entity_data:
                    entity_data["confidence"] = 0.8

                # Validate type
                entity_data = self._validate_entity_type(entity_data)

                # Normalize date properties
                entity_data = self._normalize_entity_properties(entity_data)

                # Add source provenance
                entity_data["sources"] = [context or "Unknown source"]
                if chunk_index is not None:
                    entity_data["chunk_index"] = chunk_index

                entities.append(entity_data)

            # Post-process relationships
            relationships = []
            for rel_data in result_dict.get("relationships", []):
                if "confidence" not in rel_data:
                    rel_data["confidence"] = 0.8

                # Validate type
                rel_data = self._validate_relationship_type(rel_data)

                # Add source provenance
                rel_data["source_context"] = context or "Unknown source"
                if chunk_index is not None:
                    rel_data["chunk_index"] = chunk_index

                relationships.append(rel_data)

            return {
                "entities": entities,
                "relationships": relationships,
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

    # -------------------------------------------------------------------------
    # Multi-Chunk Extraction
    # -------------------------------------------------------------------------

    def extract_from_chunks(self, chunks: List[str], context: str = None) -> Dict[str, Any]:
        """
        Extract entities and relationships from multiple text chunks with:
        - Rate-limit protection
        - Cross-chunk entity deduplication with property merging
        - Cross-chunk relationship deduplication
        - Source provenance tracking

        Args:
            chunks: List of text chunks
            context: Optional source context

        Returns:
            Merged extraction result
        """
        all_entities: Dict[str, Dict] = {}
        all_relationships: Dict[str, Dict] = {}
        failed_chunks = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")

            result = self.extract_entities_and_relationships(
                chunk, context=context, chunk_index=i
            )

            if result.get("success"):
                # Deduplicate and merge entities
                for entity in result.get("entities", []):
                    key = self._make_entity_key(entity)
                    if key in all_entities:
                        all_entities[key] = self._merge_entity_properties(
                            all_entities[key], entity
                        )
                    else:
                        all_entities[key] = entity

                # Deduplicate relationships
                for rel in result.get("relationships", []):
                    key = self._make_relationship_key(rel)
                    if key not in all_relationships:
                        all_relationships[key] = rel
                    else:
                        # Keep the higher confidence version
                        existing_conf = all_relationships[key].get("confidence", 0.5)
                        new_conf = rel.get("confidence", 0.5)
                        if new_conf > existing_conf:
                            all_relationships[key] = rel
            else:
                failed_chunks.append(i)
                logger.warning(f"Chunk {i + 1} failed: {result.get('error', 'Unknown error')}")


        result = {
            "entities": list(all_entities.values()),
            "relationships": list(all_relationships.values()),
            "success": True,
            "chunks_processed": len(chunks),
            "chunks_failed": failed_chunks,
        }

        logger.info(
            f"Extraction complete: {len(result['entities'])} entities, "
            f"{len(result['relationships'])} relationships from {len(chunks)} chunks "
            f"({len(failed_chunks)} failed)"
        )

        return result

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def simple_extract(self, text: str, context: str = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Simplified extraction that returns entities and relationships directly.

        Args:
            text: Text to process
            context: Optional source context

        Returns:
            Tuple of (entities, relationships)
        """
        result = self.extract_entities_and_relationships(text, context)
        return result.get("entities", []), result.get("relationships", [])

    def extract_with_auto_chunk(
        self, text: str, context: str = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> Dict[str, Any]:
        """
        Full pipeline: chunk text if needed, extract, deduplicate, normalize.

        Args:
            text: Full text to process
            context: Optional source context
            chunk_size: Characters per chunk
            overlap: Overlap between chunks

        Returns:
            Complete extraction result
        """
        chunks = self.chunk_text(text, chunk_size, overlap)

        if len(chunks) == 1:
            return self.extract_entities_and_relationships(chunks[0], context, chunk_index=0)
        else:
            return self.extract_from_chunks(chunks, context)

    # -------------------------------------------------------------------------
    # Query Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def get_entities_by_type(extraction_result: Dict, entity_type: str) -> List[Dict]:
        """Filter extracted entities by type."""
        return [
            e for e in extraction_result.get("entities", [])
            if e.get("type", "").lower() == entity_type.lower()
        ]

    @staticmethod
    def get_entities_with_deadlines(extraction_result: Dict) -> List[Dict]:
        """Get all entities that have a deadline property."""
        return [
            e for e in extraction_result.get("entities", [])
            if e.get("properties", {}).get("deadline")
        ]

    @staticmethod
    def get_overdue_entities(extraction_result: Dict, reference_date: str = None) -> List[Dict]:
        """
        Get entities whose deadline has passed.

        Args:
            extraction_result: Extraction result dictionary
            reference_date: ISO date string to compare against (defaults to today)

        Returns:
            List of overdue entities
        """
        from datetime import date

        if reference_date:
            today = date.fromisoformat(reference_date)
        else:
            today = date.today()

        overdue = []
        for entity in extraction_result.get("entities", []):
            deadline = entity.get("properties", {}).get("deadline", "")
            status = entity.get("properties", {}).get("status", "").lower()

            if not deadline or status in ("completed", "done", "finished"):
                continue

            try:
                deadline_date = date.fromisoformat(deadline)
                if deadline_date < today:
                    overdue.append(entity)
            except (ValueError, TypeError):
                continue

        return overdue

    @staticmethod
    def get_upcoming_deadlines(extraction_result: Dict, days_ahead: int = 7) -> List[Dict]:
        """
        Get entities with deadlines within the next N days.

        Args:
            extraction_result: Extraction result dictionary
            days_ahead: Number of days to look ahead (default 7)

        Returns:
            List of entities with upcoming deadlines, sorted by deadline
        """
        from datetime import date, timedelta

        today = date.today()
        cutoff = today + timedelta(days=days_ahead)

        upcoming = []
        for entity in extraction_result.get("entities", []):
            deadline = entity.get("properties", {}).get("deadline", "")
            status = entity.get("properties", {}).get("status", "").lower()

            if not deadline or status in ("completed", "done", "finished"):
                continue

            try:
                deadline_date = date.fromisoformat(deadline)
                if today <= deadline_date <= cutoff:
                    upcoming.append(entity)
            except (ValueError, TypeError):
                continue

        # Sort by deadline ascending
        upcoming.sort(key=lambda e: e.get("properties", {}).get("deadline", "9999-12-31"))
        return upcoming

    @staticmethod
    def get_relationships_for_entity(extraction_result: Dict, entity_name: str) -> List[Dict]:
        """Get all relationships involving a specific entity (as source or target)."""
        name_lower = entity_name.strip().lower()
        return [
            r for r in extraction_result.get("relationships", [])
            if r.get("source", "").strip().lower() == name_lower
            or r.get("target", "").strip().lower() == name_lower
        ]

    @staticmethod
    def get_relationships_by_type(extraction_result: Dict, rel_type: str) -> List[Dict]:
        """Filter extracted relationships by type."""
        return [
            r for r in extraction_result.get("relationships", [])
            if r.get("type", "").upper() == rel_type.upper()
        ]


# =============================================================================
# Fallback Extractor (No LLM Required)
# =============================================================================

class SimpleEntityExtractor:
    """
    Fallback entity extractor using spaCy NER with regex augmentation.
    Used when the LLM is not available.
    """

    _nlp = None  # Class-level spaCy model cache

    @classmethod
    def _get_nlp(cls):
        """Lazy-load spaCy model."""
        if cls._nlp is None:
            try:
                import spacy
                try:
                    cls._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning(
                        "spaCy 'en_core_web_sm' model not found. "
                        "Install with: python -m spacy download en_core_web_sm. "
                        "Falling back to regex-only extraction."
                    )
                    cls._nlp = False  # Sentinel: tried and failed
            except ImportError:
                logger.warning(
                    "spaCy not installed. Falling back to regex-only extraction. "
                    "Install with: pip install spacy"
                )
                cls._nlp = False
        return cls._nlp if cls._nlp is not False else None

    # spaCy label -> our entity type mapping
    SPACY_TYPE_MAP = {
        "PERSON": "Person",
        "ORG": "Organization",
        "GPE": "Location",
        "LOC": "Location",
        "DATE": "Date",
        "EVENT": "Event",
        "PRODUCT": "Product",
        "WORK_OF_ART": "Document",
        "LAW": "Regulation",
        "MONEY": "Metric",
        "PERCENT": "Metric",
        "QUANTITY": "Metric",
        "FAC": "Location",
        "NORP": "Organization",
    }

    @classmethod
    def extract_basic_entities(cls, text: str) -> Dict[str, Any]:
        """
        Extract basic entities using spaCy NER with regex augmentation.
        Falls back to pure regex if spaCy is unavailable.

        Args:
            text: Text to extract from

        Returns:
            Dictionary with entities and metadata
        """
        entities = []
        seen_keys = set()

        def _add_entity(name: str, etype: str, desc: str):
            key = f"{name.strip().lower()}||{etype.lower()}"
            if key not in seen_keys:
                seen_keys.add(key)
                entities.append({
                    "name": name.strip(),
                    "type": etype,
                    "description": desc,
                    "confidence": 0.6,
                    "properties": {}
                })

        # ---- spaCy NER pass ----
        nlp = cls._get_nlp()
        method = "regex_only"

        if nlp:
            method = "spacy_ner_with_regex"
            doc = nlp(text[:100000])  # spaCy limit
            for ent in doc.ents:
                etype = cls.SPACY_TYPE_MAP.get(ent.label_, "Concept")
                _add_entity(ent.text, etype, f"{etype} identified by NER")

        # ---- Regex augmentation (always runs) ----

        # Organizations: Capitalized words + Inc/Corp/LLC/Ltd/Company/University/Institute
        org_pattern = (
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+'
            r'(?:Inc|Corp|LLC|Ltd|Company|University|Institute|Foundation|Association)\b'
        )
        for org in set(re.findall(org_pattern, text)):
            _add_entity(org, "Organization", "Organization (regex match)")

        # Persons: Title + Name
        person_pattern = r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        for person in set(re.findall(person_pattern, text)):
            _add_entity(person, "Person", "Person (regex match)")

        # Dates: various formats
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|'
            r'October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|'
            r'September|October|November|December)\s+\d{4}\b',
        ]
        for pattern in date_patterns:
            for date_match in set(re.findall(pattern, text)):
                _add_entity(date_match, "Date", "Date (regex match)")

        # Emails -> Person hint
        email_pattern = r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b'
        for email in set(re.findall(email_pattern, text)):
            _add_entity(email, "Person", f"Email contact: {email}")

        # URLs -> Resource hint
        url_pattern = r'https?://[^\s<>\"\']+\b'
        for url in set(re.findall(url_pattern, text)):
            _add_entity(url, "Resource", f"Web resource: {url}")

        return {
            "entities": entities,
            "relationships": [],
            "success": True,
            "method": method
        }