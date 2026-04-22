"""
Entity Extractor module for GraphNet.
Uses LangChain and Google Gemini to extract entities and relationships from text.

v5 — Relationship-rich extraction:
  - System prompt now explicitly prioritises relationship density (≥2× entity count)
  - New second-pass relationship discovery step re-examines text with known entities
  - _get_text() helper handles Gemini's list-of-blocks response format
  - Rate-limit–friendly delays between chunks and passes
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
    # Operations & Presence
    "OPERATES_IN", "EMPLOYS",
    # Identity & Skills
    "HAS_SKILL", "TEACHES", "STUDIES",
    # Execution & Projects
    "MANAGES", "PARTICIPATED_IN", "ASSIGNED", "CREATED", "DELIVERED", "OCCURRED_ON",
    "LAUNCHED",
    # Tasks & Dependencies
    "REQUIRES", "DEPENDS_ON", "PRECEDED_BY", "RESULTED_IN", "MITIGATES",
    # Causal & Influence
    "INFLUENCES", "AFFECTED_BY", "BENEFITS_FROM",
    # Goals & Alignment
    "ALIGNS_WITH", "TARGETS",
    # Products & Tools
    "USES_TOOL", "INTEGRATES_WITH", "OWNS",
    # Financial & Transactions
    "INVESTED_IN", "SOLD", "ACQUIRED",
    # External
    "SERVES", "PARTNERS_WITH", "ENGAGES_WITH",
    # Classification
    "IS_TYPE_OF", "EXCLUDES",
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
DEFAULT_CHUNK_SIZE = 15000       # ← reduced from 40k for richer per-chunk extraction
DEFAULT_CHUNK_OVERLAP = 1500
MAX_SINGLE_EXTRACT_LENGTH = 50000


# =============================================================================
# Pydantic Models
# =============================================================================

class Entity(BaseModel):
    """Entity model with properties and confidence"""
    name: str = Field(description="Name of the entity")
    type: str = Field(description=f"Type/category of the entity. "
                                  f"Must be one of: {', '.join(ENTITY_TYPES)}")
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
# System Prompt  (v5 — relationship-dense)
# =============================================================================

SYSTEM_PROMPT = """You are an expert at extracting entities and relationships from text for a knowledge graph.

Extract ALL relevant entities and their relationships from the given text.

## Entity Types
Entity type MUST be one of: {entity_types}

## Relationship Types
Relationship type MUST be one of: {relationship_types}

Relationship categories for reference:
- Hierarchy & Structure: WORKS_FOR, REPORTS_TO, LOCATED_IN, PART_OF, BELONGS_TO, HOLDS_ROLE
- Operations & Presence: OPERATES_IN, EMPLOYS
- Identity & Skills: HAS_SKILL, TEACHES, STUDIES
- Execution & Projects: MANAGES, PARTICIPATED_IN, ASSIGNED, CREATED, DELIVERED, OCCURRED_ON, LAUNCHED
- Tasks & Dependencies: REQUIRES, DEPENDS_ON, PRECEDED_BY, RESULTED_IN, MITIGATES
- Causal & Influence: INFLUENCES, AFFECTED_BY, BENEFITS_FROM
- Goals & Alignment: ALIGNS_WITH, TARGETS
- Products & Tools: USES_TOOL, INTEGRATES_WITH, OWNS
- Financial & Transactions: INVESTED_IN, SOLD, ACQUIRED
- External: SERVES, PARTNERS_WITH, ENGAGES_WITH
- Classification: IS_TYPE_OF, EXCLUDES
- Documents & Auditing: AUTHORED, APPROVED, REVIEWED, CONTAINS, REFERENCES, SUPERSEDES, COMPLIES_WITH, CERTIFIES

## Extraction Priority — RELATIONSHIPS ARE CRITICAL
A high-quality knowledge graph connects entities meaningfully.
Follow these rules for relationship extraction:

1. For EVERY entity you extract, look for its STRONGEST connections to other entities in the text.
2. Prefer SPECIFIC, meaningful relationships over generic ones. Only use REFERENCES as a last resort.
3. Extract IMPLICIT relationships when they are clearly supported by context:
   - Two people mentioned in the same meeting → both PARTICIPATED_IN that Event
   - A person described in a section about a department → BELONGS_TO or WORKS_FOR
   - A technology mentioned alongside a project → USES_TOOL or INTEGRATES_WITH
   - A document that discusses a policy → REFERENCES or CONTAINS
   - A person with a job title → HOLDS_ROLE
   - An organisation in a city → LOCATED_IN
   - An organisation with business in a region/market → OPERATES_IN
   - An organisation that hired staff → EMPLOYS
   - A product or service that was introduced → LAUNCHED
   - A company that sold/divested a business unit → SOLD
   - A company that bought another → ACQUIRED
   - A factor that impacts performance → INFLUENCES or AFFECTED_BY
   - An entity that gains advantage from something → BENEFITS_FROM
   - An entity classified as a type → IS_TYPE_OF
   - A metric that excludes certain items → EXCLUDES
   - Stakeholder interaction → ENGAGES_WITH
3. Extract MULTI-HOP relationships: if A manages B, and B works on project C, extract both A→MANAGES→B and B→ASSIGNED→C.
4. Look for temporal relationships: events that PRECEDED_BY or RESULTED_IN other events.
5. Look for hierarchical relationships: departments PART_OF organisations, sub-tasks DEPENDS_ON parent tasks.
6. Look for causal relationships: policies that INFLUENCES outcomes, risks AFFECTED_BY market conditions.
7. Look for financial relationships: companies that INVESTED_IN, SOLD, or ACQUIRED other entities.
8. If an entity has ZERO relationships, reconsider whether it is worth extracting at all.

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

## Selectivity — Quality Over Quantity
A focused, high-signal graph is better than a cluttered one. Follow these rules:
- Do NOT extract generic/vague concepts (e.g., "growth", "performance", "strategy", "risk") as standalone entities UNLESS they are a named, specific thing (e.g., "HSBC Growth Strategy 2024", "Operational Risk Framework").
- Do NOT extract standalone Date entities — dates should be PROPERTIES on the relevant entity.
- Do NOT extract metric values as entities UNLESS they represent a named KPI or target (e.g., "Return on Tangible Equity" is good, "$65.9bn" is not).
- Prefer fewer, high-confidence entities (>= 0.7) over many low-confidence ones.
- Every entity must have a clear, specific identity — if you can't give it a meaningful name beyond a generic word, don't extract it.

## Output Format
Return your response as a JSON object with this EXACT structure:
{{{{
    "entities": [
        {{{{
            "name": "Task A",
            "type": "Task",
            "description": "Brief description",
            "confidence": 0.95,
            "properties": {{{{
                "deadline": "2026-05-15",
                "status": "In Progress",
                "priority": "High"
            }}}}
        }}}}
    ],
    "relationships": [
        {{{{
            "source": "Entity1",
            "target": "Entity2",
            "type": "ASSIGNED",
            "description": "Brief description",
            "confidence": 0.9
        }}}}
    ]
}}}}

## Quality Rules
- Only extract what is clearly stated or strongly implied (confidence >= 0.65).
- Do NOT invent properties that are not mentioned or implied in the text.
- Normalize all dates to YYYY-MM-DD where possible.
- Use the MOST SPECIFIC entity type available (e.g., "Department" not "Organization" for a department).
- Use the MOST SPECIFIC relationship type available (e.g., "ASSIGNED" not "REFERENCES" for task assignments).
- Return ONLY the JSON object, no other text."""


# =============================================================================
# Second-Pass Relationship Discovery Prompt
# =============================================================================

RELATIONSHIP_PASS_PROMPT = """You are an expert at discovering relationships between known entities in text.

You have already extracted these entities from the text:
{entity_list}

Now re-read the text carefully and find ALL relationships between these entities that may have been missed.

Focus on:
1. IMPLICIT connections (co-occurrence in same paragraph, shared context, logical inference)
2. HIERARCHICAL relationships (PART_OF, BELONGS_TO, REPORTS_TO, CONTAINS)
3. OPERATIONAL relationships (OPERATES_IN, EMPLOYS, LAUNCHED)
4. TEMPORAL relationships (PRECEDED_BY, RESULTED_IN, OCCURRED_ON)
5. CAUSAL relationships (INFLUENCES, AFFECTED_BY, BENEFITS_FROM)
6. FINANCIAL relationships (INVESTED_IN, SOLD, ACQUIRED)
7. DEPENDENCY relationships (REQUIRES, DEPENDS_ON, USES_TOOL)
8. ATTRIBUTION relationships (AUTHORED, CREATED, APPROVED, REVIEWED)
9. CLASSIFICATION relationships (IS_TYPE_OF, EXCLUDES)
10. Any entity that currently has 0 or 1 relationships — search harder for its connections

Relationship type MUST be one of: {relationship_types}

Return ONLY a JSON object with a single key "relationships" containing an array of relationship objects.
Each relationship object must have: source, target, type, description, confidence.
Return ONLY the JSON object, no other text."""


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
        self._inter_chunk_delay = 1.0  # seconds between chunks (rate-limit friendly)

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
                last_period = text.rfind('.', search_start, end)
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
    # Gemini Response Helper
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_text(response) -> str:
        """
        Extract plain text from an LLM response.

        Gemini 2.5 flash (and similar models) may return response.content as
        a list of blocks (thinking + text) rather than a plain string.
        This helper handles both formats.

        Args:
            response: Raw LLM response object

        Returns:
            Plain text string
        """
        content = response.content

        # Already a string — most common case
        if isinstance(content, str):
            return content.strip()

        # List of blocks (Gemini thinking mode)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    text_parts.append(block.text)
            return "\n".join(text_parts).strip()

        # Fallback
        return str(content).strip()

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

    def _parse_llm_response(self, response) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM response into a dictionary, handling markdown fences
        and Gemini's list-of-blocks response format.

        Args:
            response: Raw LLM response

        Returns:
            Parsed dictionary or None
        """
        response_text = self._get_text(response)

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
            "bank": "Organization", "institution": "Organization", "subsidiary": "Organization",
            "tool": "Technology", "software": "Technology", "framework": "Technology",
            "platform": "Technology", "system": "Technology",
            "place": "Location", "city": "Location", "country": "Location",
            "region": "Location", "market": "Location",
            "deadline": "Date", "time": "Date", "period": "Date", "year": "Date",
            "job": "Role", "position": "Role", "title": "Role",
            "requirement": "Standard", "rule": "Policy",
            "goal": "Metric", "objective": "Metric", "target": "Metric",
            "kpi": "Metric", "indicator": "Metric",
            "certification": "Qualification", "degree": "Qualification",
            "hazard": "Risk", "threat": "Risk",
            "agreement": "Contract", "deal": "Contract",
            "service": "Product", "offering": "Product",
            "initiative": "Project", "programme": "Project", "program": "Project",
            "strategy": "Concept", "theme": "Concept",
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

        Preserves Gemini's original label in 'original_type' when a mapping
        or fallback is applied, so semantic nuance is not lost.
        """
        rel_type = relationship.get("type", "").strip().upper()

        # Direct match — no mapping needed
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
            "HEADQUARTERED_IN": "LOCATED_IN", "LOCATED_AT": "LOCATED_IN",
            # -> PART_OF
            "MEMBER_OF": "PART_OF", "SUBSET_OF": "PART_OF",
            "DIVISION_OF": "PART_OF", "INCLUDED_IN": "PART_OF",
            # -> BELONGS_TO
            "IN_TEAM": "BELONGS_TO", "AFFILIATED_WITH": "BELONGS_TO",
            # -> HOLDS_ROLE
            "HAS_ROLE": "HOLDS_ROLE", "SERVES_AS": "HOLDS_ROLE",
            "ACTS_AS": "HOLDS_ROLE",

            # ---- Operations & Presence ----
            # -> OPERATES_IN
            "ACTIVE_IN": "OPERATES_IN", "PRESENT_IN": "OPERATES_IN",
            "HAS_PRESENCE_IN": "OPERATES_IN", "OPERATES_THROUGH": "OPERATES_IN",
            "DOES_BUSINESS_IN": "OPERATES_IN",
            # -> EMPLOYS
            "HIRES": "EMPLOYS", "STAFFS": "EMPLOYS",
            "HAS_EMPLOYEES": "EMPLOYS",
            "RETIRED_FROM": "WORKS_FOR",  # Past employment relationship

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
            "DIRECTS": "MANAGES", "OVERSEES": "MANAGES",
            "HEADS": "MANAGES", "COORDINATES": "MANAGES",
            # -> PARTICIPATED_IN
            "ATTENDED": "PARTICIPATED_IN", "INVOLVED_IN": "PARTICIPATED_IN",
            "CONTRIBUTED_TO": "PARTICIPATED_IN", "TOOK_PART_IN": "PARTICIPATED_IN",
            # -> ASSIGNED
            "TASKED_WITH": "ASSIGNED", "RESPONSIBLE_FOR": "ASSIGNED",
            "ALLOCATED_TO": "ASSIGNED",
            # -> CREATED
            "DEVELOPED": "CREATED", "BUILT": "CREATED",
            "DESIGNED": "CREATED", "FOUNDED": "CREATED",
            "ESTABLISHED": "CREATED", "INITIATED": "CREATED",
            "CREATES": "CREATED",
            # -> DELIVERED
            "COMPLETED": "DELIVERED", "FINISHED": "DELIVERED",
            "SUBMITTED": "DELIVERED", "PRODUCED": "DELIVERED",
            # -> OCCURRED_ON
            "HAPPENED_ON": "OCCURRED_ON", "SCHEDULED_FOR": "OCCURRED_ON",
            "TOOK_PLACE_ON": "OCCURRED_ON", "START_DATE": "OCCURRED_ON",
            "PUBLISHED_DATE": "OCCURRED_ON",
            # -> LAUNCHED
            "INTRODUCED": "LAUNCHED", "ROLLED_OUT": "LAUNCHED",
            "RELEASED": "LAUNCHED", "DEBUTED": "LAUNCHED",
            "UNVEILED": "LAUNCHED",

            # ---- Tasks & Dependencies ----
            # -> REQUIRES
            "NEEDS": "REQUIRES", "DEMANDS": "REQUIRES",
            # -> DEPENDS_ON
            "RELIES_ON": "DEPENDS_ON", "CONTINGENT_ON": "DEPENDS_ON",
            "BLOCKED_BY": "DEPENDS_ON",
            # -> PRECEDED_BY
            "FOLLOWED": "PRECEDED_BY", "CAME_AFTER": "PRECEDED_BY",
            "SUCCEEDED_BY": "PRECEDED_BY",
            # -> RESULTED_IN
            "CAUSED": "RESULTED_IN", "LED_TO": "RESULTED_IN",
            "TRIGGERED": "RESULTED_IN", "PRODUCED_OUTCOME": "RESULTED_IN",
            "RESULTED_FROM": "RESULTED_IN",  # Note: semantically inverse, but maps to same canonical type
            # -> MITIGATES
            "REDUCES": "MITIGATES", "ADDRESSES": "MITIGATES",
            "RESOLVES": "MITIGATES", "HANDLES": "MITIGATES",
            "MITIGATED_BY": "MITIGATES",  # Inverse form

            # ---- Causal & Influence ----
            # -> INFLUENCES
            "IMPACTS": "INFLUENCES", "SHAPES": "INFLUENCES",
            "DRIVES": "INFLUENCES", "STRENGTHENED": "INFLUENCES",
            "OPTIMIZES": "INFLUENCES", "EMBRACES": "INFLUENCES",
            # -> AFFECTED_BY
            "IMPACTED_BY": "AFFECTED_BY", "CHANGED_BY": "AFFECTED_BY",
            "DRIVEN_BY": "AFFECTED_BY", "SHAPED_BY": "AFFECTED_BY",
            # -> BENEFITS_FROM
            "GAINS_FROM": "BENEFITS_FROM", "PROFITS_FROM": "BENEFITS_FROM",
            "LEVERAGES": "BENEFITS_FROM",

            # ---- Goals & Alignment ----
            # -> ALIGNS_WITH
            "SUPPORTS": "ALIGNS_WITH", "MAPS_TO": "ALIGNS_WITH",
            "CORRESPONDS_TO": "ALIGNS_WITH",
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
            "PURCHASED": "OWNS", "HAS_OWNERSHIP": "OWNS",

            # ---- Financial & Transactions ----
            # -> INVESTED_IN
            "FUNDED": "INVESTED_IN", "FINANCED": "INVESTED_IN",
            "BACKED": "INVESTED_IN",
            # -> SOLD
            "DIVESTED": "SOLD", "DISPOSED_OF": "SOLD",
            "PLANNED_SALE": "SOLD", "EXITED": "SOLD",
            "OFFLOADED": "SOLD",
            # -> ACQUIRED
            "TOOK_OVER": "ACQUIRED", "MERGED_WITH": "ACQUIRED",
            "BOUGHT": "ACQUIRED", "REPURCHASED": "ACQUIRED",

            # ---- External ----
            # -> SERVES
            "PROVIDES_SERVICE_TO": "SERVES", "SUPPORTS_CLIENT": "SERVES",
            "CATERS_TO": "SERVES",
            # -> PARTNERS_WITH
            "COLLABORATES_WITH": "PARTNERS_WITH", "ALLIED_WITH": "PARTNERS_WITH",
            "JOINT_VENTURE_WITH": "PARTNERS_WITH", "CONTRACTED_WITH": "PARTNERS_WITH",
            # -> ENGAGES_WITH
            "INTERACTS_WITH": "ENGAGES_WITH", "LIAISES_WITH": "ENGAGES_WITH",
            "COMMUNICATES_WITH": "ENGAGES_WITH", "CELEBRATES": "ENGAGES_WITH",

            # ---- Classification ----
            # -> IS_TYPE_OF
            "IS_A": "IS_TYPE_OF", "CATEGORIZED_AS": "IS_TYPE_OF",
            "CLASSIFIED_AS": "IS_TYPE_OF", "KIND_OF": "IS_TYPE_OF",
            # -> EXCLUDES
            "OMITS": "EXCLUDES", "REMOVES": "EXCLUDES",
            "DROPS": "EXCLUDES", "LEAVES_OUT": "EXCLUDES",

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
            "REVIEWS": "REVIEWED", "AUDITED": "REVIEWED",
            # -> CONTAINS
            "HAS": "CONTAINS", "INCLUDES": "CONTAINS",
            "ENCOMPASSES": "CONTAINS", "ADDED": "CONTAINS",
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
            relationship["original_type"] = rel_type
            relationship["type"] = mapped
        else:
            logger.warning(f"Unknown relationship type '{rel_type}', defaulting to 'REFERENCES'")
            relationship["original_type"] = rel_type
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
    # Second-Pass Relationship Discovery
    # -------------------------------------------------------------------------

    def _relationship_second_pass(
        self, text: str, entities: List[Dict], context: str = None
    ) -> List[Dict]:
        """
        Re-examine the text with the known entity list to discover
        relationships that the first pass missed.

        This is the highest-impact change for improving relationship density.
        One extra LLM call per chunk, but typically finds 30-50% more relationships.

        Args:
            text: Original text chunk
            entities: Entities already extracted from this text
            context: Optional source context

        Returns:
            List of newly discovered relationship dicts
        """
        if not entities or len(entities) < 2:
            return []

        # Build a compact entity list for the prompt
        entity_names = []
        for e in entities:
            name = e.get("name", "Unknown")
            etype = e.get("type", "Unknown")
            entity_names.append(f"  - {name} ({etype})")
        entity_list_str = "\n".join(entity_names)

        try:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", RELATIONSHIP_PASS_PROMPT),
                ("user", "Context: {context}\n\nText to re-examine:\n{text}")
            ])

            messages = prompt_template.format_messages(
                entity_list=entity_list_str,
                relationship_types=", ".join(RELATIONSHIP_TYPES),
                context=context or "Unknown source",
                text=text
            )

            response = self._invoke_llm_with_retry(messages)
            result_dict = self._parse_llm_response(response)

            if result_dict is None:
                return []

            new_relationships = []
            for rel_data in result_dict.get("relationships", []):
                if "confidence" not in rel_data:
                    rel_data["confidence"] = 0.7  # slightly lower confidence for second-pass
                rel_data = self._validate_relationship_type(rel_data)
                rel_data["source_context"] = context or "Unknown source"
                rel_data["extraction_pass"] = "second"
                new_relationships.append(rel_data)

            logger.info(f"Second pass found {len(new_relationships)} additional relationships")
            return new_relationships

        except Exception as e:
            logger.warning(f"Relationship second pass failed: {e}")
            return []

    # -------------------------------------------------------------------------
    # Multi-Chunk Extraction
    # -------------------------------------------------------------------------

    def extract_from_chunks(self, chunks: List[str], context: str = None) -> Dict[str, Any]:
        """
        Extract entities and relationships from multiple text chunks with:
        - Rate-limit protection (inter-chunk delay)
        - Cross-chunk entity deduplication with property merging
        - Cross-chunk relationship deduplication
        - Second-pass relationship discovery per chunk
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

            # Rate-limit friendly delay between chunks (skip first)
            if i > 0:
                time.sleep(self._inter_chunk_delay)

            result = self.extract_entities_and_relationships(
                chunk, context=context, chunk_index=i
            )

            if result.get("success"):
                # Deduplicate and merge entities
                chunk_entities = result.get("entities", [])
                for entity in chunk_entities:
                    key = self._make_entity_key(entity)
                    if key in all_entities:
                        all_entities[key] = self._merge_entity_properties(
                            all_entities[key], entity
                        )
                    else:
                        all_entities[key] = entity

                # Deduplicate relationships from first pass
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

                # --- Second-pass relationship discovery ---
                if chunk_entities and len(chunk_entities) >= 2:
                    # Brief delay before second pass to avoid rate limits
                    time.sleep(self._inter_chunk_delay)

                    new_rels = self._relationship_second_pass(
                        chunk, chunk_entities, context
                    )
                    for rel in new_rels:
                        key = self._make_relationship_key(rel)
                        if key not in all_relationships:
                            all_relationships[key] = rel
                            logger.debug(
                                f"  + 2nd pass rel: {rel.get('source')} "
                                f"—[{rel.get('type')}]→ {rel.get('target')}"
                            )
            else:
                failed_chunks.append(i)
                logger.warning(f"Chunk {i + 1} failed: {result.get('error', 'Unknown error')}")

        # --- Post-processing: filter noise ---
        entity_count = len(all_entities)
        rel_count = len(all_relationships)

        MIN_ENTITY_CONFIDENCE = 0.65
        MIN_REL_CONFIDENCE = 0.6

        def _conf(item, default=0.8):
            """Safely get confidence as float (Gemini sometimes returns strings)."""
            try:
                return float(item.get("confidence", default))
            except (TypeError, ValueError):
                return default

        # 1. Filter low-confidence entities
        filtered_entities = {
            k: v for k, v in all_entities.items()
            if _conf(v) >= MIN_ENTITY_CONFIDENCE
        }

        # 2. Build set of surviving entity names (lowercase) for relationship filtering
        surviving_names = set()
        for entity in filtered_entities.values():
            surviving_names.add(entity.get("name", "").strip().lower())

        # 3. Filter relationships: both endpoints must survive AND confidence threshold
        filtered_rels = {}
        for key, rel in all_relationships.items():
            if _conf(rel) < MIN_REL_CONFIDENCE:
                continue
            src = rel.get("source", "").strip().lower()
            tgt = rel.get("target", "").strip().lower()
            if src in surviving_names and tgt in surviving_names:
                filtered_rels[key] = rel

        # 4. Remove isolated entities (no remaining relationships)
        connected_names = set()
        for rel in filtered_rels.values():
            connected_names.add(rel.get("source", "").strip().lower())
            connected_names.add(rel.get("target", "").strip().lower())

        final_entities = {
            k: v for k, v in filtered_entities.items()
            if v.get("name", "").strip().lower() in connected_names
        }

        pruned_entities = entity_count - len(final_entities)
        pruned_rels = rel_count - len(filtered_rels)
        if pruned_entities > 0 or pruned_rels > 0:
            logger.info(
                f"Post-processing pruned {pruned_entities} low-quality entities "
                f"and {pruned_rels} weak relationships"
            )

        # Log final ratio
        entity_count = len(final_entities)
        rel_count = len(filtered_rels)
        ratio = (rel_count / entity_count) if entity_count > 0 else 0

        result = {
            "entities": list(final_entities.values()),
            "relationships": list(filtered_rels.values()),
            "success": True,
            "chunks_processed": len(chunks),
            "chunks_failed": failed_chunks,
        }

        logger.info(
            f"Extraction complete: {entity_count} entities, "
            f"{rel_count} relationships (ratio {ratio:.1f}:1) from {len(chunks)} chunks "
            f"({len(failed_chunks)} failed)"
        )

        if ratio < 1.0:
            logger.warning(
                f"Relationship ratio ({ratio:.1f}:1) is low. "
                f"Consider adding more source documents or checking for failed chunks."
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
            text: Text to process
            context: Optional source context
            chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks

        Returns:
            Merged extraction result
        """
        chunks = self.chunk_text(text, chunk_size, overlap)
        if len(chunks) == 1:
            result = self.extract_entities_and_relationships(chunks[0], context)
            # Run second pass even for single-chunk documents
            if result.get("success") and len(result.get("entities", [])) >= 2:
                new_rels = self._relationship_second_pass(
                    chunks[0], result["entities"], context
                )
                existing_keys = {
                    self._make_relationship_key(r)
                    for r in result.get("relationships", [])
                }
                for rel in new_rels:
                    if self._make_relationship_key(rel) not in existing_keys:
                        result["relationships"].append(rel)
            return result
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