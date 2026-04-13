"""
Query Agent module for GraphNet.
Handles natural language queries and generates Cypher queries for Neo4j.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from config import config
from graph.graph_manager import GraphManager
from langchain_google_genai import ChatGoogleGenerativeAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryAgent:
    """Agent for processing natural language queries against the knowledge graph"""

    def __init__(self, graph_manager: GraphManager):
        """
        Initialize the query agent

        Args:
            graph_manager: GraphManager instance
        """
        self.graph_manager = graph_manager
        self.llm = None
        self.initialized = False

    def initialize(self) -> bool:
        """
        Initialize the LLM for query processing

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
            logger.info(f"Query agent initialized with Gemini ({config.AI_MODEL})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            return False

    def generate_cypher_query(self, natural_language_query: str) -> Optional[str]:
        """
        Convert natural language query to Cypher query

        Args:
            natural_language_query: User's natural language query

        Returns:
            Cypher query string or None
        """
        if not self.initialized:
            logger.error("Query agent not initialized")
            return None

        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an expert at converting natural language questions 
into Neo4j Cypher queries. Generate ONLY the Cypher query, no explanations.

CRITICAL RULES FOR QUERY GENERATION:
1. ALWAYS use case-insensitive matching with toLower() and CONTAINS for searching by name.
   - CORRECT: WHERE toLower(n.name) CONTAINS toLower("dangote")
   - WRONG: WHERE n.name = "Dangote Group"
2. When searching for an entity, search ALL nodes regardless of label.
   - CORRECT: MATCH (n) WHERE toLower(n.name) CONTAINS toLower("dangote")
   - WRONG: MATCH (n:Organization {name: "Dangote Group"})
3. To find relationships and connected entities, use this pattern:
   MATCH (n)-[r]-(m) WHERE toLower(n.name) CONTAINS toLower("search term") RETURN n, r, m
4. Always include LIMIT to prevent returning too many results.
5. ALWAYS return n.source (the source document) in your query results. This is critical for traceability.
6. When asked for "information about X", return the entity AND all its relationships WITH sources:
   MATCH (n)-[r]-(m) WHERE toLower(n.name) CONTAINS toLower("X") RETURN n.name AS entity, n.source AS source, type(r) AS relationship, m.name AS connected_entity, m.source AS connected_source, labels(n) AS entity_type, labels(m) AS connected_type LIMIT 25
7. When asked to "show all" or "list all" of a type, match by label and include source:
   MATCH (n:Person) RETURN n.name AS name, n.source AS source, n.description AS description, labels(n) AS type LIMIT 25
8. For counting queries:
   MATCH (n:Organization) RETURN count(n) AS count
9. For single entity queries without relationships:
   MATCH (n) WHERE toLower(n.name) CONTAINS toLower("X") RETURN n.name AS name, n.source AS source, n.description AS description, labels(n) AS type LIMIT 10

Common query patterns:
- "Tell me about X" → Find X and all its connections with sources
- "What is related to X" → Find all nodes connected to X with sources
- "Find all people" → MATCH (n:Person) RETURN n.name AS name, n.source AS source, labels(n) AS type LIMIT 25
- "How are X and Y connected" → MATCH path = shortestPath((a)-[*]-(b)) WHERE toLower(a.name) CONTAINS toLower("X") AND toLower(b.name) CONTAINS toLower("Y") RETURN path LIMIT 10
- "Show relationships for X" → MATCH (n)-[r]-(m) WHERE toLower(n.name) CONTAINS toLower("X") RETURN n.name AS entity, n.source AS source, type(r) AS relationship, m.name AS connected_entity, m.source AS connected_source LIMIT 25

Entity types in the graph: Person, Organization, Location, Concept, Product, Date, Event, Technology, Process, Other
Relationship types: WORKS_FOR, LOCATED_IN, RELATED_TO, OWNS, CREATED, MANAGES, PARTICIPATED_IN, PRODUCES, OPERATES_IN, SUBSIDIARY_OF, FOUNDED, HAS, IS_A, PART_OF, SUPPLIES, INVESTED_IN

Generate only the Cypher query without any markdown formatting or explanations."""),
                HumanMessage(content=f"Convert this question to Cypher: {natural_language_query}")
            ])

            response = self.llm.invoke(prompt.format_messages())
            cypher_query = response.content.strip()

            # Clean up the query
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()

            logger.info(f"Generated Cypher query: {cypher_query}")
            return cypher_query

        except Exception as e:
            logger.error(f"Error generating Cypher query: {str(e)}")
            return None

    def process_query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Process a natural language query and return results

        Args:
            natural_language_query: User's natural language query

        Returns:
            Dictionary containing query results and metadata
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "Query agent not initialized",
                "results": []
            }

        try:
            # Generate Cypher query
            cypher_query = self.generate_cypher_query(natural_language_query)

            if not cypher_query:
                return {
                    "success": False,
                    "error": "Could not generate Cypher query",
                    "results": []
                }

            # Execute query
            results = self.graph_manager.query_graph(cypher_query)

            # If no results, try a broader fallback search
            if not results:
                logger.info("No results from generated query, trying fallback search...")
                fallback_query = self._generate_fallback_query(natural_language_query)
                if fallback_query:
                    results = self.graph_manager.query_graph(fallback_query)
                    if results:
                        cypher_query = fallback_query

            # Extract unique source documents from results
            sources = self._extract_sources(results)

            # Generate explanation
            explanation = self.explain_results(
                natural_language_query,
                cypher_query,
                results,
                sources
            )

            return {
                "success": True,
                "query": cypher_query,
                "results": results,
                "explanation": explanation,
                "result_count": len(results),
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    def _extract_sources(self, results: List[Dict]) -> List[str]:
        """
        Extract unique source document names from query results.

        Args:
            results: Query results

        Returns:
            List of unique source document names
        """
        sources = set()
        for record in results:
            for key, value in record.items():
                # Check string values for source fields
                if 'source' in key.lower() and isinstance(value, str) and value:
                    sources.add(value)
                # Check if the value is a node dict with a source property
                elif isinstance(value, dict) and value.get('source'):
                    sources.add(value['source'])
        return sorted(list(sources))

    def _generate_fallback_query(self, natural_language_query: str) -> Optional[str]:
        """
        Generate a broad fallback query by extracting key terms
        and doing a simple CONTAINS search across all nodes and relationships.

        Args:
            natural_language_query: Original query

        Returns:
            Fallback Cypher query or None
        """
        try:
            # Ask LLM to extract the main search term
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Extract the main entity or search term from this question.
Return ONLY the search term, nothing else. No quotes, no explanation.
For example:
- "Tell me about Dangote Group" → dangote
- "Where is Apple located?" → apple
- "Who works for Microsoft?" → microsoft
- "Show me all people" → return NONE
- "List all organizations" → return NONE"""),
                HumanMessage(content=natural_language_query)
            ])

            response = self.llm.invoke(prompt.format_messages())
            search_term = response.content.strip().lower()

            if search_term and search_term != "none":
                return f"""
                MATCH (n)-[r]-(m) 
                WHERE toLower(n.name) CONTAINS "{search_term}" 
                RETURN n.name AS entity, n.source AS source, labels(n) AS entity_type,
                       type(r) AS relationship, 
                       m.name AS connected_entity, m.source AS connected_source, labels(m) AS connected_type
                LIMIT 25
                """
            return None

        except Exception as e:
            logger.error(f"Error generating fallback query: {str(e)}")
            return None

    def explain_results(self, natural_query: str, cypher_query: str,
                       results: List[Dict], sources: List[str] = None) -> str:
        """
        Generate a natural language explanation of query results

        Args:
            natural_query: Original natural language query
            cypher_query: Generated Cypher query
            results: Query results
            sources: List of source documents

        Returns:
            Natural language explanation
        """
        if not self.initialized:
            return "Query agent not initialized"

        try:
            source_info = ""
            if sources:
                source_info = f"\n\nSource documents this information came from: {', '.join(sources)}"

            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are explaining query results from a knowledge graph. 
Provide a clear, concise explanation of what was found. Be specific and mention entity names.
If relationships were found, describe the connections between entities.
If no results were found, say so clearly and suggest alternative search terms.

IMPORTANT: Always mention which source document(s) the information came from at the end of your explanation.
Format it as: "📄 Source: [document name]" 
If multiple sources, list them all."""),
                HumanMessage(content=f"""
Original question: {natural_query}

Query executed: {cypher_query}

Results found: {len(results)}

Sample results: {str(results[:5]) if results else "No results"}
{source_info}

Provide a brief, natural explanation of these results. Always mention the source document(s).""")
            ])

            response = self.llm.invoke(prompt.format_messages())
            return response.content.strip()

        except Exception as e:
            logger.error(f"Error explaining results: {str(e)}")
            return f"Found {len(results)} results"

    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """
        Get detailed information about an entity

        Args:
            entity_name: Name of the entity

        Returns:
            Dictionary with entity information
        """
        try:
            # Get entity details
            entity = self.graph_manager.get_entity(entity_name)

            # Get relationships
            relationships = self.graph_manager.get_entity_relationships(entity_name)

            # Generate summary
            if self.initialized and entity:
                summary = self.summarize_entity(entity_name, entity, relationships)
            else:
                summary = f"Entity: {entity_name}"

            return {
                "success": True,
                "entity": entity,
                "relationships": relationships,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"Error getting entity info: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def summarize_entity(self, entity_name: str, entity_data: Dict,
                        relationships: List[Dict]) -> str:
        """
        Generate a natural language summary of an entity

        Args:
            entity_name: Name of the entity
            entity_data: Entity properties
            relationships: List of relationships

        Returns:
            Natural language summary
        """
        try:
            source = entity_data.get('source', 'Unknown')

            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are summarizing information about an entity from a knowledge graph.
Provide a clear, informative summary mentioning the entity's properties and key relationships.
Always mention the source document at the end of your summary."""),
                HumanMessage(content=f"""
Entity: {entity_name}
Properties: {entity_data}
Relationships: {relationships}
Source document: {source}

Provide a brief summary of this entity and its connections. Mention the source document.""")
            ])

            response = self.llm.invoke(prompt.format_messages())
            return response.content.strip()

        except Exception as e:
            logger.error(f"Error summarizing entity: {str(e)}")
            return f"{entity_name} - {len(relationships)} relationships"

    def get_suggestions(self, partial_query: str) -> List[str]:
        """
        Get query suggestions based on partial input

        Args:
            partial_query: Partial query from user

        Returns:
            List of query suggestions
        """
        suggestions = [
            "Show me all entities",
            "Find all organizations",
            "What are the relationships for [entity name]?",
            "Find entities related to [entity name]",
            "Show all people in the graph",
            "List all locations",
            "What does [entity name] relate to?",
            "Find connections between [entity1] and [entity2]"
        ]

        if partial_query:
            # Filter suggestions based on partial query
            suggestions = [s for s in suggestions if partial_query.lower() in s.lower()]

        return suggestions[:5]