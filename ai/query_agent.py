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

                # Switch to Google Gemini
            self.llm = ChatGoogleGenerativeAI(
                model=config.AI_MODEL,
                temperature=0,
                google_api_key=config.GOOGLE_API_KEY
            )

            self.initialized = True
            logger.info("Query agent initialized with Gemini")
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

Common patterns:
- Find entity: MATCH (e:Type {name: "EntityName"}) RETURN e
- Find relationships: MATCH (e1)-[r]->(e2) WHERE e1.name = "Name" RETURN e1, r, e2
- Search: MATCH (e) WHERE e.name CONTAINS "SearchTerm" RETURN e
- Count: MATCH (n:Type) RETURN count(n)
- Get related: MATCH (e {name: "Name"})-[r]-(other) RETURN other

Entity types in the graph: Person, Organization, Location, Concept, Product, Date, Event, Technology
Relationship types: WORKS_FOR, LOCATED_IN, RELATED_TO, OWNS, CREATED, MANAGES, PARTICIPATED_IN

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

            # Generate explanation
            explanation = self.explain_results(
                natural_language_query,
                cypher_query,
                results
            )

            return {
                "success": True,
                "query": cypher_query,
                "results": results,
                "explanation": explanation,
                "result_count": len(results)
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    def explain_results(self, natural_query: str, cypher_query: str,
                       results: List[Dict]) -> str:
        """
        Generate a natural language explanation of query results

        Args:
            natural_query: Original natural language query
            cypher_query: Generated Cypher query
            results: Query results

        Returns:
            Natural language explanation
        """
        if not self.initialized:
            return "Query agent not initialized"

        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are explaining query results from a knowledge graph. 
Provide a clear, concise explanation of what was found. Be specific and mention entity names."""),
                HumanMessage(content=f"""
Original question: {natural_query}

Query executed: {cypher_query}

Results found: {len(results)}

Sample results: {str(results[:3]) if results else "No results"}

Provide a brief, natural explanation of these results.""")
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
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are summarizing information about an entity from a knowledge graph.
Provide a clear, informative summary mentioning the entity's properties and key relationships."""),
                HumanMessage(content=f"""
Entity: {entity_name}
Properties: {entity_data}
Relationships: {relationships}

Provide a brief summary of this entity and its connections.""")
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