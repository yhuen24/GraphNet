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

            # Initialize Google Gemini
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
        if not self.initialized:
            return None

        try:
            # Pass 1: Categorize Intent and Extract Parameters
            intent_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a query intent analyzer. 
                Your goal is to extract the core 'Entity', 'Relationship Type', and 'Result Limit' 
                from a user prompt, regardless of how it is phrased.

                RULES:
                1. Identify the subject.
                2. Identify if the user wants a specific count (e.g., "5", "all", "top").
                3. If the user uses vague words like "information", "facts", or "details", 
                   translate this to a 'Neighborhood Search' (all connections).

                Generate a clean Cypher query based on these rules. 
                Use 'toLower()' and 'CONTAINS' for all entity names.
                Always use '-[r]-' (bi-directional) for relationships.
                If a number is mentioned, always end with 'LIMIT [number]'.

                EXAMPLES:
                - "5 info on company A" -> MATCH (n)-[r]-(m) WHERE toLower(n.name) CONTAINS "company A" RETURN n,r,m LIMIT 5
                - "Who works for them?" -> MATCH (n)-[r:WORKS_FOR]-(m) WHERE toLower(n.name) CONTAINS "company A" RETURN n,r,m
                - "Show me everything" -> MATCH (n)-[r]-(m) RETURN n,r,m LIMIT 100
                """),
                HumanMessage(content=f"User Prompt: {natural_language_query}")
            ])

            response = self.llm.invoke(intent_prompt.format_messages())
            return response.content.strip().replace("```cypher", "").replace("```", "")
        except Exception as e:
            logger.error(f"Error: {e}")
            return None

    def process_query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Process a natural language query and return results
        """
        if not self.initialized:
            return {"success": False, "error": "Query agent not initialized", "results": []}

        try:
            cypher_query = self.generate_cypher_query(natural_language_query)

            if not cypher_query:
                return {"success": False, "error": "Could not generate Cypher query", "results": []}

            # Execute query against the graph manager
            results = self.graph_manager.query_graph(cypher_query)

            # Generate explanation
            explanation = self.explain_results(natural_language_query, cypher_query, results)

            return {
                "success": True,
                "query": cypher_query,
                "results": results,
                "explanation": explanation,
                "result_count": len(results)
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {"success": False, "error": str(e), "results": []}

    def explain_results(self, natural_query: str, cypher_query: str, results: List[Dict]) -> str:
        """
        Generate a natural language explanation of query results
        """
        if not self.initialized:
            return "Query agent not initialized"

        try:
            # Provide more context to the explainer LLM
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are explaining query results from a knowledge graph. 
                If results are empty, politely explain that the specific entity or connection wasn't found in the current graph. 
                If results exist, summarize them clearly."""),
                                HumanMessage(content=f"""
                Original question: {natural_query}
                Query executed: {cypher_query}
                Results found count: {len(results)}
                Data found: {str(results[:10])}
                
                Provide a natural, helpful response.""")
                            ])

            response = self.llm.invoke(prompt.format_messages())
            return response.content.strip()

        except Exception as e:
            logger.error(f"Error explaining results: {str(e)}")
            return f"Found {len(results)} results in the knowledge graph."

    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """Get detailed information about an entity"""
        try:
            entity = self.graph_manager.get_entity(entity_name)
            relationships = self.graph_manager.get_entity_relationships(entity_name)

            summary = self.summarize_entity(entity_name, entity, relationships) if (self.initialized and entity) else f"Entity: {entity_name}"

            return {
                "success": True,
                "entity": entity,
                "relationships": relationships,
                "summary": summary
            }
        except Exception as e:
            logger.error(f"Error getting entity info: {str(e)}")
            return {"success": False, "error": str(e)}

    def summarize_entity(self, entity_name: str, entity_data: Dict, relationships: List[Dict]) -> str:
        """Generate a natural language summary of an entity"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Summarize this entity and its connections clearly."),
                HumanMessage(content=f"Entity: {entity_name}\nProperties: {entity_data}\nRelationships: {relationships}")
            ])
            response = self.llm.invoke(prompt.format_messages())
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error summarizing entity: {str(e)}")
            return f"{entity_name} has {len(relationships)} known connections."

    def get_suggestions(self, partial_query: str) -> List[str]:
        """Get query suggestions"""
        suggestions = [
            "Show me all entities",
            "Find all organizations",
            "What are the relationships for [entity]?",
            "Show all people in the graph",
            "List all locations"
        ]
        if partial_query:
            suggestions = [s for s in suggestions if partial_query.lower() in s.lower()]
        return suggestions[:5]