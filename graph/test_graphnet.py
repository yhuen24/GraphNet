"""
Test script for GraphNet application.
Tests core functionality without the UI.
"""

import os
from main import GraphNet
from config import config

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")

def test_configuration():
    """Test configuration"""
    print_header("Testing Configuration")
    
    validation = config.validate()
    
    if validation['valid']:
        print_success("Configuration is valid")
    else:
        print_error("Configuration has issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    print_info(f"Neo4j URI: {config.NEO4J_URI}")
    print_info(f"OpenAI Model: {config.OPENAI_MODEL}")
    print_info(f"Supported formats: {', '.join(config.SUPPORTED_FORMATS)}")
    
    return validation['valid']

def test_initialization():
    """Test GraphNet initialization"""
    print_header("Testing Initialization")
    
    graphnet = GraphNet()
    
    print_info("Initializing components...")
    init_status = graphnet.initialize()
    
    print(f"\nComponents Status:")
    print(f"  Graph Manager: {'✅' if init_status['graph_manager'] else '❌'}")
    print(f"  Entity Extractor: {'✅' if init_status['entity_extractor'] else '❌'}")
    print(f"  Query Agent: {'✅' if init_status['query_agent'] else '❌'}")
    print(f"  Overall: {'✅' if init_status['overall'] else '❌'}")
    
    if init_status['errors']:
        print_error("Errors encountered:")
        for error in init_status['errors']:
            print(f"  - {error}")
    
    return graphnet if init_status['overall'] else None

def test_document_processing(graphnet):
    """Test document processing"""
    print_header("Testing Document Processing")
    
    # Create a test document
    test_text = """
    John Smith works for Acme Corporation in New York City. 
    He is the Chief Technology Officer and manages the AI Research team.
    The company recently launched a new product called SmartAI.
    Mary Johnson, the CEO, announced the launch on January 15, 2024.
    The product uses machine learning to analyze customer data.
    """
    
    # Save to test file
    test_file = "test_document.txt"
    with open(test_file, 'w') as f:
        f.write(test_text)
    
    print_info(f"Created test document: {test_file}")
    
    # Process the document
    print_info("Processing document...")
    result = graphnet.process_document(
        file_path=test_file,
        file_extension=".txt",
        filename="test_document.txt"
    )
    
    if result.get('success'):
        print_success("Document processed successfully")
        print(f"  Entities extracted: {result['entities_extracted']}")
        print(f"  Relationships extracted: {result['relationships_extracted']}")
        print(f"  Entities added to graph: {result['entities_added']}")
        print(f"  Relationships added to graph: {result['relationships_added']}")
    else:
        print_error(f"Document processing failed: {result.get('error')}")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    return result.get('success', False)

def test_querying(graphnet):
    """Test natural language querying"""
    print_header("Testing Natural Language Queries")
    
    test_queries = [
        "Show me all entities",
        "Find all organizations",
        "List all people"
    ]
    
    for query in test_queries:
        print_info(f"Query: {query}")
        result = graphnet.query(query)
        
        if result.get('success'):
            print_success(f"Query successful - found {result.get('result_count', 0)} results")
            if result.get('explanation'):
                print(f"  Explanation: {result['explanation'][:100]}...")
        else:
            print_error(f"Query failed: {result.get('error')}")
        print()

def test_graph_stats(graphnet):
    """Test graph statistics"""
    print_header("Testing Graph Statistics")
    
    stats = graphnet.get_graph_statistics()
    
    if stats:
        db_stats = stats.get('database', {})
        print_info("Database Statistics:")
        print(f"  Total Nodes: {db_stats.get('node_count', 0)}")
        print(f"  Total Relationships: {db_stats.get('relationship_count', 0)}")
        
        node_types = db_stats.get('node_types', [])
        if node_types:
            print_info("Node Types:")
            for node_type in node_types[:5]:
                labels = node_type.get('labels', ['Unknown'])
                count = node_type.get('count', 0)
                print(f"    {labels[0]}: {count}")
        
        print_success("Statistics retrieved successfully")
    else:
        print_error("Failed to retrieve statistics")

def test_search(graphnet):
    """Test entity search"""
    print_header("Testing Entity Search")
    
    search_terms = ["John", "Acme", "New York"]
    
    for term in search_terms:
        print_info(f"Searching for: {term}")
        results = graphnet.search_entities(term)
        
        if results:
            print_success(f"Found {len(results)} matching entities")
            for result in results[:3]:
                print(f"    {result.get('name')} ({', '.join(result.get('types', ['Unknown']))})")
        else:
            print_info("No results found")
        print()

def test_visualization(graphnet):
    """Test graph visualization"""
    print_header("Testing Visualization")
    
    try:
        print_info("Generating visualization...")
        filename = graphnet.visualize_graph(limit=50)
        
        if os.path.exists(filename):
            print_success(f"Visualization created: {filename}")
            print_info("Open this file in a web browser to view the graph")
        else:
            print_error("Visualization file not created")
    except Exception as e:
        print_error(f"Visualization failed: {str(e)}")

def test_export(graphnet):
    """Test graph export"""
    print_header("Testing Graph Export")
    
    try:
        print_info("Exporting graph to JSON...")
        filename = graphnet.export_graph("test_export.json")
        
        if os.path.exists(filename):
            print_success(f"Graph exported: {filename}")
            
            # Check file size
            size = os.path.getsize(filename)
            print_info(f"File size: {size} bytes")
            
            # Clean up
            os.remove(filename)
            print_info("Test export file removed")
        else:
            print_error("Export file not created")
    except Exception as e:
        print_error(f"Export failed: {str(e)}")

def run_all_tests():
    """Run all tests"""
    print_header("GraphNet Test Suite")
    print_info("Starting comprehensive tests...")
    
    # Test configuration
    if not test_configuration():
        print_error("Configuration test failed. Please fix configuration before proceeding.")
        return
    
    # Initialize GraphNet
    graphnet = test_initialization()
    
    if not graphnet:
        print_error("Initialization failed. Cannot proceed with tests.")
        return
    
    # Run tests
    try:
        # Document processing
        doc_success = test_document_processing(graphnet)
        
        # Only run query tests if documents were processed
        if doc_success:
            test_querying(graphnet)
            test_search(graphnet)
        
        # Graph statistics
        test_graph_stats(graphnet)
        
        # Visualization
        test_visualization(graphnet)
        
        # Export
        test_export(graphnet)
        
        print_header("Test Suite Complete")
        print_success("All tests completed!")
        print_info("Check the results above for any issues")
        
    except Exception as e:
        print_error(f"Test suite error: {str(e)}")
    
    finally:
        # Cleanup
        print_info("Cleaning up...")
        graphnet.shutdown()
        print_info("Tests finished")

if __name__ == "__main__":
    run_all_tests()
