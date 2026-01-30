"""
Streamlit UI for GraphNet application.
User interface for uploading documents, querying the graph, and visualizing results.
"""

import streamlit as st
import os
from pathlib import Path
from typing import List
import streamlit.components.v1 as components

from main import GraphNet
from config import config

# Page configuration
st.set_page_config(
    page_title="GraphNet - AI Knowledge Graph",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #4ECDC4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #95A5A6;
        margin-bottom: 2rem;
    }
    .stat-box {
        background-color: #2C3E50;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #95A5A6;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4ECDC4;
    }
    .success-box {
        background-color: #27AE60;
        padding: 1rem;
        border-radius: 5px;
        color: white;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #E74C3C;
        padding: 1rem;
        border-radius: 5px;
        color: white;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #3498DB;
        padding: 1rem;
        border-radius: 5px;
        color: white;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'graphnet' not in st.session_state:
    st.session_state.graphnet = None
    st.session_state.initialized = False
    st.session_state.processed_files = []
    st.session_state.query_history = []


def initialize_graphnet():
    """Initialize GraphNet application"""
    if st.session_state.graphnet is None:
        with st.spinner("Initializing GraphNet..."):
            st.session_state.graphnet = GraphNet()
            init_status = st.session_state.graphnet.initialize()
            st.session_state.initialized = init_status['overall']
            return init_status
    return {"overall": st.session_state.initialized}


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<p class="main-header">🕸️ GraphNet</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Knowledge Graph System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/graph.png", width=150)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home", "📤 Upload Documents", "🔍 Query Graph", 
             "📊 Visualize", "📈 Statistics", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Initialize button
        if st.button("🔄 Initialize/Reconnect", use_container_width=True):
            with st.spinner("Initializing..."):
                init_status = initialize_graphnet()
                
                if init_status['overall']:
                    st.success("✓ Connected successfully!")
                else:
                    st.error("✗ Initialization failed")
                    if init_status.get('errors'):
                        for error in init_status['errors']:
                            st.error(f"  {error}")
        
        # Connection status
        if st.session_state.initialized:
            st.success("🟢 Connected")
        else:
            st.warning("🔴 Not Connected")
        
        st.markdown("---")
        
        # Quick stats
        if st.session_state.initialized and st.session_state.graphnet:
            st.subheader("Quick Stats")
            try:
                stats = st.session_state.graphnet.get_graph_statistics()
                db_stats = stats.get('database', {})
                
                st.metric("Nodes", db_stats.get('node_count', 0))
                st.metric("Relationships", db_stats.get('relationship_count', 0))
                st.metric("Files Processed", len(st.session_state.processed_files))
            except:
                st.info("Unable to load statistics")
    
    # Main content area
    if page == "🏠 Home":
        show_home_page()
    elif page == "📤 Upload Documents":
        show_upload_page()
    elif page == "🔍 Query Graph":
        show_query_page()
    elif page == "📊 Visualize":
        show_visualization_page()
    elif page == "📈 Statistics":
        show_statistics_page()
    elif page == "⚙️ Settings":
        show_settings_page()


def show_home_page():
    """Display home page"""
    st.header("Welcome to GraphNet")
    
    st.markdown("""
    ### What is GraphNet?
    
    GraphNet is an AI-powered knowledge graph system that transforms unstructured corporate data 
    into structured, explainable, and actionable insights. It combines:
    
    - 🤖 **Large Language Models (LLMs)** for intelligent entity extraction
    - 🔗 **LangChain** for coordinated multi-agent processing
    - 🗄️ **Neo4j** for powerful graph database storage
    - 🎨 **Interactive Visualization** for intuitive knowledge exploration
    
    ### Key Features
    
    1. **📄 Multi-Format Document Processing**
       - Support for PDF, Word, Excel, PowerPoint, text files, and more
       - Automatic text extraction and chunking
    
    2. **🧠 Intelligent Entity Extraction**
       - Identifies people, organizations, locations, concepts, and more
       - Discovers relationships between entities
       - Source-linked and traceable results
    
    3. **💬 Natural Language Querying**
       - Ask questions in plain English
       - AI-powered query understanding
       - Explainable answers with entity connections
    
    4. **📊 Graph Visualization**
       - Interactive network diagrams
       - Color-coded entity types
       - Relationship exploration
    
    ### Getting Started
    
    1. **Initialize**: Click "Initialize/Reconnect" in the sidebar
    2. **Upload**: Go to "Upload Documents" to add your data
    3. **Query**: Use "Query Graph" to ask questions
    4. **Visualize**: Explore the "Visualize" page to see your knowledge graph
    
    ### Requirements
    
    Make sure you have configured:
    - ✅ Gemini API Key (for LLM processing)
    - ✅ Neo4j Database (running and accessible)
    
    Check the Settings page to verify your configuration.
    """)
    
    # Status check
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        if st.session_state.initialized:
            st.markdown('✅ <b>System Status</b><br>Ready', unsafe_allow_html=True)
        else:
            st.markdown('⚠️ <b>System Status</b><br>Not Initialized', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        api_status = "✅ Configured" if config.GOOGLE_API_KEY else "❌ Not Set"
        st.markdown(f'<b>Gemini API</b><br>{api_status}', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        if st.session_state.initialized:
            db_status = "✅ Connected"
        else:
            db_status = "❌ Not Connected"
        st.markdown(f'<b>Neo4j Database</b><br>{db_status}', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def show_upload_page():
    """Display document upload page"""
    st.header("📤 Upload Documents")
    
    if not st.session_state.initialized:
        st.warning("⚠️ Please initialize GraphNet first using the sidebar button.")
        return
    
    st.markdown("""
    Upload your documents to build the knowledge graph. Supported formats:
    **PDF, DOCX, XLSX, PPTX, TXT, MD, CSV, JSON**
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'docx', 'xlsx', 'pptx', 'txt', 'md', 'csv', 'json'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📎 {len(uploaded_files)} file(s) selected")
        
        if st.button("🚀 Process Documents", type="primary"):
            process_documents(uploaded_files)
    
    # Show processed files
    if st.session_state.processed_files:
        st.markdown("---")
        st.subheader("📋 Processed Files")
        
        for file_info in st.session_state.processed_files:
            with st.expander(f"📄 {file_info['filename']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Entities Extracted", file_info.get('entities_extracted', 0))
                    st.metric("Entities Added", file_info.get('entities_added', 0))
                with col2:
                    st.metric("Relationships Extracted", file_info.get('relationships_extracted', 0))
                    st.metric("Relationships Added", file_info.get('relationships_added', 0))


def process_documents(uploaded_files: List):
    """Process uploaded documents"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    results = []
    
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name} ({idx + 1}/{total_files})...")
        
        # Get file extension
        file_extension = Path(uploaded_file.name).suffix
        
        # Read file bytes
        file_bytes = uploaded_file.read()
        
        # Process document
        result = st.session_state.graphnet.process_document(
            file_bytes=file_bytes,
            file_extension=file_extension,
            filename=uploaded_file.name
        )
        
        results.append(result)
        
        if result.get('success'):
            st.session_state.processed_files.append(result)
        
        progress_bar.progress((idx + 1) / total_files)
    
    status_text.empty()
    progress_bar.empty()
    
    # Show results
    success_count = sum(1 for r in results if r.get('success'))
    
    if success_count == total_files:
        st.success(f"✅ Successfully processed all {total_files} documents!")
    elif success_count > 0:
        st.warning(f"⚠️ Processed {success_count} out of {total_files} documents")
    else:
        st.error("❌ Failed to process documents")
    
    # Show detailed results
    for result in results:
        if result.get('success'):
            st.info(f"""
            **{result['filename']}**
            - Extracted: {result['entities_extracted']} entities, {result['relationships_extracted']} relationships
            - Added to graph: {result['entities_added']} entities, {result['relationships_added']} relationships
            """)
        else:
            st.error(f"**{result.get('filename', 'Unknown')}**: {result.get('error', 'Processing failed')}")


def execute_query(query: str):
    with st.spinner("Thinking..."):
        # 1. Get raw data from the Graph
        result = st.session_state.graphnet.query(query)

        if result.get('success'):
            # 2. Convert raw Graph Data into a ChatGPT-style response
            # We call explain_results which uses Gemini to 'read' the graph output
            explanation = result.get('explanation', "No explanation generated.")

            st.markdown(f"### 🤖 Answer\n{explanation}")

            # 3. Handle the Visualization
            results_data = result.get('results', [])
            if results_data:
                # Find the main entity name from the results to center the graph
                focal_node = None
                # Check different possible result structures
                first_res = results_data[0]
                if 'n' in first_res:
                    focal_node = first_res['n'].get('name')
                elif 'name' in first_res:
                    focal_node = first_res.get('name')

                if focal_node:
                    st.session_state.graphnet.visualize_focused(focal_node)



def show_query_page():
    """Display query page"""
    st.header("🔍 Query Knowledge Graph")
    
    if not st.session_state.initialized:
        st.warning("⚠️ Please initialize GraphNet first using the sidebar button.")
        return
    
    st.markdown("Ask questions about your data in natural language!")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., 'Show me all organizations' or 'What are the relationships for John Doe?'"
    )
    
    # Example queries
    with st.expander("💡 Example Queries"):
        examples = [
            "Show me all entities",
            "Find all organizations",
            "List all people in the graph",
            "What locations are mentioned?",
            "Show relationships for [entity name]"
        ]
        for example in examples:
            if st.button(example, key=f"example_{example}"):
                query = example
    
    if query:
        if st.button("🔍 Search", type="primary"):
            execute_query(query)
    
    # Query history
    if st.session_state.query_history:
        st.markdown("---")
        st.subheader("📜 Query History")
        
        for i, hist in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Query: {hist['query'][:50]}..."):
                st.write("**Question:**", hist['query'])
                st.write("**Results:**", hist['result_count'])
                if hist.get('explanation'):
                    st.write("**Explanation:**", hist['explanation'])




def show_visualization_page():
    """Display visualization page"""
    st.header("📊 Graph Visualization")
    
    if not st.session_state.initialized:
        st.warning("⚠️ Please initialize GraphNet first using the sidebar button.")
        return
    
    st.markdown("Explore your knowledge graph visually!")
    
    # Visualization options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        limit = st.slider("Max nodes to display", 10, 500, 100)
        
        if st.button("🎨 Generate Visualization", type="primary"):
            generate_visualization(limit)
    
    # Display visualization
    if os.path.exists("graph_visualization.html"):
        with open("graph_visualization.html", 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        components.html(html_content, height=800, scrolling=True)
    else:
        st.info("👆 Click 'Generate Visualization' to view your knowledge graph")


def generate_visualization(limit: int):
    """Generate graph visualization"""
    with st.spinner("Creating visualization..."):
        try:
            filename = st.session_state.graphnet.visualize_graph(limit=limit)
            st.success(f"✅ Visualization created: {filename}")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Visualization failed: {str(e)}")


def show_statistics_page():
    """Display statistics page"""
    st.header("📈 Graph Statistics")
    
    if not st.session_state.initialized:
        st.warning("⚠️ Please initialize GraphNet first using the sidebar button.")
        return
    
    try:
        stats = st.session_state.graphnet.get_graph_statistics()
        db_stats = stats.get('database', {})
        vis_stats = stats.get('visualization', {})
        
        # Overview metrics
        st.subheader("📊 Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Nodes", db_stats.get('node_count', 0))
        with col2:
            st.metric("Total Relationships", db_stats.get('relationship_count', 0))
        with col3:
            st.metric("Files Processed", len(st.session_state.processed_files))
        with col4:
            avg_degree = vis_stats.get('avg_degree', 0)
            st.metric("Avg Connections", f"{avg_degree:.1f}")
        
        # Node types
        st.subheader("🏷️ Entity Types")
        node_types = db_stats.get('node_types', [])
        
        if node_types:
            for node_type in node_types:
                labels = node_type.get('labels', ['Unknown'])
                count = node_type.get('count', 0)
                st.progress(count / max(1, db_stats.get('node_count', 1)), 
                           text=f"{labels[0]}: {count}")
        else:
            st.info("No entities in the graph yet")
        
        # Relationship types
        st.subheader("🔗 Relationship Types")
        rel_types = db_stats.get('relationship_types', [])
        
        if rel_types:
            for rel_type in rel_types:
                rel_name = rel_type.get('type', 'Unknown')
                count = rel_type.get('count', 0)
                st.progress(count / max(1, db_stats.get('relationship_count', 1)),
                           text=f"{rel_name}: {count}")
        else:
            st.info("No relationships in the graph yet")
        
        # Top entities
        st.subheader("⭐ Most Connected Entities")
        top_entities = vis_stats.get('top_entities', [])
        
        if top_entities:
            for entity in top_entities:
                st.write(f"**{entity.get('name', 'Unknown')}** - Centrality: {entity.get('centrality', 0)}")
        else:
            st.info("No entity rankings available")
        
    except Exception as e:
        st.error(f"❌ Error loading statistics: {str(e)}")


def show_settings_page():
    """Display settings page"""
    st.header("⚙️ Settings")
    
    st.subheader("🔧 Configuration")
    
    config_dict = config.get_config_dict()
    
    st.write("**Neo4j Configuration:**")
    st.code(f"""
URI: {config_dict['neo4j_uri']}
Username: {config_dict['neo4j_username']}
    """)
    
    st.write("**Gemini Configuration:**")
    api_key_status = "✅ Configured" if config.GOOGLE_API_KEY else "❌ Not Set"
    st.code(f"""
Model: {config_dict['ai_model']}
API Key: {api_key_status}
    """)
    
    st.write("**Processing Settings:**")
    st.code(f"""
Max File Size: {config_dict['max_file_size_mb']} MB
Chunk Size: {config_dict['chunk_size']}
Chunk Overlap: {config_dict['chunk_overlap']}
    """)
    
    st.write("**Supported Formats:**")
    st.write(", ".join(config_dict['supported_formats']))
    
    st.markdown("---")
    
    st.subheader("🗑️ Data Management")
    
    st.warning("⚠️ Danger Zone")
    
    if st.button("🗑️ Clear All Graph Data", type="secondary"):
        if st.checkbox("I understand this will delete all data"):
            if st.session_state.initialized:
                with st.spinner("Clearing graph..."):
                    success = st.session_state.graphnet.clear_graph()
                    if success:
                        st.success("✅ Graph data cleared")
                        st.session_state.processed_files = []
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear graph")
            else:
                st.error("❌ Please initialize GraphNet first")
    
    st.markdown("---")
    
    st.subheader("📥 Export Data")
    
    if st.button("💾 Export Graph to JSON"):
        if st.session_state.initialized:
            with st.spinner("Exporting..."):
                try:
                    filename = st.session_state.graphnet.export_graph()
                    st.success(f"✅ Graph exported to {filename}")
                    
                    with open(filename, 'r') as f:
                        st.download_button(
                            "⬇️ Download JSON",
                            f,
                            file_name=filename,
                            mime="application/json"
                        )
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
        else:
            st.error("❌ Please initialize GraphNet first")


if __name__ == "__main__":
    main()
