# GraphNet - AI-Powered Knowledge Graph

GraphNet is an intelligent knowledge graph system that transforms unstructured corporate data into structured, explainable, and actionable insights using Large Language Models (LLMs), LangChain, and Neo4j.

## 🎯 Project Overview

### Main Goal
Turn unstructured corporate data (reports, emails, chat logs, internal documentation) into structured, explainable, and useful insights through an AI-powered knowledge graph.

### Key Features

- **📄 Multi-Format Document Processing**: Support for PDF, Word, Excel, PowerPoint, text files, CSV, JSON, and more
- **🧠 Intelligent Entity Extraction**: Automatically identifies entities (people, organizations, locations, concepts) and their relationships using LLMs
- **💬 Natural Language Querying**: Ask questions in plain English and get explainable, source-linked answers
- **📊 Interactive Visualization**: Explore your knowledge graph through interactive network diagrams
- **🔗 Relationship Discovery**: Automatically discovers and maps relationships between entities
- **📈 Graph Analytics**: Get insights about your knowledge graph with statistics and centrality measures

## 🏗️ Architecture

### Tech Stack

- **Python 3.8+**: Core programming language
- **Streamlit**: Interactive web UI
- **Neo4j**: Graph database for storing entities and relationships
- **LangChain**: Framework for LLM application development
- **OpenAI API**: Language model for entity extraction and query processing
- **PyVis**: Interactive network visualizations
- **NetworkX**: Graph analytics

### Project Structure

```
graphnet/
├── app.py                  # Streamlit UI application
├── main.py                 # Core GraphNet functionality
├── config.py               # Configuration management
├── document_processor.py   # Document parsing and text extraction
├── entity_extractor.py     # LLM-based entity and relationship extraction
├── graph_manager.py        # Neo4j database operations
├── query_agent.py          # Natural language query processing
├── visualizer.py           # Graph visualization
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## 🚀 Installation

### Prerequisites

1. **Python 3.8 or higher**
2. **Neo4j Database**
   - Download from: https://neo4j.com/download/
   - Or use Docker: `docker run -p 7474:7474 -p 7687:7687 neo4j`
3. **OpenAI API Key**
   - Get one at: https://platform.openai.com/api-keys

### Setup Steps

1. **Clone or download the project**

```bash
cd graphnet
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

3. **Download spaCy language model (optional, for fallback entity extraction)**

```bash
python -m spacy download en_core_web_sm
```

4. **Configure environment variables**

Copy the `.env.example` file to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here

# Application Settings
MAX_FILE_SIZE_MB=10
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

5. **Start Neo4j Database**

Make sure your Neo4j database is running and accessible.

6. **Run the application**

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Initialize the System

- Click the "🔄 Initialize/Reconnect" button in the sidebar
- Wait for all components to initialize
- Check for green status indicators

### 2. Upload Documents

Navigate to the "📤 Upload Documents" page:

1. Click "Browse files" to select one or more documents
2. Supported formats: PDF, DOCX, XLSX, PPTX, TXT, MD, CSV, JSON
3. Click "🚀 Process Documents"
4. Wait for processing to complete

The system will:
- Extract text from your documents
- Identify entities (people, organizations, locations, etc.)
- Discover relationships between entities
- Add everything to the knowledge graph

### 3. Query the Knowledge Graph

Go to the "🔍 Query Graph" page:

1. Type your question in natural language
2. Examples:
   - "Show me all organizations"
   - "Find all people in the graph"
   - "What are the relationships for John Doe?"
   - "List all locations"
3. Click "🔍 Search"
4. View results and explanations

### 4. Visualize the Graph

Visit the "📊 Visualize" page:

1. Adjust the maximum number of nodes to display
2. Click "🎨 Generate Visualization"
3. Interact with the graph:
   - Drag nodes to rearrange
   - Zoom in/out with mouse wheel
   - Click nodes for details
   - Hover over edges to see relationships

### 5. View Statistics

Check the "📈 Statistics" page to see:

- Total nodes and relationships
- Entity type distribution
- Relationship type distribution
- Most connected entities
- Graph metrics

### 6. Manage Settings

Access the "⚙️ Settings" page to:

- View configuration
- Export graph data to JSON
- Clear all graph data (use with caution!)

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL` | OpenAI model to use | gpt-3.5-turbo |
| `NEO4J_URI` | Neo4j connection URI | bolt://localhost:7687 |
| `NEO4J_USERNAME` | Neo4j username | neo4j |
| `NEO4J_PASSWORD` | Neo4j password | Required |
| `MAX_FILE_SIZE_MB` | Maximum file upload size | 10 |
| `CHUNK_SIZE` | Text chunk size for processing | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 200 |
| `MAX_ENTITIES_PER_CHUNK` | Max entities to extract per chunk | 20 |
| `CONFIDENCE_THRESHOLD` | Entity extraction confidence | 0.7 |

## 🏛️ System Components

### 1. Document Processor (`document_processor.py`)

Handles document parsing and text extraction for multiple formats:
- PDF: Uses PyPDF2
- Word: Uses python-docx
- Excel: Uses openpyxl and pandas
- PowerPoint: Uses python-pptx
- Text/CSV/JSON: Native Python parsing

### 2. Entity Extractor (`entity_extractor.py`)

Uses LangChain and OpenAI to:
- Extract entities from text
- Classify entity types (Person, Organization, Location, etc.)
- Identify relationships between entities
- Generate descriptions

### 3. Graph Manager (`graph_manager.py`)

Manages Neo4j database operations:
- Create/update entities
- Create relationships
- Execute Cypher queries
- Retrieve graph data
- Generate statistics

### 4. Query Agent (`query_agent.py`)

Processes natural language queries:
- Converts questions to Cypher queries
- Executes queries against Neo4j
- Generates explanations
- Provides entity details

### 5. Visualizer (`visualizer.py`)

Creates interactive visualizations:
- PyVis network diagrams
- Color-coded entity types
- Interactive exploration
- Graph analytics

### 6. Main Application (`main.py`)

Orchestrates all components:
- Initializes system
- Processes documents
- Manages workflow
- Coordinates agents

### 7. Streamlit UI (`app.py`)

Provides user interface:
- File upload
- Query interface
- Visualization display
- Statistics dashboard
- Settings management

## 📊 Supported Entity Types

- **Person**: Individuals mentioned in documents
- **Organization**: Companies, institutions, groups
- **Location**: Places, cities, countries
- **Concept**: Ideas, theories, methodologies
- **Product**: Products, services, tools
- **Date**: Temporal references
- **Event**: Occurrences, meetings, incidents
- **Technology**: Technical systems, platforms
- **Other**: Miscellaneous entities

## 🔗 Relationship Types

Common relationship types include:
- `WORKS_FOR`: Employment relationships
- `LOCATED_IN`: Location associations
- `RELATED_TO`: General associations
- `OWNS`: Ownership relationships
- `CREATED`: Creation relationships
- `MANAGES`: Management relationships
- `PARTICIPATED_IN`: Participation in events

## 🐛 Troubleshooting

### Connection Issues

**Problem**: Cannot connect to Neo4j

**Solution**:
1. Verify Neo4j is running: check http://localhost:7474
2. Check URI in `.env` file
3. Verify username and password
4. Ensure port 7687 is not blocked

**Problem**: OpenAI API errors

**Solution**:
1. Verify API key is correct
2. Check API key has sufficient credits
3. Verify internet connection
4. Check OpenAI service status

### Processing Issues

**Problem**: Document processing fails

**Solution**:
1. Check file format is supported
2. Verify file is not corrupted
3. Check file size is within limit
4. Look at error messages for specific issues

**Problem**: No entities extracted

**Solution**:
1. Verify document has meaningful text content
2. Check OpenAI API is working
3. Try processing in smaller chunks
4. Review entity extraction settings

### Performance Issues

**Problem**: Slow processing

**Solution**:
1. Reduce chunk size in configuration
2. Process fewer documents at once
3. Limit visualization node count
4. Use more powerful OpenAI model

## 🔒 Security Considerations

- **API Keys**: Never commit `.env` file to version control
- **Data Privacy**: Use only non-sensitive, open-source data
- **Database Access**: Secure Neo4j with strong passwords
- **Network**: Run on trusted networks only

## 🎓 Example Use Cases

### 1. Research Paper Analysis
Upload research papers and discover:
- Key authors and institutions
- Research topics and concepts
- Collaboration networks
- Technology mentions

### 2. Corporate Document Analysis
Process internal documents to find:
- Organizational structures
- Project relationships
- Key stakeholders
- Department connections

### 3. News Article Analysis
Analyze news articles to identify:
- People and organizations in the news
- Event relationships
- Geographic connections
- Temporal patterns

## 📝 Development Notes

### Adding New Entity Types

Edit `entity_extractor.py`:
1. Update the entity type list in prompts
2. Add color mapping in `visualizer.py`
3. Test extraction with sample data

### Custom Relationship Types

Edit `query_agent.py`:
1. Add relationship type to prompt
2. Update Cypher query patterns
3. Test with example queries

### Extending Document Support

Edit `document_processor.py`:
1. Add new file extension to `SUPPORTED_FORMATS`
2. Create processing method for format
3. Update file type routing logic

## 🤝 Contributing

This is an academic project. For improvements:
1. Test thoroughly
2. Document changes
3. Follow existing code style
4. Update README if needed

## 📄 License

This project is for educational purposes. Please respect data privacy and usage rights.

## 🙏 Acknowledgments

- **LangChain**: For LLM application framework
- **Neo4j**: For graph database technology
- **OpenAI**: For powerful language models
- **Streamlit**: For rapid UI development
- **PyVis**: For graph visualization

## 📧 Support

For issues or questions:
1. Check this README
2. Review error messages
3. Verify configuration
4. Check component logs

---

**GraphNet** - Transforming unstructured data into structured knowledge 🕸️
