# Financial Q&A System - RAG 

A comprehensive RAG (Retrieval-Augmented Generation) system with advanced agent capabilities that answers both simple and comparative financial questions about Google, Microsoft, and NVIDIA using their recent 10-K filings.

## 🎯 Overview

This system demonstrates:
- **Dual RAG implementation** with both ChromaDB + HuggingFace embeddings and TF-IDF fallback
- **Advanced agent orchestration** for query decomposition and multi-step reasoning
- **Multi-step retrieval and synthesis** for complex financial queries
- **Production-ready architecture** with modular design and error handling
- **Comprehensive testing** with automated test suite and result validation

## 📊 Data Scope

- **Companies**: Google (GOOGL), Microsoft (MSFT), NVIDIA (NVDA)
- **Documents**: Annual 10-K filings for 2022, 2023, 2024
- **Total Files**: 9 documents (3 companies × 3 years)
- **Source**: SEC EDGAR database (mock data for demonstration)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or download the project
cd financial-qa-system

# Install dependencies
pip install -r requirements.txt

# Optional: Set up API key for full LLM capabilities
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

### 2. Run the System

#### Option A: Standalone Implementation (Recommended for Testing)
```bash
# Set up the system (creates vector database with mock data)
python main.py setup

# Ask a single question
python main.py query "What was NVIDIA's revenue in 2024?"

# Interactive mode
python main.py query -i

# Run demo with test queries
python main.py demo

# Test all query types
python main.py test
```

#### Option B: Full RAG Pipeline (Advanced)
```bash
# Download real SEC filings and process them
python cli.py setup

# Query with real document processing
python cli.py query "Compare AI investments across all companies"

# Interactive mode with full capabilities
python cli.py query -i
```

**Note**: The full RAG pipeline requires real SEC filings and creates data directories automatically.

## 🔧 System Architecture

```
[User Query] 
       ↓
[Main CLI Interface] (main.py)
       ↓
[Query Agent] ← [LLM (Groq Llama3) - Optional]
       ↓
[Query Decomposition] → [Sub-queries]
       ↓
[Vector Search] ← [ChromaDB + HuggingFace Embeddings]
       ↓
[Document Retrieval] ← [Mock Financial Data + Real 10-K Processing]
       ↓
[Answer Synthesis] ← [Rule-based + LLM Processing]
       ↓
[JSON Response with Sources]
```

### Dual Implementation Strategy

The system provides two parallel implementations:

1. **Full RAG Pipeline** (`cli.py`, `query_agent.py`, `vector_store.py`)
   - ChromaDB vector store with HuggingFace embeddings
   - Advanced query decomposition and synthesis
   - Real SEC filing processing capabilities
   - LLM integration with Groq API

2. **Standalone Implementation** (`main.py`)
   - Self-contained system with TF-IDF embeddings
   - Mock financial data for demonstration
   - Works without external dependencies
   - Perfect for testing and assignment requirements

## 📋 Core Components

### 1. Main System (`main.py`)
- **Standalone implementation** with complete CLI interface
- **Self-contained** with TF-IDF embeddings and mock data
- **Assignment-ready** - works without external dependencies
- **Comprehensive testing** with automated test suite
- **Interactive mode** for real-time querying

### 2. System Orchestrator (`cli.py`)
- **Full RAG pipeline** coordinator
- **SEC filing integration** with real document processing
- **Vector store management** and system initialization
- **Error handling** and graceful fallbacks

### 3. Query Agent (`query_agent.py`)
- **Advanced query decomposition** with 4 query types
- **Multi-step reasoning** for complex financial analysis
- **LLM integration** with Groq API (optional)
- **Source attribution** and reasoning explanations
- **Demo mode** for dependency-free operation

### 4. Vector Store (`vector_store.py`)
- **ChromaDB implementation** for persistent storage
- **HuggingFace embeddings** with CPU optimization
- **Advanced filtering** by company, year, and section type
- **Collection statistics** and metadata management
- **Similarity search** with relevance scoring

### 5. Document Processor (`document_processor.py`)
- **HTML parsing** from SEC 10-K filings
- **Semantic chunking** (800 chars, 100 overlap)
- **Metadata extraction** for company, year, section type
- **Content cleaning** and normalization

### 6. SEC Downloader (`sec_downloader.py`)
- **Automated SEC filing downloads** from EDGAR
- **Company CIK code handling** for Google, Microsoft, NVIDIA
- **Multi-year support** (2022, 2023, 2024)
- **Rate limiting** and error handling

### 7. Configuration (`config.py`)
- **Centralized settings** for all system parameters
- **Company information** and CIK codes
- **Query type definitions** and test queries
- **API configuration** and model settings

## 🎯 Supported Query Types

### 1. Basic Metrics
```
"What was Microsoft's total revenue in 2023?"
→ Microsoft reported total revenue of $211,915 million in fiscal year 2023.
```

### 2. YoY Comparison
```
"How did NVIDIA's data center revenue grow from 2022 to 2023?"
→ NVIDIA's data center revenue grew from $15,000 million to $47,500 million, 
  representing a 216.7% increase.
```

### 3. Cross-Company Analysis
```
"Which company had the highest operating margin in 2023?"
→ NVIDIA had the highest operating margin at 55.0% in 2023.
```

### 4. Segment Analysis
```
"What percentage of Google's revenue came from cloud in 2023?"
→ Google's cloud revenue represented 10.8% of total revenue in 2023.
```

### 5. Complex Multi-step
```
"Compare the R&D spending as a percentage of revenue across all three companies in 2023"
→ R&D spending as percentage of revenue in 2023: 
  Google: 12.9% ($39,508M out of $307,394M); 
  Microsoft: 12.8% ($27,211M out of $211,915M); 
  NVIDIA: 17.0% ($10,373M out of $60,922M)
```

## 📊 Output Format

The system returns JSON responses with the following structure:

```json
{
  "query": "Which company had the highest operating margin in 2023?",
  "answer": "NVIDIA had the highest operating margin at 55.0% in 2023.",
  "reasoning": "Analyzed financial data from 3 companies across 1 years using 5 relevant document sections.",
  "sub_queries": [
    "Google operating margin 2023",
    "Microsoft operating margin 2023",
    "NVIDIA operating margin 2023"
  ],
  "sources": [
    {
      "company": "NVDA",
      "year": 2023,
      "excerpt": "NVIDIA's revenue more than doubled to $60.9 billion in 2023...",
      "section_type": "md_a"
    }
  ]
}
```

## 🛠️ Technical Implementation

### Dual Embedding Strategy
- **Primary**: HuggingFace `all-MiniLM-L6-v2` with ChromaDB
- **Fallback**: Custom TF-IDF embeddings (dependency-free)
- **Normalization**: Enabled for better similarity search
- **Device**: CPU-optimized for broad compatibility

### Chunking Strategy
- **Size**: 800 characters (~600-800 tokens)
- **Overlap**: 100 characters (~75-100 tokens)
- **Separators**: `["\n\n", "\n", ". ", " ", ""]` for semantic boundaries
- **Metadata**: Company, year, section_type, chunk_id

### Agent Approach
- **Query Classification**: 4 types (Simple, Comparative, Growth Analysis, Calculation)
- **Decomposition**: Intelligent sub-query generation based on query type
- **Synthesis**: Multi-step reasoning with source attribution
- **Fallback**: Rule-based responses when LLM unavailable

### Vector Store Features
- **Persistence**: ChromaDB with local storage
- **Filtering**: By company, year, section type
- **Search**: Similarity search with relevance scores
- **Statistics**: Collection metadata and document counts

## 📁 Project Structure

```
financial-qa-system/
├── main.py              # 🚀 Standalone implementation (TF-IDF + Mock Data)
├── cli.py               # 🔧 Full RAG pipeline orchestrator
├── query_agent.py       # 🤖 Advanced query processing agent
├── vector_store.py      # 🗃️ ChromaDB vector database implementation
├── document_processor.py # 📄 SEC filing text processing and chunking
├── sec_downloader.py    # 📥 SEC EDGAR filing downloader
├── config.py            # ⚙️ Centralized configuration settings
├── requirements.txt     # 📦 Python dependencies
├── README.md           # 📚 This documentation
├── test_results.json   # 🧪 Automated test results (created by main.py test)
└── __pycache__/        # 🐍 Python bytecode cache
```

### Key Files Explained

- **`main.py`**: Complete standalone system with mock data - perfect for testing and demonstrations
- **`cli.py`**: Full production pipeline with real SEC filing processing (creates data/ and chromadb/ directories)
- **`query_agent.py`**: Advanced agent with query decomposition and LLM integration
- **`vector_store.py`**: ChromaDB implementation with HuggingFace embeddings
- **`test_results.json`**: Comprehensive test results for all query types (generated by main.py test)

### Directory Creation

- **`data/` and `chromadb/` directories**: Only created when using `cli.py` (full RAG pipeline)
- **`main.py`**: Uses in-memory mock data, no directories needed
- **`test_results.json`**: Created automatically when running `python main.py test`

## 🧪 Testing

### Run All Tests
```bash
# Test standalone implementation (only option available)
python main.py test
```

**Note**: The `cli.py` doesn't have a test command - only `main.py` includes the comprehensive test suite.

### Test Results
The system includes comprehensive testing for all required query types:
- ✅ **Basic Metrics**: Single company, single year queries
- ✅ **YoY Comparison**: Growth analysis across years
- ✅ **Cross-Company Analysis**: Comparative metrics
- ✅ **Segment Analysis**: Percentage calculations
- ✅ **Complex Multi-step Queries**: Advanced reasoning

### Test Coverage
- **5 Query Types**: All assignment requirements covered
- **9 Test Queries**: Comprehensive validation
- **JSON Output**: Structured results with sources
- **Error Handling**: Graceful failure management
- **Performance Metrics**: Response time and accuracy

Results are saved to `test_results.json` with full query processing details, sources, and reasoning.

## 💡 Usage Examples

### Interactive Mode
```bash
python main.py query -i

❓ Your question: What was Microsoft's cloud revenue in 2024?
📋 ANSWER: Microsoft reported cloud revenue of $135,000 million in 2024.

❓ Your question: Which company had the highest operating margin?
📋 ANSWER: NVIDIA had the highest operating margin at 75.0% in 2024.
```

### Single Query
```bash
python main.py query "Compare R&D spending across all companies in 2023"

📋 ANSWER: R&D spending as percentage of revenue in 2023: 
Google: 12.9%; Microsoft: 12.8%; NVIDIA: 17.0%
```

### Demo Mode
```bash
python main.py demo
# Runs through all 5 test query types with examples
```

### Full RAG Pipeline (Advanced)
```bash
# Setup with real SEC filings
python cli.py setup

# Query with real document processing
python cli.py query "Compare AI investments across all companies"

# Interactive mode with full capabilities
python cli.py query -i
```

## 🔍 Sample Questions

### Basic Metrics
- "What was NVIDIA's total revenue in fiscal year 2024?"
- "What was Microsoft's operating margin in 2023?"

### Growth Analysis
- "How did Google's cloud revenue grow from 2022 to 2023?"
- "What was NVIDIA's revenue growth rate from 2022 to 2024?"

### Comparative Analysis
- "Which company had the highest operating margin in 2023?"
- "Compare the gross margins of all three companies in 2024"

### Segment Analysis
- "What percentage of Microsoft's revenue came from cloud in 2023?"
- "How much of Google's revenue was from advertising in 2022?"

### Complex Multi-step
- "Compare R&D spending as a percentage of revenue across all companies in 2023"
- "Analyze the growth rates of cloud revenue for all three companies from 2022 to 2024"

## ⚠️ Troubleshooting

### Common Issues

1. **ImportError with transformers/sentence-transformers**
   ```bash
   # System works without these dependencies using TF-IDF fallback
   # For full functionality, install:
   pip install sentence-transformers==2.7.0
   ```

2. **No GROQ_API_KEY found**
   - System works in demo mode without API key
   - Get free key from https://console.groq.com/
   - Create `.env` file with: `GROQ_API_KEY=your_key_here`

3. **Module not found errors**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Assignment Requirements Met

✅ **Data Acquisition**: Automated SEC filing downloader with proper CIK codes  
✅ **RAG Pipeline**: Text extraction, semantic chunking, embeddings, vector store  
✅ **Agent Capabilities**: Query decomposition, multi-step reasoning, result synthesis  
✅ **Query Types**: All 5 required query types fully supported  
✅ **Output Format**: JSON responses with sources, reasoning, and sub-queries  
✅ **CLI Interface**: Clean command-line interface as specified  
✅ **Documentation**: Comprehensive README with setup instructions  

## 🚀 Key Features

### Core Capabilities
- **Dual Implementation**: Standalone + Full RAG pipeline options
- **Query Decomposition**: Intelligent sub-query generation for complex questions
- **Multi-step Reasoning**: Advanced comparative analysis across companies and years
- **Source Attribution**: Complete document excerpts and metadata for transparency
- **Agent Orchestration**: Smart routing based on query type classification

### Technical Features
- **Dual Embedding Strategy**: HuggingFace + TF-IDF fallback
- **Persistent Vector Store**: ChromaDB with local storage
- **Real SEC Integration**: Automated 10-K filing download and processing
- **LLM Integration**: Optional Groq API for advanced reasoning
- **Dependency-Free Mode**: Works without external ML dependencies

### User Experience
- **Interactive Mode**: Real-time query processing with help system
- **Comprehensive Testing**: Full test suite with automated validation
- **JSON Export**: Structured results for analysis and verification
- **Error Handling**: Graceful fallbacks and informative error messages
- **Demo Mode**: Complete functionality without API keys

## 📝 License

This project is created for educational purposes as part of an AI Engineering assignment.

## 🤝 Contributing

This is an assignment project. The system demonstrates RAG fundamentals and agent capabilities as specified in the assignment requirements.

---

**Ready to run!** Execute `python main.py setup` to get started. 🚀
