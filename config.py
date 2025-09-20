"""Configuration settings for the Financial Q&A System."""

import os
from dotenv import load_dotenv

load_dotenv()

# Company information - Assignment Requirements
COMPANIES = {
    "GOOGL": {"name": "Google", "cik": "1652044"},
    "MSFT": {"name": "Microsoft", "cik": "789019"},
    "NVDA": {"name": "NVIDIA", "cik": "1045810"}
}

# Years to fetch - Assignment Requirements
YEARS = [2022, 2023, 2024]

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️  GROQ_API_KEY not found in environment variables.")
    print("Please create a .env file with: GROQ_API_KEY=your_key_here")
    print("Get your free API key from: https://console.groq.com/")
    # Don't raise error, allow demo mode

# Model settings - Assignment Requirements
GROQ_MODEL = "llama3-8b-8192"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking parameters - Assignment Requirements (200-1000 tokens)
CHUNK_SIZE = 800  # ~600-800 tokens
CHUNK_OVERLAP = 100  # ~75-100 tokens

# Vector database settings - Assignment Requirements
CHROMADB_PERSIST_DIR = "./chromadb"
COLLECTION_NAME = "financial_documents"

# SEC API settings
SEC_BASE_URL = "https://www.sec.gov"
USER_AGENT = "Financial Q&A System (educational@example.com)"

# Directories
DATA_DIR = "./data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Query types to support - Assignment Requirements
QUERY_TYPES = {
    "basic_metrics": "What was Microsoft's total revenue in 2023?",
    "yoy_comparison": "How did NVIDIA's data center revenue grow from 2022 to 2023?", 
    "cross_company": "Which company had the highest operating margin in 2023?",
    "segment_analysis": "What percentage of Google's revenue came from cloud in 2023?",
    "ai_strategy": "Compare AI investments mentioned by all three companies in their 2024 10-Ks"
}

# Sample test queries - Assignment Requirements
TEST_QUERIES = [
    "What was NVIDIA's total revenue in fiscal year 2024?",
    "What percentage of Google's 2023 revenue came from advertising?",
    "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
    "Which of the three companies had the highest gross margin in 2023?",
    "Compare the R&D spending as a percentage of revenue across all three companies in 2023"
]