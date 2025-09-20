"""Command Line Interface for the Financial Q&A System - Assignment Requirements."""

import os
import json
from typing import Dict, Any
from sec_downloader import SECDownloader
from document_processor import DocumentProcessor
from vector_store import FinancialVectorStore
from query_agent import QueryAgent
import config


class FinancialQASystem:
    """Main system orchestrator - Assignment Requirements."""
    
    def __init__(self):
        self.downloader = SECDownloader()
        self.processor = DocumentProcessor()
        self.vector_store = FinancialVectorStore()
        self.agent = None
        
    def setup_system(self, skip_download: bool = False) -> bool:
        """Set up the complete system pipeline - Assignment Requirements."""
        try:
            print("🚀 Starting system setup...")
            
            # Step 1: Download filings (optional)
            if not skip_download:
                print("📥 Step 1: Downloading SEC filings...")
                filings = self.downloader.download_all_filings()
                
                if not any(filings.values()):
                    print("❌ No filings downloaded. Cannot proceed.")
                    return False
                else:
                    print(f"✅ Downloaded {sum(len(years) for years in filings.values())} filings")
            else:
                # Check for existing filings
                print("📁 Step 1: Checking for existing filings...")
                filings = self._scan_existing_filings()
                if not filings:
                    print("❌ No existing filings found. Run without skip_download.")
                    return False
                else:
                    print(f"✅ Found {sum(len(years) for years in filings.values())} existing filings")
            
            # Step 2: Process documents
            print("🔄 Step 2: Processing documents...")
            documents = self.processor.process_all_documents(filings)
            
            if not documents:
                print("❌ No documents processed. Check the filings.")
                return False
            
            print(f"✅ Created {len(documents)} document chunks")
            
            # Step 3: Create vector store
            print("🗃️  Step 3: Creating vector database...")
            self.vector_store.create_vectorstore(documents)
            
            # Step 4: Initialize agent
            print("🤖 Step 4: Initializing query agent...")
            self.agent = QueryAgent(self.vector_store)
            
            print("✅ System setup complete!")
            return True
            
        except Exception as e:
            print(f"❌ Error during setup: {e}")
            return False
    
    def _scan_existing_filings(self) -> Dict[str, Dict[int, str]]:
        """Scan for existing downloaded filings."""
        filings = {}
        
        if not os.path.exists(config.RAW_DATA_DIR):
            return filings
        
        for filename in os.listdir(config.RAW_DATA_DIR):
            if filename.endswith('.htm'):
                try:
                    parts = filename.replace('.htm', '').split('_')
                    if len(parts) >= 3 and parts[1].isdigit():
                        company = parts[0]
                        year = int(parts[1])
                        
                        if company not in filings:
                            filings[company] = {}
                        filings[company][year] = os.path.join(config.RAW_DATA_DIR, filename)
                except (ValueError, IndexError):
                    continue
        
        return filings
    
    def load_existing_system(self) -> bool:
        """Load existing vector store and initialize agent - Assignment Requirements."""
        try:
            if self.vector_store.load_vectorstore():
                self.agent = QueryAgent(self.vector_store)
                print("✅ Loaded existing system successfully!")
                return True
            else:
                print("❌ No existing vector store found. Run setup first.")
                return False
        except Exception as e:
            print(f"❌ Error loading system: {e}")
            return False
    
    def query_system(self, query: str) -> Dict[str, Any]:
        """Query the system."""
        if not self.agent:
            raise ValueError("System not initialized. Run setup or load first.")
        
        return self.agent.answer_query(query)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return self.vector_store.get_collection_stats()


# Note: CLI commands moved to main.py for assignment requirements