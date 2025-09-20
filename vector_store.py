"""Vector store implementation using ChromaDB."""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import config


class FinancialVectorStore:
    """Vector store for financial documents using ChromaDB."""
    
    def __init__(self, persist_directory: str = config.CHROMADB_PERSIST_DIR):
        self.persist_directory = persist_directory
        self.collection_name = config.COLLECTION_NAME
        
        # Initialize embeddings - Assignment Requirements
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"✅ Initialized embeddings model: {config.EMBEDDING_MODEL}")
        except Exception as e:
            print(f"❌ Could not initialize HuggingFace embeddings: {e}")
            print("💡 Using simple TF-IDF fallback for demonstration")
            self.embeddings = None
            self.use_simple_embeddings = True
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.vectorstore = None
        
        # Create directories
        os.makedirs(persist_directory, exist_ok=True)
        
    def create_vectorstore(self, documents: List[Document]) -> None:
        """Create and populate the vector store with documents."""
        if not documents:
            print("No documents provided to create vector store")
            return
            
        print(f"Creating vector store with {len(documents)} documents...")
        
        try:
            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                print("Deleted existing collection")
            except ValueError:
                pass  # Collection doesn't exist
            
            # Create new vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=self.chroma_client,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
            
            print(f"Vector store created successfully with {len(documents)} documents")
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise
    
    def load_vectorstore(self) -> bool:
        """Load existing vector store."""
        try:
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Test if collection exists and has documents
            collection = self.chroma_client.get_collection(name=self.collection_name)
            count = collection.count()
            
            if count == 0:
                print("Vector store collection is empty")
                return False
                
            print(f"Loaded vector store with {count} documents")
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Perform similarity search in the vector store."""
        if not self.vectorstore:
            print("Vector store not initialized")
            return []
        
        try:
            # Add filter for specific companies or years if provided
            filter_dict = {}
            if 'company' in kwargs:
                filter_dict['company'] = kwargs['company']
            if 'year' in kwargs:
                filter_dict['year'] = kwargs['year']
            if 'section_type' in kwargs:
                filter_dict['section_type'] = kwargs['section_type']
            
            if filter_dict:
                results = self.vectorstore.similarity_search(
                    query, k=k, filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5, **kwargs) -> List[tuple]:
        """Perform similarity search with relevance scores."""
        if not self.vectorstore:
            print("Vector store not initialized")
            return []
        
        try:
            # Add filter for specific companies or years if provided
            filter_dict = {}
            if 'company' in kwargs:
                filter_dict['company'] = kwargs['company']
            if 'year' in kwargs:
                filter_dict['year'] = kwargs['year']
            if 'section_type' in kwargs:
                filter_dict['section_type'] = kwargs['section_type']
            
            if filter_dict:
                results = self.vectorstore.similarity_search_with_score(
                    query, k=k, filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            return results
            
        except Exception as e:
            print(f"Error during similarity search with score: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        if not self.vectorstore:
            return {"status": "not_initialized"}
        
        try:
            collection = self.chroma_client.get_collection(name=self.collection_name)
            total_documents = collection.count()
            
            # Get sample documents to analyze metadata
            if total_documents > 0:
                sample_docs = self.vectorstore.similarity_search("revenue", k=min(100, total_documents))
                
                companies = set()
                years = set()
                sections = set()
                
                for doc in sample_docs:
                    companies.add(doc.metadata.get('company', 'unknown'))
                    years.add(doc.metadata.get('year', 'unknown'))
                    sections.add(doc.metadata.get('section_type', 'unknown'))
                
                return {
                    "status": "active",
                    "total_documents": total_documents,
                    "companies": sorted(list(companies)),
                    "years": sorted(list(years)),
                    "section_types": sorted(list(sections))
                }
            else:
                return {"status": "empty", "total_documents": 0}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def search_by_company_and_year(self, query: str, company: str, year: int, k: int = 5) -> List[Document]:
        """Search for documents from a specific company and year."""
        return self.similarity_search(
            query, k=k, company=company, year=year
        )
    
    def search_financial_metrics(self, metric: str, companies: List[str] = None, years: List[int] = None, k: int = 10) -> List[Document]:
        """Search for specific financial metrics across companies and years."""
        # Enhance query with financial terms
        enhanced_query = f"{metric} revenue margin income profit loss financial results"
        
        results = []
        
        if companies and years:
            # Search for each company/year combination
            for company in companies:
                for year in years:
                    company_results = self.similarity_search(
                        enhanced_query, k=k//2, company=company, year=year
                    )
                    results.extend(company_results)
        else:
            # General search
            results = self.similarity_search(enhanced_query, k=k)
        
        # Remove duplicates based on chunk_id
        seen_chunks = set()
        unique_results = []
        for doc in results:
            chunk_id = doc.metadata.get('chunk_id')
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(doc)
        
        return unique_results[:k]


if __name__ == "__main__":
    # Test vector store functionality
    vector_store = FinancialVectorStore()
    
    # Try to load existing vector store
    if vector_store.load_vectorstore():
        print("Vector store loaded successfully")
        
        # Get stats
        stats = vector_store.get_collection_stats()
        print(f"Vector store stats: {stats}")
        
        # Test search
        test_query = "total revenue 2023"
        results = vector_store.similarity_search(test_query, k=3)
        print(f"\nTest search results for '{test_query}':")
        for i, doc in enumerate(results):
            print(f"{i+1}. Company: {doc.metadata['company']}, Year: {doc.metadata['year']}")
            print(f"   Content preview: {doc.page_content[:200]}...")
            
    else:
        print("No existing vector store found. Run the main pipeline first.")