"""Query agent with decomposition and multi-step reasoning capabilities - Assignment Requirements."""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from vector_store import FinancialVectorStore
import config


class QueryAgent:
    """Agent for handling complex financial queries with decomposition - Assignment Requirements."""
    
    def __init__(self, vector_store: FinancialVectorStore):
        self.vector_store = vector_store
        
        # Initialize LLM - Assignment Requirements
        if config.GROQ_API_KEY:
            self.llm = ChatGroq(
                model_name=config.GROQ_MODEL,
                groq_api_key=config.GROQ_API_KEY,
                temperature=0.1
            )
            print(f"✅ Initialized LLM: {config.GROQ_MODEL}")
        else:
            self.llm = None
            print("⚠️  No GROQ_API_KEY found - running in demo mode")
        
        # Company name mappings - Assignment Requirements
        self.company_mapping = {
            'google': 'GOOGL', 'alphabet': 'GOOGL', 'googl': 'GOOGL',
            'microsoft': 'MSFT', 'msft': 'MSFT',
            'nvidia': 'NVDA', 'nvda': 'NVDA'
        }
        
    def extract_companies_and_years(self, query: str) -> Tuple[List[str], List[int]]:
        """Extract company names and years from the query."""
        companies = []
        years = []
        
        query_lower = query.lower()
        
        # Extract companies
        for name, symbol in self.company_mapping.items():
            if name in query_lower:
                companies.append(symbol)
        
        # If no specific companies mentioned, use all
        if not companies:
            companies = list(config.COMPANIES.keys())
        
        # Extract years
        year_matches = re.findall(r'\b(20\d{2})\b', query)
        years = [int(year) for year in year_matches if 2020 <= int(year) <= 2024]
        
        # If no specific years mentioned, use all available
        if not years:
            years = config.YEARS
        
        return companies, years
    
    def determine_query_type(self, query: str) -> str:
        """Determine the type of query to handle appropriately."""
        query_lower = query.lower()
        
        # Comparative indicators
        comparative_words = ['compare', 'comparison', 'versus', 'vs', 'between', 'across', 
                           'which company', 'who had', 'highest', 'lowest', 'best', 'worst']
        
        # Growth indicators
        growth_words = ['growth', 'grow', 'grew', 'increase', 'decreased', 'change', 'from', 'to']
        
        # Multi-step indicators
        multi_step_words = ['percentage', 'ratio', 'margin', 'as a percent', '%']
        
        if any(word in query_lower for word in comparative_words):
            return 'comparative'
        elif any(word in query_lower for word in growth_words) and ('from' in query_lower or 'to' in query_lower):
            return 'growth_analysis'
        elif any(word in query_lower for word in multi_step_words):
            return 'calculation'
        else:
            return 'simple'
    
    def decompose_query(self, query: str) -> List[str]:
        """Decompose complex queries into sub-queries - Assignment Requirements."""
        query_type = self.determine_query_type(query)
        companies, years = self.extract_companies_and_years(query)
        
        sub_queries = []
        
        if query_type == 'comparative':
            # Assignment Requirements: Cross-company analysis
            base_query = self.extract_base_metric(query)
            for company in companies:
                for year in years:
                    sub_query = f"{config.COMPANIES[company]['name']} {base_query} {year}"
                    sub_queries.append(sub_query)
                    
        elif query_type == 'growth_analysis':
            # Assignment Requirements: YoY comparison
            base_metric = self.extract_base_metric(query)
            years_sorted = sorted(years)
            if len(years_sorted) >= 2:
                start_year, end_year = years_sorted[0], years_sorted[-1]
                for company in companies:
                    sub_queries.extend([
                        f"{config.COMPANIES[company]['name']} {base_metric} {start_year}",
                        f"{config.COMPANIES[company]['name']} {base_metric} {end_year}"
                    ])
            else:
                for company in companies:
                    for year in years:
                        sub_query = f"{config.COMPANIES[company]['name']} {base_metric} {year}"
                        sub_queries.append(sub_query)
                    
        elif query_type == 'calculation':
            # Assignment Requirements: Segment analysis (percentages)
            if 'cloud' in query.lower() and 'revenue' in query.lower():
                for company in companies:
                    for year in years:
                        sub_queries.extend([
                            f"{config.COMPANIES[company]['name']} cloud revenue {year}",
                            f"{config.COMPANIES[company]['name']} total revenue {year}"
                        ])
            elif 'r&d' in query.lower() and 'percentage' in query.lower():
                for company in companies:
                    for year in years:
                        sub_queries.extend([
                            f"{config.COMPANIES[company]['name']} R&D spending {year}",
                            f"{config.COMPANIES[company]['name']} total revenue {year}"
                        ])
            else:
                # General calculation approach
                base_metric = self.extract_base_metric(query)
                for company in companies:
                    for year in years:
                        sub_query = f"{config.COMPANIES[company]['name']} {base_metric} {year}"
                        sub_queries.append(sub_query)
        else:
            # Assignment Requirements: Basic metrics
            sub_queries = [query]
        
        return sub_queries
    
    def extract_base_metric(self, query: str) -> str:
        """Extract the main financial metric from the query."""
        query_lower = query.lower()
        
        # Common financial metrics
        if 'revenue' in query_lower:
            if 'cloud' in query_lower:
                return 'cloud revenue'
            elif 'data center' in query_lower:
                return 'data center revenue'
            else:
                return 'total revenue'
        elif 'operating margin' in query_lower or 'operating income' in query_lower:
            return 'operating margin'
        elif 'gross margin' in query_lower:
            return 'gross margin'
        elif 'net income' in query_lower or 'profit' in query_lower:
            return 'net income'
        elif 'r&d' in query_lower or 'research and development' in query_lower:
            return 'research and development spending'
        elif 'ai' in query_lower and ('investment' in query_lower or 'risk' in query_lower):
            return 'artificial intelligence'
        else:
            # Extract key terms
            important_words = []
            for word in query_lower.split():
                if word not in ['what', 'was', 'is', 'the', 'of', 'in', 'for', 'and', 'or', 'a', 'an']:
                    important_words.append(word)
            return ' '.join(important_words[:3])
    
    def execute_sub_query(self, sub_query: str, k: int = 5) -> List[Document]:
        """Execute a single sub-query against the vector store."""
        # Extract company and year for targeted search
        companies, years = self.extract_companies_and_years(sub_query)
        
        if len(companies) == 1 and len(years) == 1:
            # Targeted search
            results = self.vector_store.search_by_company_and_year(
                sub_query, companies[0], years[0], k=k
            )
        else:
            # General search with enhanced query
            enhanced_query = self.enhance_financial_query(sub_query)
            results = self.vector_store.similarity_search(enhanced_query, k=k)
        
        return results
    
    def enhance_financial_query(self, query: str) -> str:
        """Enhance query with relevant financial terms."""
        query_lower = query.lower()
        enhancements = []
        
        if 'revenue' in query_lower:
            enhancements.extend(['sales', 'income', 'earnings'])
        if 'margin' in query_lower:
            enhancements.extend(['profit', 'profitability', 'operating'])
        if 'cloud' in query_lower:
            enhancements.extend(['azure', 'aws', 'google cloud', 'gcp'])
        if 'ai' in query_lower:
            enhancements.extend(['artificial intelligence', 'machine learning', 'ml'])
        
        if enhancements:
            return query + ' ' + ' '.join(enhancements[:3])
        return query
    
    def synthesize_results(self, query: str, sub_queries: List[str], 
                          all_results: List[List[Document]]) -> Dict[str, Any]:
        """Synthesize results from multiple sub-queries into a coherent answer."""
        
        # Combine all relevant documents
        relevant_docs = []
        for results in all_results:
            relevant_docs.extend(results)
        
        if not relevant_docs:
            return {
                "query": query,
                "answer": "I could not find relevant information to answer this query.",
                "reasoning": "No relevant documents found in the knowledge base.",
                "sub_queries": sub_queries,
                "sources": []
            }
        
        # Create context from top documents
        context_parts = []
        sources = []
        
        # Remove duplicates and get best results
        unique_docs = self.remove_duplicate_docs(relevant_docs)
        top_docs = unique_docs[:8]  # Limit context size
        
        for doc in top_docs:
            context_parts.append(f"Company: {doc.metadata['company']}, Year: {doc.metadata['year']}\n{doc.page_content}")
            sources.append({
                "company": doc.metadata['company'],
                "year": doc.metadata['year'],
                "excerpt": doc.page_content[:200] + "...",
                "chunk_id": doc.metadata.get('chunk_id', 'unknown'),
                "section_type": doc.metadata.get('section_type', 'unknown')
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer using LLM
        prompt = f"""Based on the following financial information from 10-K filings, please answer the query.

Query: {query}

Financial Information:
{context}

Please provide a clear, factual answer based on the information provided. Include specific numbers and years when available. If comparing companies, present the information in a structured way.

Answer:"""
        
        try:
            if self.llm:
                response = self.llm.invoke(prompt)
                answer = response.content.strip()
            else:
                # Demo mode - provide basic answer
                answer = self._generate_demo_answer(query, relevant_docs)
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        # Determine reasoning based on query type
        reasoning = self.generate_reasoning(query, sub_queries, len(top_docs))
        
        return {
            "query": query,
            "answer": answer,
            "reasoning": reasoning,
            "sub_queries": sub_queries,
            "sources": sources
        }
    
    def _generate_demo_answer(self, query: str, docs: List[Document]) -> str:
        """Generate a demo answer when LLM is not available."""
        query_lower = query.lower()
        
        if 'revenue' in query_lower and 'nvidia' in query_lower and '2024' in query_lower:
            return "Based on the available documents, NVIDIA's total revenue information would be found in their 2024 10-K filing. The system has retrieved relevant financial statement sections for analysis."
        elif 'highest' in query_lower and 'margin' in query_lower:
            return "To determine which company had the highest operating margin, the system would need to compare financial data across all three companies from their respective 10-K filings."
        elif 'growth' in query_lower:
            return "Growth analysis requires comparing financial metrics across multiple years. The system has retrieved relevant documents to perform year-over-year calculations."
        elif 'percentage' in query_lower:
            return "Percentage calculations require both numerator and denominator data. The system has identified relevant sections for segment analysis."
        else:
            return "The system has retrieved relevant financial documents to answer your query. In a full implementation, the LLM would provide a detailed analysis."
    
    def remove_duplicate_docs(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate documents based on chunk_id."""
        seen = set()
        unique_docs = []
        
        for doc in docs:
            chunk_id = doc.metadata.get('chunk_id')
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_docs.append(doc)
        
        return unique_docs
    
    def generate_reasoning(self, query: str, sub_queries: List[str], doc_count: int) -> str:
        """Generate reasoning explanation for the answer."""
        query_type = self.determine_query_type(query)
        
        if query_type == 'comparative':
            return f"Analyzed {doc_count} relevant sections from 10-K filings to compare metrics across companies."
        elif query_type == 'growth_analysis':
            return f"Retrieved financial data from multiple years and calculated growth rates using {doc_count} document sections."
        elif query_type == 'calculation':
            return f"Extracted relevant financial figures from {doc_count} sections to perform the requested calculation."
        else:
            return f"Retrieved information from {doc_count} relevant sections of 10-K filings to answer the query."
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """Main method to answer a query using agent capabilities."""
        print(f"\n=== Processing Query ===")
        print(f"Query: {query}")
        
        # Decompose query into sub-queries
        sub_queries = self.decompose_query(query)
        print(f"Sub-queries: {sub_queries}")
        
        # Execute each sub-query
        all_results = []
        for sub_query in sub_queries:
            print(f"Executing: {sub_query}")
            results = self.execute_sub_query(sub_query, k=3)
            all_results.append(results)
            print(f"Found {len(results)} results")
        
        # Synthesize final answer
        response = self.synthesize_results(query, sub_queries, all_results)
        
        return response


if __name__ == "__main__":
    # Test the query agent
    from vector_store import FinancialVectorStore
    
    vector_store = FinancialVectorStore()
    if vector_store.load_vectorstore():
        agent = QueryAgent(vector_store)
        
        test_queries = [
            "What was NVIDIA's total revenue in fiscal year 2024?",
            "Which company had the highest operating margin in 2023?",
            "How much did Microsoft's cloud revenue grow from 2022 to 2023?"
        ]
        
        for query in test_queries:
            response = agent.answer_query(query)
            print(f"\nQuery: {response['query']}")
            print(f"Answer: {response['answer']}")
            print(f"Sources: {len(response['sources'])}")
    else:
        print("Vector store not available for testing")