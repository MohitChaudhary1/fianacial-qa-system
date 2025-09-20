"""Document processing module for extracting and chunking text from 10-K filings - Assignment Requirements."""

import os
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import config


class DocumentProcessor:
    """Processes 10-K HTML filings and creates text chunks - Assignment Requirements."""
    
    def __init__(self):
        # Assignment Requirements: 200-1000 tokens, semantic chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,  # ~600-800 tokens
            chunk_overlap=config.CHUNK_OVERLAP,  # ~75-100 tokens
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Semantic boundaries
        )
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        print(f"📁 Created processed data directory: {config.PROCESSED_DATA_DIR}")
    
    def clean_html_text(self, html_content: str) -> str:
        """Extract and clean text from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def extract_key_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from 10-K filing - Assignment Requirements."""
        sections = {}
        
        # Assignment Requirements: Focus on key sections for financial analysis
        section_patterns = {
            'business_overview': r'item\s+1[\.\s]*business|part\s+i\s+item\s+1',
            'risk_factors': r'item\s+1a[\.\s]*risk\s+factors',  # AI risks mentioned here
            'md_a': r'item\s+7[\.\s]*management[\'s\s]*discussion\s+and\s+analysis',  # Key financial metrics
            'financial_statements': r'item\s+8[\.\s]*financial\s+statements'  # Revenue, margins, etc.
        }
        
        text_lower = text.lower()
        
        for section_name, pattern in section_patterns.items():
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            if matches:
                start_pos = matches[0].start()
                
                # Find the end (next major section or end of document)
                next_section_patterns = [
                    r'item\s+\d+[a-z]*[\.\s]*[a-z\s]+',
                    r'part\s+[iv]+',
                    r'signatures',
                    r'exhibits'
                ]
                
                end_pos = len(text)
                for next_pattern in next_section_patterns:
                    next_matches = list(re.finditer(next_pattern, text_lower[start_pos + 100:], re.IGNORECASE))
                    if next_matches:
                        candidate_end = start_pos + 100 + next_matches[0].start()
                        if candidate_end > start_pos + 1000:  # Ensure reasonable section size
                            end_pos = min(end_pos, candidate_end)
                
                section_text = text[start_pos:end_pos]
                if len(section_text) > 500:  # Only include substantial sections
                    sections[section_name] = section_text
        
        return sections
    
    def create_document_chunks(self, file_path: str, company: str, year: int) -> List[Document]:
        """Process a single 10-K filing and create document chunks."""
        print(f"Processing {company} {year} filing: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
        
        # Extract text from HTML
        text = self.clean_html_text(html_content)
        
        if len(text) < 1000:
            print(f"Warning: Very short document for {company} {year}")
            return []
        
        # Extract key sections
        sections = self.extract_key_sections(text)
        
        # Create chunks from the full document and sections
        all_chunks = []
        
        # Process full document
        full_doc_chunks = self.text_splitter.split_text(text)
        for i, chunk in enumerate(full_doc_chunks):
            if len(chunk.strip()) > 100:  # Filter out very short chunks
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'company': company,
                        'year': year,
                        'source_file': os.path.basename(file_path),
                        'chunk_id': f"{company}_{year}_full_{i}",
                        'section_type': 'full_document',
                        'chunk_size': len(chunk)
                    }
                )
                all_chunks.append(doc)
        
        # Process individual sections with higher priority
        for section_name, section_text in sections.items():
            if len(section_text) > 500:  # Only process substantial sections
                section_chunks = self.text_splitter.split_text(section_text)
                for i, chunk in enumerate(section_chunks):
                    if len(chunk.strip()) > 100:
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                'company': company,
                                'year': year,
                                'source_file': os.path.basename(file_path),
                                'chunk_id': f"{company}_{year}_{section_name}_{i}",
                                'section_type': section_name,
                                'chunk_size': len(chunk)
                            }
                        )
                        all_chunks.append(doc)
        
        print(f"Created {len(all_chunks)} chunks for {company} {year}")
        return all_chunks
    
    def process_all_documents(self, filings: Dict[str, Dict[int, str]]) -> List[Document]:
        """Process all downloaded filings and create document chunks."""
        all_documents = []
        
        for company, year_files in filings.items():
            for year, file_path in year_files.items():
                if os.path.exists(file_path):
                    chunks = self.create_document_chunks(file_path, company, year)
                    all_documents.extend(chunks)
                else:
                    print(f"File not found: {file_path}")
        
        print(f"\nTotal documents created: {len(all_documents)}")
        
        # Print summary by company and year
        summary = {}
        for doc in all_documents:
            company = doc.metadata['company']
            year = doc.metadata['year']
            key = f"{company}_{year}"
            summary[key] = summary.get(key, 0) + 1
        
        print("\nDocument summary:")
        for key, count in sorted(summary.items()):
            print(f"  {key}: {count} chunks")
        
        return all_documents


if __name__ == "__main__":
    # Test with sample filings
    processor = DocumentProcessor()
    
    # Mock filings dict for testing
    test_filings = {}
    for file in os.listdir(config.RAW_DATA_DIR):
        if file.endswith('.htm'):
            parts = file.replace('.htm', '').split('_')
            if len(parts) >= 3:
                company = parts[0]
                year = int(parts[1])
                if company not in test_filings:
                    test_filings[company] = {}
                test_filings[company][year] = os.path.join(config.RAW_DATA_DIR, file)
    
    documents = processor.process_all_documents(test_filings)
    print(f"Processed {len(documents)} document chunks")