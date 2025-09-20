"""SEC Filing downloader module - Assignment Requirements."""

import os
import re
import time
import requests
from typing import Dict, List, Optional
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm
import config


class SECDownloader:
    """Downloads 10-K filings from SEC EDGAR database - Assignment Requirements."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.USER_AGENT,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        })
        
        # Create directories
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
        print(f"📁 Created data directory: {config.RAW_DATA_DIR}")
        
    def get_filing_urls(self, cik: str, years: List[int]) -> Dict[int, str]:
        """Get 10-K filing URLs for specified years - Assignment Requirements."""
        filing_urls = {}
        
        # Browse EDGAR for the company - Assignment Requirements
        browse_url = f"{config.SEC_BASE_URL}/cgi-bin/browse-edgar"
        params = {
            'CIK': cik,
            'type': '10-K',
            'dateb': '',
            'owner': 'exclude',
            'count': '40'
        }
        
        print(f"🔍 Fetching filing list for CIK {cik}...")
        
        try:
            response = self.session.get(browse_url, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"❌ Error fetching filings for CIK {cik}: {e}")
            return filing_urls
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find filing table
        filing_table = soup.find('table', class_='tableFile2')
        if not filing_table:
            print(f"⚠️  No filings table found for CIK {cik}")
            return filing_urls
            
        rows = filing_table.find_all('tr')[1:]  # Skip header
        print(f"📋 Found {len(rows)} filing entries")
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 4:
                continue
                
            filing_type = cells[0].text.strip()
            filing_date = cells[3].text.strip()
            
            # Only process 10-K filings (not 10-K/A amendments) - Assignment Requirements
            if filing_type != '10-K':
                continue
                
            # Extract year from filing date
            try:
                filing_year = int(filing_date.split('-')[0])
            except (ValueError, IndexError):
                continue
                
            # Check if this year is in our target years - Assignment Requirements
            if filing_year not in years or filing_year in filing_urls:
                continue
                
            # Get the documents link
            docs_link = cells[1].find('a')
            if docs_link:
                docs_url = urljoin(config.SEC_BASE_URL, docs_link['href'])
                filing_urls[filing_year] = docs_url
                print(f"✅ Found {filing_year} 10-K filing")
                
        return filing_urls
    
    def get_10k_document_url(self, documents_url: str) -> Optional[str]:
        """Extract the actual 10-K document URL from documents page - Assignment Requirements."""
        print(f"Getting document URL from: {documents_url}")
        
        try:
            response = self.session.get(documents_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"❌ Error fetching documents page: {e}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main 10-K document (usually first .htm file)
        table = soup.find('table', class_='tableFile')
        if not table:
            print("⚠️  No tableFile found in documents page")
            return None
            
        rows = table.find_all('tr')[1:]  # Skip header
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 4:
                continue
                
            doc_type = cells[3].text.strip()
            if doc_type == '10-K':
                link = cells[2].find('a')
                if link and link['href'].endswith('.htm'):
                    doc_url = urljoin(config.SEC_BASE_URL, link['href'])
                    print(f"✅ Found 10-K document URL: {doc_url}")
                    return doc_url
                    
        print("⚠️  No 10-K document found in documents page")
        return None
    
    def download_filing(self, url: str, save_path: str) -> bool:
        """Download a single filing - Assignment Requirements."""
        try:
            print(f"⬇️  Downloading: {os.path.basename(save_path)}")
            
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            # Ensure content is valid
            if len(response.text) < 1000:
                print(f"⚠️  Warning: Downloaded content seems too short ({len(response.text)} chars)")
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
                
            print(f"✅ Saved to: {save_path} ({len(response.text)} chars)")
            time.sleep(0.1)  # Be respectful to SEC servers - Assignment Requirements
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
            return False
    
    def download_all_filings(self) -> Dict[str, Dict[int, str]]:
        """Download all 10-K filings for specified companies and years."""
        all_filings = {}
        
        for symbol, info in config.COMPANIES.items():
            print(f"\n=== Processing {info['name']} ({symbol}) ===")
            company_filings = {}
            
            # Get filing URLs
            filing_urls = self.get_filing_urls(info['cik'], config.YEARS)
            
            if not filing_urls:
                print(f"No filings found for {symbol}")
                continue
                
            for year, docs_url in filing_urls.items():
                print(f"\nProcessing {year} filing...")
                
                # Get actual document URL
                doc_url = self.get_10k_document_url(docs_url)
                if not doc_url:
                    print(f"Could not find 10-K document for {symbol} {year}")
                    continue
                
                # Download the filing
                filename = f"{symbol}_{year}_10K.htm"
                save_path = os.path.join(config.RAW_DATA_DIR, filename)
                
                if os.path.exists(save_path):
                    print(f"File already exists: {save_path}")
                    company_filings[year] = save_path
                    continue
                
                if self.download_filing(doc_url, save_path):
                    company_filings[year] = save_path
                else:
                    print(f"Failed to download {symbol} {year}")
            
            all_filings[symbol] = company_filings
            
        return all_filings


if __name__ == "__main__":
    downloader = SECDownloader()
    filings = downloader.download_all_filings()
    
    print("\n=== Download Summary ===")
    for symbol, years in filings.items():
        print(f"{symbol}: {len(years)} filings downloaded")
        for year in years:
            print(f"  - {year}")