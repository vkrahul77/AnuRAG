"""
AnuRAG: Web Scraper Module
Utility for scraping additional resources from the web
"""

import os
import time
import requests
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# Import centralized configuration
from config import GEMINI_CHAT_MODEL

# Import Gemini for summarization
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
except ImportError:
    GEMINI_AVAILABLE = False


def fetch_webpage(url: str) -> Optional[str]:
    """Fetch and extract text content from a webpage."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove scripts and styles
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Extract text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text
        
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def summarize_content(content: str, question: str = None) -> str:
    """Summarize webpage content using Gemini."""
    if not GEMINI_AVAILABLE:
        return content[:2000]  # Return truncated content if Gemini not available
    
    try:
        model = genai.GenerativeModel(GEMINI_CHAT_MODEL)
        
        prompt = f"""Summarize the following web content, focusing on technical information 
about analog circuit design, specifications, and implementations.

{f'Specifically answer: {question}' if question else ''}

CONTENT:
{content[:50000]}

Provide a concise, technical summary."""

        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error summarizing: {e}")
        return content[:2000]


def search_arxiv(query: str, max_results: int = 5) -> list:
    """Search arXiv for relevant papers."""
    try:
        import urllib.parse
        
        base_url = "http://export.arxiv.org/api/query?"
        search_query = urllib.parse.quote(query)
        url = f"{base_url}search_query=all:{search_query}&start=0&max_results={max_results}"
        
        response = requests.get(url, timeout=30)
        soup = BeautifulSoup(response.content, 'xml')
        
        results = []
        entries = soup.find_all('entry')
        
        for entry in entries:
            result = {
                'title': entry.find('title').text.strip() if entry.find('title') else '',
                'summary': entry.find('summary').text.strip() if entry.find('summary') else '',
                'authors': [a.find('name').text for a in entry.find_all('author')],
                'link': entry.find('id').text if entry.find('id') else '',
                'published': entry.find('published').text if entry.find('published') else ''
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return []


def main(query: str) -> Dict[str, Any]:
    """
    Main function for web scraping.
    
    Args:
        query: URL to scrape or search query
        
    Returns:
        Dictionary with scraped content
    """
    if query.startswith('http://') or query.startswith('https://'):
        # Fetch and summarize webpage
        content = fetch_webpage(query)
        if content:
            summary = summarize_content(content)
            return {
                'type': 'webpage',
                'url': query,
                'content': content[:5000],
                'summary': summary
            }
        return {'error': 'Failed to fetch webpage'}
    
    else:
        # Search arXiv
        results = search_arxiv(query)
        return {
            'type': 'arxiv_search',
            'query': query,
            'results': results
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Web scraper for AnuRAG')
    parser.add_argument('--url', type=str, help='URL to scrape')
    parser.add_argument('--search', type=str, help='arXiv search query')
    
    args = parser.parse_args()
    
    if args.url:
        result = main(args.url)
    elif args.search:
        result = main(args.search)
    else:
        print("Please provide --url or --search argument")
        exit(1)
    
    import json
    print(json.dumps(result, indent=2))
