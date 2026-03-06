"""
AnuRAG: Title Loader Module
Loads and manages paper titles from the database
"""

import json
import os
from typing import Dict, List, Optional


def load_titles(titles_path: str = "../finalAgent_db/titles.json") -> Dict[str, str]:
    """
    Load and return titles from titles.json
    
    Args:
        titles_path: Path to the titles JSON file
        
    Returns:
        Dictionary of numbered titles
    """
    try:
        with open(titles_path, 'r', encoding='utf-8') as f:
            titles = json.load(f)
        return titles
    except FileNotFoundError:
        print(f"Warning: titles.json not found at {titles_path}")
        # Try to generate titles from documents.json
        return generate_titles_from_documents()
    except json.JSONDecodeError as e:
        print(f"Error parsing titles.json: {e}")
        return {}
    except Exception as e:
        print(f"Error loading titles: {e}")
        return {}


def generate_titles_from_documents(documents_path: str = "../finalAgent_db/documents.json",
                                    output_path: str = "../finalAgent_db/titles.json") -> Dict[str, str]:
    """
    Generate titles.json from documents.json using filenames or extracted titles.
    
    Args:
        documents_path: Path to the documents JSON file
        output_path: Path to save the generated titles
        
    Returns:
        Dictionary of titles
    """
    titles = {}
    
    try:
        with open(documents_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        for i, doc in enumerate(documents, 1):
            # Try to get title from document, otherwise use filename
            if 'title' in doc and doc['title']:
                title = doc['title']
            elif 'filename' in doc:
                # Remove .pdf extension and clean up filename
                title = doc['filename'].replace('.pdf', '').replace('_', ' ')
            elif 'pdf_path' in doc:
                title = os.path.basename(doc['pdf_path']).replace('.pdf', '').replace('_', ' ')
            else:
                title = f"Document {i}"
            
            titles[str(i)] = title
        
        # Save the generated titles
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(titles, f, indent=2, ensure_ascii=False)
        
        print(f"Generated titles.json with {len(titles)} titles")
        
    except FileNotFoundError:
        print(f"documents.json not found at {documents_path}")
    except Exception as e:
        print(f"Error generating titles: {e}")
    
    return titles


def get_titles_list() -> List[str]:
    """
    Get a list of all paper titles.
    
    Returns:
        List of title strings
    """
    titles = load_titles()
    return list(titles.values())


def search_titles(query: str) -> List[str]:
    """
    Search for titles containing the query string.
    
    Args:
        query: Search string (case-insensitive)
        
    Returns:
        List of matching titles
    """
    titles = load_titles()
    query_lower = query.lower()
    
    matching = []
    for title in titles.values():
        if query_lower in title.lower():
            matching.append(title)
    
    return matching


def get_title_by_id(doc_id: str) -> Optional[str]:
    """
    Get a title by document ID.
    
    Args:
        doc_id: Document ID (e.g., "1" or "doc_1")
        
    Returns:
        Title string or None if not found
    """
    titles = load_titles()
    
    # Handle both "1" and "doc_1" formats
    if doc_id.startswith("doc_"):
        doc_id = doc_id.replace("doc_", "")
    
    return titles.get(doc_id)


# Example usage if run directly
if __name__ == "__main__":
    print("Loading titles...")
    titles = load_titles()
    
    if titles:
        print(f"\nLoaded {len(titles)} titles:")
        for i, (key, title) in enumerate(titles.items()):
            if i < 5:  # Print first 5
                print(f"  {key}: {title[:80]}..." if len(title) > 80 else f"  {key}: {title}")
        if len(titles) > 5:
            print(f"  ... and {len(titles) - 5} more")
    else:
        print("No titles found. Run pdf2json_chunked.py first to process PDFs.")
