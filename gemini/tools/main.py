"""
AnuRAG: Main Entry Point
Multimodal Large Language Model for Analog Circuit Design using Google Gemini

This module provides the main entry point for the AnuRAG system, which can:
1. Process PDF research papers and extract text, images, equations, and schematics
2. Create contextual embeddings using Google Gemini
3. Perform hybrid search (semantic + BM25) with reranking
4. Answer questions using a ReAct-style agent
"""

import os
import sys
import re
import json
import shutil
import argparse
import time
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Import project modules
from messages import system_message, system_message_2, system_message_stage1, system_message_stage2, _build_stage2_prompt
from search import main as search_db
from pdf2json_chunked import main as pdf2json_chunked
from fullcontext import main as full_document_search
from load_titles import load_titles
from config import (
    GEMINI_CHAT_MODEL, 
    TEMPERATURE, 
    MAX_OUTPUT_TOKENS,
    LLM_PROVIDER,
    get_active_chat_model,
    get_model_costs,
    get_lut_info,
    LUT_NMOS_PATH,
    LUT_PMOS_PATH
)
from llm_provider import get_llm_provider

# Import Google Generative AI - try new package first, fall back to deprecated
GEMINI_AVAILABLE = False
genai = None
genai_client = None
USE_NEW_API = False

# Try the new google-genai package first (recommended)
try:
    from google import genai as new_genai
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        genai_client = new_genai.Client(api_key=api_key)
        GEMINI_AVAILABLE = True
        USE_NEW_API = True
        print("... Using new google-genai package (recommended)")
except ImportError:
    pass

# Fallback to deprecated google-generativeai package
if not GEMINI_AVAILABLE:
    try:
        import google.generativeai as genai_old
        genai = genai_old
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            GEMINI_AVAILABLE = True
            print("  Using deprecated google-generativeai package. Consider upgrading to google-genai.")
    except ImportError:
        print("Error: No Gemini package installed. Run: pip install google-genai")
        sys.exit(1)

if not GEMINI_AVAILABLE:
    print("Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in environment")
    print("Please set your API key in a .env file or environment variable")
    sys.exit(1)

# Cost per million tokens -- use config-driven costs
def calculate_cost(input_tokens: int, output_tokens: int) -> Dict[str, float]:
    """Calculate estimated cost based on token usage and active model."""
    costs = get_model_costs()
    input_cost = (input_tokens / 1_000_000) * costs.get('input', 0.10)
    output_cost = (output_tokens / 1_000_000) * costs.get('output', 0.40)
    total_cost = input_cost + output_cost
    return {
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens
    }


def extract_topology_images(search_results: Dict, answer_text: str, output_dir: str = "output_images") -> Dict[str, List[str]]:
    """
    Extract topology images from search results and copy them to output directory.
    
    This function:
    1. Identifies images related to OTA/amplifier topologies from search results
    2. Also searches the database for images from documents found in search
    3. Copies relevant images to the output directory
    4. Returns a mapping of topology names to image paths
    
    Args:
        search_results: Dictionary containing search results with 'text' and 'images' keys
        answer_text: The model's answer text to identify mentioned topologies
        output_dir: Directory to save the images
        
    Returns:
        Dictionary mapping topology names to list of image paths
    """
    import shutil
    import pickle
    from datetime import datetime
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"topology_images_{timestamp}")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    saved_images = []
    
    # Keywords to identify topology-related images
    topology_keywords = [
        'ota', 'amplifier', 'cascode', 'folded', 'telescopic', 'miller', 
        'two-stage', 'two stage', 'gain boost', 'regulated', 'feedforward',
        'current mirror', 'differential', 'opamp', 'op-amp', 'operational',
        'schematic', 'circuit', 'topology', 'architecture', 'adc', 'dac',
        'comparator', 'buffer', 'stage'
    ]
    
    if not search_results:
        return {'images': [], 'output_dir': output_path, 'count': 0}
    
    # Collect document IDs from search results
    doc_ids_found = set()
    results = search_results.get('text', [])
    
    for result in results:
        item = result.get('item', {})
        doc_id = item.get('doc_id', '')
        if doc_id:
            doc_ids_found.add(doc_id)
    
    # Process direct image results from search
    for idx, result in enumerate(results):
        if result.get('content_type') == 'image':
            item = result.get('item', {})
            image_path = item.get('path', '')
            
            if not image_path:
                continue
            
            # Handle relative paths
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.path.dirname(__file__), image_path)
            
            if not os.path.exists(image_path):
                continue
            
            context = (item.get('contextualized_content', '') or '').lower()
            doc_id = item.get('doc_id', 'unknown')
            
            # Check if topology-related
            is_topology_image = any(kw in context for kw in topology_keywords)
            
            if is_topology_image or doc_id in doc_ids_found:
                try:
                    filename = os.path.basename(image_path)
                    new_filename = f"{idx+1:02d}_{doc_id}_{filename}"
                    dest_path = os.path.join(output_path, new_filename)
                    
                    shutil.copy2(image_path, dest_path)
                    saved_images.append({
                        'original_path': image_path,
                        'saved_path': dest_path,
                        'doc_id': doc_id,
                        'context': context[:200]
                    })
                except Exception as e:
                    print(f"Warning: Could not copy image {image_path}: {e}")
    
    # Load the database to find images from documents in search results
    db_path = os.path.join(os.path.dirname(__file__), '../finalAgent_db/base_db/vector_db.pkl')
    if os.path.exists(db_path) and doc_ids_found:
        try:
            with open(db_path, 'rb') as f:
                data = pickle.load(f)
            
            metadata = data.get('metadata', [])
            
            # Find images from the same documents
            for meta in metadata:
                if 'image_id' not in meta:
                    continue
                    
                doc_id = meta.get('doc_id', '')
                if doc_id not in doc_ids_found:
                    continue
                
                image_path = meta.get('path', '')
                if not image_path:
                    continue
                
                # Handle relative paths
                if not os.path.isabs(image_path):
                    image_path = os.path.join(os.path.dirname(__file__), image_path)
                
                if not os.path.exists(image_path):
                    continue
                
                # Check if already saved
                if any(img['original_path'] == image_path for img in saved_images):
                    continue
                
                context = (meta.get('contextualized_content', '') or '').lower()
                
                # Check if topology-related
                is_topology_image = any(kw in context for kw in topology_keywords)
                
                if is_topology_image:
                    try:
                        filename = os.path.basename(image_path)
                        image_id = meta.get('image_id', 'unknown')
                        new_filename = f"db_{doc_id}_{image_id}_{filename}"
                        dest_path = os.path.join(output_path, new_filename)
                        
                        shutil.copy2(image_path, dest_path)
                        saved_images.append({
                            'original_path': image_path,
                            'saved_path': dest_path,
                            'doc_id': doc_id,
                            'context': context[:200],
                            'image_id': image_id
                        })
                        
                        # Limit to 15 images max
                        if len(saved_images) >= 15:
                            break
                            
                    except Exception as e:
                        pass
                        
        except Exception as e:
            print(f"Warning: Could not load database for image search: {e}")
    
    # Also check the images dictionary directly from search results
    images_dict = search_results.get('images', {})
    for idx, image_path in images_dict.items():
        if not image_path:
            continue
            
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(__file__), image_path)
            
        if not os.path.exists(image_path):
            continue
            
        # Check if already saved
        if any(img['original_path'] == image_path for img in saved_images):
            continue
            
        try:
            filename = os.path.basename(image_path)
            new_filename = f"img_{idx}_{filename}"
            dest_path = os.path.join(output_path, new_filename)
            
            shutil.copy2(image_path, dest_path)
            saved_images.append({
                'original_path': image_path,
                'saved_path': dest_path,
                'doc_id': f'result_{idx}',
                'context': ''
            })
        except Exception as e:
            pass
    
    if saved_images:
        print(f"\n" Saved {len(saved_images)} topology images to: {output_path}")
        for img in saved_images[:5]:
            print(f"   - {os.path.basename(img['saved_path'])}")
        if len(saved_images) > 5:
            print(f"   ... and {len(saved_images) - 5} more")
    else:
        print(f"\n" No topology images found in search results")
    
    return {
        'images': saved_images,
        'output_dir': output_path,
        'count': len(saved_images)
    }


def format_anurag_output(text: str, sources_found: List[str] = None, topology_images: Dict = None) -> str:
    """
    Format the output with the AnuRAG branded header for topology analysis questions.
    
    This function ensures the output follows the expected AnuRAG format:
    - Adds the "### AnuRAG Topology Analysis" header if missing
    - Adds the Status line with sources
    - Adds image references section if images are available
    - Cleans up table formatting issues
    
    Args:
        text: The model's response text
        sources_found: List of document sources found during search
        topology_images: Dictionary with extracted topology images info
        
    Returns:
        Formatted text with AnuRAG branding
    """
    if not text:
        return text
    
    # Check if this looks like a topology analysis response
    topology_keywords = ['topology', 'ota', 'amplifier', 'cascode', 'folded', 'telescopic', 
                         'miller', 'two-stage', 'gain boost', 'feasibility', 'swing']
    is_topology_response = any(kw.lower() in text.lower() for kw in topology_keywords)
    
    if not is_topology_response:
        return text
    
    # Check if header already exists
    has_anurag_header = '### AnuRAG Topology Analysis' in text
    
    if not has_anurag_header:
        # Build the header
        header_lines = ["### AnuRAG Topology Analysis"]
        
        if sources_found:
            sources_str = ", ".join(sources_found[:5])  # Limit to 5 sources
            header_lines.append(f"**Status:** Retrieval Complete (Sources: {sources_str})")
        else:
            header_lines.append("**Status:** Analysis Complete")
        
        header_lines.append("")
        header = "\n".join(header_lines)
        
        # Remove any existing generic headers and prepend our header
        text = re.sub(r'^#+\s*(?:Topology|OTA|Amplifier|Circuit|Analysis)[^\n]*\n*', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = header + text
    
    # Add topology images section if images are available
    if topology_images and topology_images.get('count', 0) > 0:
        # Collect unique source doc IDs
        source_docs = sorted(set(img.get('doc_id', 'unknown') for img in topology_images.get('images', [])))
        sources_str = ", ".join(source_docs[:5])
        images_section = f"\n\n---\n" **{topology_images['count']} reference images** saved from {sources_str}"
        images_section += f"  \n" `{topology_images['output_dir']}`\n"
        text += images_section
    
    # Fix table formatting - ensure each row is on its own line
    # Find tables and fix them
    table_pattern = r'\|[^\n]+\|(?:\s*\|[^\n]+\|)+'
    
    def fix_table(match):
        table_text = match.group(0)
        # Split on | but preserve the structure
        # Detect if rows are concatenated (no newlines between |...|...|)
        if '\n' not in table_text.strip():
            # All on one line - need to split
            # Find row boundaries: each row should start with | and end with |
            rows = re.findall(r'\|[^|]+(?:\|[^|]+)+\|', table_text)
            if len(rows) > 1:
                return '\n'.join(rows)
        return table_text
    
    text = re.sub(table_pattern, fix_table, text)
    
    return text


def clean_thinking_artifacts(text: str) -> str:
    """
    Remove internal thinking artifacts that the model shouldn't output.
    
    These are patterns like "(Self-Correction)", "One detail:", "Let's write.", etc.
    that indicate the model is dumping its chain-of-thought instead of final answer.
    """
    import re
    
    # First, try to extract just the Answer section if it exists
    # This is the cleanest way to get just the final response
    # Priority: Look for AnuRAG formatted answer first, then generic Answer:
    answer_markers = [
        r'(?:^|\n)\s*### AnuRAG',  # AnuRAG branded header (highest priority)
        r'(?:^|\n)\s*Answer:\s*(?:\n\s*)?### AnuRAG',  # Answer: followed by AnuRAG
        r'(?:^|\n)\s*Answer:\s*',
        r'(?:^|\n)\s*ANSWER:\s*',
        r'(?:^|\n)\s*Final Answer:\s*',
        r'(?:^|\n)\s*## Answer\s*',
    ]
    
    for marker in answer_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            # Found an Answer section - extract everything after it
            # For AnuRAG header, include it in the output
            if 'AnuRAG' in marker:
                answer_text = text[match.start():].strip()
                # Remove any leading "Answer:" if present before AnuRAG
                answer_text = re.sub(r'^Answer:\s*\n?', '', answer_text, flags=re.IGNORECASE)
            else:
                answer_text = text[match.end():].strip()
            
            # But stop at "Thought:" or "Action:" if they appear (shouldn't in answer)
            stop_match = re.search(r'(?:^|\n)\s*(?:Thought|Action):\s*', answer_text)
            if stop_match:
                answer_text = answer_text[:stop_match.start()].strip()
            if len(answer_text) > 100:  # Only use if substantial
                text = answer_text
                break
    
    # Patterns that indicate internal reasoning (should not be in final answer)
    thinking_patterns = [
        # Meta-commentary patterns
        r'\(Self-Correction\):?[^\n]*\n?',
        r'\(End of thought process\)[^\n]*\n?',
        r'\(End\)[^\n]*\n?',
        r'\(Wait,[^)]+\)[^\n]*\n?',
        
        # Planning statements
        r'One detail:[^\n]*\n?',
        r'One more thing:[^\n]*\n?',
        r'One minor point:[^\n]*\n?',
        r'One correction:[^\n]*\n?',
        r'One last check[^\n]*\n?',
        r'One specific detail:[^\n]*\n?',
        
        # Intent statements ("I will...", "Let me...", "Let's...")
        r"(?:^|\n)I will do this\.?\s*\n?",
        r"(?:^|\n)I will write[^\n]*\n?",
        r"(?:^|\n)I will provide[^\n]*\n?",
        r"(?:^|\n)I will use[^\n]*\n?",
        r"(?:^|\n)I will assume[^\n]*\n?",
        r"(?:^|\n)I will add[^\n]*\n?",
        r"(?:^|\n)I will include[^\n]*\n?",
        r"(?:^|\n)I will proceed[^\n]*\n?",
        r"(?:^|\n)I will ensure[^\n]*\n?",
        r"(?:^|\n)I will note[^\n]*\n?",
        r"(?:^|\n)I will mention[^\n]*\n?",
        r"(?:^|\n)I will formulate[^\n]*\n?",
        r"(?:^|\n)I will list[^\n]*\n?",
        r"(?:^|\n)I will produce[^\n]*\n?",
        r"(?:^|\n)I will generate[^\n]*\n?",
        r"(?:^|\n)I will create[^\n]*\n?",
        r"(?:^|\n)I will select[^\n]*\n?",
        r"(?:^|\n)I will calculate[^\n]*\n?",
        r"(?:^|\n)I will reference[^\n]*\n?",
        r"(?:^|\n)I will stick[^\n]*\n?",
        r"(?:^|\n)I will follow[^\n]*\n?",
        r"(?:^|\n)I will report[^\n]*\n?",
        r"(?:^|\n)I will recommend[^\n]*\n?",
        r"(?:^|\n)I will be[^\n]*\n?",
        r"(?:^|\n)I'm ready[^\n]*\n?",
        r"(?:^|\n)I'm done[^\n]*\n?",
        r"(?:^|\n)I have done[^\n]*\n?",
        r"(?:^|\n)I have [0-9]+ OTAs?[^\n]*\n?",
        r"(?:^|\n)Let's write\.[^\n]*\n?",
        r"(?:^|\n)Let's go\.[^\n]*\n?",
        r"(?:^|\n)Let me [^\n]*\n?",
        r"(?:^|\n)Let's proceed\.[^\n]*\n?",
        
        # Checking/verification statements
        r"(?:^|\n)Final check[:\.]?[^\n]*\n?",
        r"(?:^|\n)Final check on[^\n]*\n?",
        r"(?:^|\n)Double check[:\.]?[^\n]*\n?",
        r"(?:^|\n)Double-check[^\n]*\n?",
        r"(?:^|\n)Ready\.[^\n]*\n?",
        r"(?:^|\n)Plan:[^\n]*\n?",
        r"(?:^|\n)Structure:[^\n]*\n?",
        
        # Search deliberations
        r"(?:^|\n)I'll search for[^\n]*\n?",
        r"(?:^|\n)I'll use[^\n]*\n?",
        r"(?:^|\n)Better [0-9]+th option[^\n]*\n?",
        r'(?:^|\n)"[^"]+" in doc_[0-9]+\.?\s*\n?',  # "X" in doc_123.
        r"(?:^|\n)Actually, doc_[0-9]+[^\n]*\n?",
        r"(?:^|\n)Doc_[0-9]+ mentions[^\n]*\n?",
        r"(?:^|\n)Doc_[0-9]+ says[^\n]*\n?",
        r"(?:^|\n)Doc_[0-9]+ describes[^\n]*\n?",
        r"(?:^|\n)Wait, [^\n]*\n?",
        r"(?:^|\n)Actually, [^\n]*\n?",
        r"(?:^|\n)Hmm,? [^\n]*\n?",
        
        # Self-talk about formatting
        r"(?:^|\n)This confirms[^\n]*\n?",
        r"(?:^|\n)This means[^\n]*\n?",
        r"(?:^|\n)This is the[^\n]*\n?",
        r"(?:^|\n)This covers[^\n]*\n?",
        r"(?:^|\n)These are the[^\n]*\n?",
        r"(?:^|\n)These are safe\.[^\n]*\n?",
    ]
    
    cleaned = text
    for pattern in thinking_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove blocks of repeated "I will do this" lines
    cleaned = re.sub(r'(I will do this\.?\s*\n?){2,}', '', cleaned, flags=re.IGNORECASE)
    
    # Remove excessive blank lines left behind
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # Remove leading/trailing whitespace from each line
    lines = cleaned.split('\n')
    lines = [line.rstrip() for line in lines]
    cleaned = '\n'.join(lines)
    
    if len(cleaned) < len(text):
        removed = len(text) - len(cleaned)
        print(f" Removed {removed} characters of thinking artifacts")
    
    return cleaned.strip()


def clean_search_deliberations(text: str) -> str:
    """
    Remove search deliberation patterns that leak internal reasoning.
    
    This catches patterns like:
    - 'I'll search for one more "OTA".'
    - '"Current-reuse OTA" in doc_728.'
    - 'Better 5th option: **Telescopic Cascode**'
    - 'I have 5 OTAs now:'
    """
    import re
    
    # Patterns for search deliberations
    deliberation_patterns = [
        # Search statements
        r"I'll search for[^\n]+\n?",
        r"I need to search[^\n]+\n?",
        r"I should search[^\n]+\n?",
        r"Let me search[^\n]+\n?",
        
        # Document reference deliberations (doc_XXX patterns)
        r'"[^"]+"\s+in\s+doc_\d+\.?\s*\n?',  # "X" in doc_123.
        r'doc_\d+\s+mentions[^\n]+\n?',
        r'doc_\d+\s+says[^\n]+\n?',
        r'Doc_\d+[:\s]+[^\n]+\n?',
        r'Actually,?\s+doc_\d+[^\n]+\n?',
        
        # Selection deliberations
        r'Better \d+(?:st|nd|rd|th) option[^\n]+\n?',
        r'I have \d+ (?:OTAs?|candidates?|options?)[^\n]+\n?',
        r'Final list:[^\n]+\n?',
        r'My \d+ candidates[^\n]+\n?',
        
        # Feasibility check deliberations (internal)
        r'(?:^|\n)Final check on "[^"]+":?\n?(?:[^\n]+\n?)*?(?:Passes?|Fails?)[^\n]*\n?',
        r'Stack: \d+ transistors[^\n]+\n?',
        r'Available Swing[^\n]+\n?',
        r'Required Single-Ended[^\n]+\n?',
        r'Required = [^\n]+\n?',
        r'It fails the[^\n]+\n?',
        r'\d+\s*\*\s*\d+mV\s*=[^\n]+\n?',
        
        # Planning statements
        r'I will formulate[^\n]+\n?',
        r'I will produce[^\n]+\n?',
        r'I will generate[^\n]+\n?',
        r'I will stick[^\n]+\n?',
        r'I will treat[^\n]+\n?',
        r'I will reference[^\n]+\n?',
        r'For "[^"]+",? I will cite[^\n]+\n?',
        
        # Self-corrections
        r'Wait,?\s+[^\n]+\n?',
        r'Actually,?\s+[^\n]+\n?',
        r'One last check[^\n]+\n?',
        r'One correction[^\n]+\n?',
    ]
    
    cleaned = text
    for pattern in deliberation_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove excessive blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()


def detect_and_fix_repetition(text: str, min_repeat_len: int = 30, max_repeats: int = 2) -> str:
    """
    Detect and truncate repetitive output from the model.
    
    This helps handle the repetition loop bug in Gemini models where
    the model gets stuck repeating phrases or paragraphs.
    
    Args:
        text: The model's response text
        min_repeat_len: Minimum length of a phrase to consider for repetition detection
        max_repeats: Maximum allowed repetitions before truncation
    
    Returns:
        Cleaned text with repetitive sections removed
    """
    if not text or len(text) < min_repeat_len * 2:
        return text
    
    # First, clean thinking artifacts and search deliberations
    text = clean_thinking_artifacts(text)
    text = clean_search_deliberations(text)
    
    original_len = len(text)
    
    # Method 1: Check for paragraph-level repetition (most common issue)
    # Split by double newlines to get paragraphs
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 5:
        para_counts = {}
        for para in paragraphs:
            stripped = para.strip()
            if len(stripped) >= 50:  # Only count substantial paragraphs
                para_counts[stripped] = para_counts.get(stripped, 0) + 1
        
        # Find repeated paragraphs
        repeated_paras = {para for para, count in para_counts.items() if count > max_repeats}
        
        if repeated_paras:
            cleaned_paras = []
            repeat_count = {}
            for para in paragraphs:
                stripped = para.strip()
                if stripped in repeated_paras:
                    repeat_count[stripped] = repeat_count.get(stripped, 0) + 1
                    if repeat_count[stripped] <= 1:  # Keep only first occurrence
                        cleaned_paras.append(para)
                else:
                    cleaned_paras.append(para)
            
            if len(cleaned_paras) < len(paragraphs):
                print("  Detected paragraph repetition - cleaning output")
                text = '\n\n'.join(cleaned_paras)
    
    # Method 2: Check for line-level repetition (same line repeated many times)
    lines = text.split('\n')
    if len(lines) > 10:
        line_counts = {}
        for line in lines:
            stripped = line.strip()
            if len(stripped) >= min_repeat_len:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1
        
        # Find repeated lines - more aggressive: any line appearing 3+ times
        repeated_lines = {line for line, count in line_counts.items() if count >= 3}
        
        if repeated_lines:
            cleaned_lines = []
            repeat_count = {}
            for line in lines:
                stripped = line.strip()
                if stripped in repeated_lines:
                    repeat_count[stripped] = repeat_count.get(stripped, 0) + 1
                    if repeat_count[stripped] <= 1:  # Keep only FIRST occurrence
                        cleaned_lines.append(line)
                else:
                    cleaned_lines.append(line)
            
            if len(cleaned_lines) < len(lines):
                print("  Detected line repetition - cleaning output")
                text = '\n'.join(cleaned_lines)
    
    # Method 3: Check for phrase-level repetition in the latter half of text
    if len(text) > 1000:
        # Check the last 3000 characters for repetition
        check_region = text[-min(3000, len(text)):]
        
        # Try different pattern lengths
        for pattern_len in [100, 150, 200, 300]:
            if len(check_region) < pattern_len * 3:
                continue
            
            # Take a sample pattern from the end
            pattern = check_region[-pattern_len:]
            count = check_region.count(pattern)
            
            if count > max_repeats:
                # Find where the repetition starts
                first_occurrence = text.find(pattern)
                if first_occurrence > 0:
                    # Keep content up to just after the first occurrence
                    truncate_at = first_occurrence + pattern_len
                    if truncate_at < len(text) - 200:
                        print("  Detected phrase repetition loop - truncating output")
                        text = text[:truncate_at] + "\n\n[... repetitive output truncated ...]"
                        break
    
    if len(text) < original_len:
        removed = original_len - len(text)
        print(f" Removed {removed} characters of repetitive content")
    
    # Method 4: Clean corrupted line fragments (e.g., ")}", stray quotes)
    # These appear when the model's output gets corrupted mid-repetition
    lines = text.split('\n')
    cleaned_lines = []
    prev_line_stripped = ""
    
    for line in lines:
        stripped = line.strip()
        
        # Skip corrupted fragments (just punctuation, stray quotes/braces)
        if re.match(r'^[\"\'\)\}\]\,\.\;]+$', stripped):
            continue
        
        # Skip exact duplicate of previous line
        if stripped == prev_line_stripped and len(stripped) > 20:
            continue
            
        cleaned_lines.append(line)
        if stripped:
            prev_line_stripped = stripped
    
    if len(cleaned_lines) < len(lines):
        text = '\n'.join(cleaned_lines)
        print(f" Cleaned {len(lines) - len(cleaned_lines)} corrupted/duplicate lines")
    
    # Method 5: Truncate at code block end if answer section has duplicates
    # Find the ANSWER section and clean it
    if "ANSWER:" in text or "Answer:" in text:
        # Find where Answer section starts
        answer_start = text.find("Answer:")
        if answer_start == -1:
            answer_start = text.find("ANSWER:")
        
        if answer_start > 0:
            before_answer = text[:answer_start]
            answer_section = text[answer_start:]
            
            # Check if answer has a code block that ends properly
            if "```" in answer_section:
                # Find the last proper ``` closing
                code_blocks = re.findall(r'```python.*?```', answer_section, re.DOTALL)
                if code_blocks:
                    last_good_block = code_blocks[-1]
                    last_block_end = answer_section.rfind(last_good_block) + len(last_good_block)
                    
                    # If there's a lot of content after the code block, check if it's repetition
                    remaining = answer_section[last_block_end:]
                    if len(remaining) > 500:
                        # Check if it's just repeating content from before
                        sample = answer_section[:200]
                        if remaining.count(sample[:50]) > 1:
                            answer_section = answer_section[:last_block_end]
                            text = before_answer + answer_section
                            print(" Truncated repetitive content after code block")
    
    return text


def extract_images_from_query(query: str) -> tuple[str, List[str]]:
    """
    Extract image paths from a query string.
    Supports various formats:
    - Direct path: C:\path\to\image.png
    - Unix path: /path/to/image.png
    - With prefix: with image: path/to/image.png
    
    Returns:
        Tuple of (cleaned_query, list_of_image_paths)
    """
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
    ext_group = r'\.(?:png|jpg|jpeg|gif|bmp|webp)'
    image_paths = []
    cleaned_query = query

    # -- Strategy 1: "path:" prefix  ----------------------------------
    # Matches  "path: C:\...\file.png"  or  "path: :\...\file.png" (missing drive)
    path_prefix_pat = re.compile(
        r'path:\s*([A-Za-z]?:?\\[^"*<>|]+?' + ext_group + r')',
        re.IGNORECASE
    )
    for m in path_prefix_pat.finditer(query):
        raw = m.group(1).rstrip(' .,;:')          # trim trailing punctuation
        # Fix missing drive letter (":\" without "C")
        if raw.startswith(':\\'):
            raw = 'C' + raw
        if os.path.exists(raw):
            image_paths.append(raw)
            cleaned_query = cleaned_query.replace(m.group(0), '[IMAGE PROVIDED]')

    # -- Strategy 2: Windows path (drive letter present, supports spaces) -
    # Greedy up to the extension; rstrip trailing punctuation.
    win_pattern = re.compile(
        r'([A-Za-z]:\\[^"*<>|]+?' + ext_group + r')',
        re.IGNORECASE
    )
    for m in win_pattern.finditer(query):
        raw = m.group(1).rstrip(' .,;:')
        if raw in image_paths:
            continue                               # already captured above
        if os.path.exists(raw):
            image_paths.append(raw)
            cleaned_query = cleaned_query.replace(m.group(1), '[IMAGE PROVIDED]')

    # -- Strategy 3: Unix absolute path -------------------------------
    unix_pattern = re.compile(
        r'(/[^"*<>|]+?' + ext_group + r')',
        re.IGNORECASE
    )
    for m in unix_pattern.finditer(query):
        raw = m.group(1).rstrip(' .,;:')
        if raw in image_paths:
            continue
        if os.path.exists(raw):
            image_paths.append(raw)
            cleaned_query = cleaned_query.replace(m.group(1), '[IMAGE PROVIDED]')

    if image_paths:
        print(f"" Extracted {len(image_paths)} image path(s): {image_paths}")
    else:
        print("  No image paths extracted from query")

    return cleaned_query.strip(), image_paths


def load_image_for_gemini(image_path: str) -> Optional[Any]:
    """
    Load an image and prepare it for Gemini multimodal input.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object or None if failed
    """
    try:
        if not os.path.exists(image_path):
            print(f"  Image not found: {image_path}")
            return None
        
        img = Image.open(image_path)
        
        # Convert to RGB if needed (Gemini prefers RGB)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        print(f"" Loaded image: {os.path.basename(image_path)} ({img.size[0]}x{img.size[1]})")
        return img
        
    except Exception as e:
        print(f"  Error loading image {image_path}: {e}")
        return None


class Agent:
    """
    ReAct-style agent for analog circuit design reasoning.
    Uses the LLM provider abstraction to support Gemini, Claude, etc.
    Implements thought-action-observation loop for complex reasoning.
    Supports multimodal input (text + images).
    
    Switch LLM by setting LLM_PROVIDER in .env (database/embeddings unchanged).
    """
    
    def __init__(self, system: str = ""):
        self.system = system
        self.messages = []
        self.total_cost = 0
        self.total_latency = 0
        self.user_images = []  # Store user-provided images
        
        # Get the LLM provider (Gemini or Claude -- configured in .env / config.py)
        self.provider = get_llm_provider()
        self.chat_history = []  # Provider-agnostic history
        
        print(f"- Using model: {self.provider.model_name()} ({self.provider.provider_name()})")
    
    def set_user_images(self, images: List[Any]):
        """Set user-provided images for multimodal queries."""
        self.user_images = images
    
    
    def __call__(self, message: str, include_images: bool = True) -> str:
        """Send message to agent and get response."""
        result, cost_info = self.execute(message, include_images)
        self.total_cost += cost_info.get('total_cost', 0)
        self.total_latency += cost_info.get('latency', 0)
        return result
    
    def execute(self, message: str, include_images: bool = True) -> tuple:
        """
        Execute a single turn of conversation via the LLM provider.
        
        Args:
            message: Text message to send
            include_images: Whether to include user images in this call (first call only)
        """
        start_time = time.time()
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                return self._execute_via_provider(message, include_images, start_time)
                
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(x in error_str for x in [
                    'disconnected', 'timeout', 'connection', 
                    '500', '503', '504', 'internal', 'server error',
                    'unavailable', 'overloaded', 'resource exhausted',
                    'deadline exceeded', 'temporarily unavailable',
                    'rate_limit', 'rate limit', '429', 'too many requests'
                ])
                
                if is_retryable and attempt < max_retries - 1:
                    wait_time = 5 * (2 ** attempt)
                    print(f"  Server error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                    print(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                print(f"Error: {e}")
                return f"Error: API call failed - {str(e)}", {'latency': time.time() - start_time, 'total_cost': 0}
        
        return "Error: Max retries exceeded", {'latency': time.time() - start_time, 'total_cost': 0}
    
    def _execute_via_provider(self, message: str, include_images: bool, start_time: float) -> tuple:
        """Execute using the LLM provider abstraction (Gemini or Claude)."""
        
        # Collect images for this turn
        images = None
        if include_images and self.user_images:
            print(f"" Sending {len(self.user_images)} image(s) with query")
            images = self.user_images
            self.user_images = []  # Clear after use
        
        # Call provider with history
        response_text, self.chat_history = self.provider.generate_with_history(
            history=self.chat_history,
            new_message=message,
            system_instruction=self.system,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            images=images,
        )
        
        # Handle None/empty response
        if not response_text:
            response_text = "Error: Model returned empty response"
        
        # Detect and fix repetition loops
        response_text = detect_and_fix_repetition(response_text)
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Estimate token counts
        input_tokens = int(len(message.split()) * 1.3) if message else 0
        output_tokens = int(len(response_text.split()) * 1.3) if response_text else 0
        
        cost_info = calculate_cost(input_tokens, output_tokens)
        cost_info['latency'] = latency
        
        return response_text, cost_info


# Available actions for the agent
known_actions = {
    "search_db": search_db,
    "full_document_search": lambda x: full_document_search(*x.split(", ", 1)),
    "load_titles": load_titles
}

action_re = re.compile(r'^Action: (\w+): (.*)$')
system_message = system_message_2.strip()


def create_image_mapping(observation: Any) -> Dict[str, str]:
    """Create mapping between figure references and image paths."""
    image_map = {}
    
    try:
        if isinstance(observation, str):
            results = json.loads(observation)
        else:
            results = observation
        
        for result in results.get('text', []):
            if result.get('content_type') == 'image':
                path = result['item'].get('path')
                if path and os.path.exists(path):
                    filename = os.path.basename(path)
                    base_name = os.path.splitext(filename)[0]
                    
                    if match := re.search(r'image_(\d+)', base_name):
                        img_num = match.group(1)
                        image_map[f"Figure {img_num}"] = filename
                        
    except Exception as e:
        pass
    
    return image_map


def extract_and_save_answer_images(answer_text: str, image_mappings: Dict, 
                                   observation: Dict) -> Optional[str]:
    """Extract and save images mentioned in the answer."""
    try:
        output_dir = "output_images"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        figure_pattern = r'Figure (\d+)'
        references = re.finditer(figure_pattern, answer_text)
        saved_images = []
        
        for ref in references:
            fig_num = ref.group(1)
            fig_ref = f"Figure {fig_num}"
            
            if fig_ref in image_mappings:
                filename = image_mappings[fig_ref]
                
                for result in observation.get('text', []):
                    if (result.get('content_type') == 'image' and 
                        'item' in result and 
                        os.path.basename(result['item'].get('path', '')) == filename):
                        
                        source_path = result['item']['path']
                        dest_path = os.path.join(output_dir, filename)
                        
                        if os.path.exists(source_path):
                            shutil.copy2(source_path, dest_path)
                            saved_images.append((fig_ref, dest_path))
                        break
        
        if saved_images:
            print(f"\nSaved {len(saved_images)} images to {output_dir}")
            return output_dir
        return None
        
    except Exception as e:
        print(f"Error saving images: {e}")
        return None


def rag_presearch_for_sizing(question: str) -> str:
    """
    Pre-query the vector DB to gather relevant circuit design info
    (equations, specs, sizing methodologies) BEFORE generating sizing code.
    This makes Stage 2 outputs more accurate by grounding them in paper data.
    
    Returns a formatted context string with relevant RAG snippets.
    """
    try:
        from search import ContextualVectorDB
        
        db_path = os.path.join(os.path.dirname(__file__), '..', 'finalAgent_db', 'base_db', 'vector_db.pkl')
        if not os.path.exists(db_path):
            print("  \u26a0\ufe0f Vector DB not found, skipping RAG pre-search for sizing")
            return ""
        
        vector_db = ContextualVectorDB("base_db")
        vector_db.load_db()
        
        # Build multiple targeted queries from the user's sizing question
        sizing_queries = []
        
        # Primary: the user's actual question
        sizing_queries.append(question)
        
        # Extract topology keywords for targeted searches
        topology_keywords = [
            'OTA', 'opamp', 'amplifier', 'comparator', 'ADC', 'DAC', 'PLL',
            'LDO', 'bandgap', 'VCO', 'mixer', 'LNA', 'TIA', 'PTAT',
            'telescopic', 'folded cascode', 'two-stage', 'feed-forward',
            'current mirror', 'differential pair', 'common source',
            'inverter-based', 'ring oscillator', 'charge pump',
            'SAR', 'sigma-delta', 'flash', 'pipeline'
        ]
        
        q_lower = question.lower()
        found_topologies = [kw for kw in topology_keywords if kw.lower() in q_lower]
        
        for topo in found_topologies:
            sizing_queries.append(f"{topo} sizing gm/ID methodology design equations")
            sizing_queries.append(f"{topo} transistor dimensions W L current bias specifications")
        
        # If specs mentioned, search for similar designs
        import re
        if re.search(r'gain|GBW|bandwidth|power|swing|noise|CMRR|PSRR', question, re.IGNORECASE):
            sizing_queries.append(f"circuit specifications gain bandwidth power tradeoff sizing")
        
        # Deduplicate and limit
        sizing_queries = list(dict.fromkeys(sizing_queries))[:5]
        
        # Run searches and collect results
        all_snippets = []
        seen_chunks = set()
        
        for sq in sizing_queries:
            try:
                results = vector_db.search(sq, top_k=5)
                for r in results:
                    meta = r.get('metadata', {})
                    content = meta.get('contextualized_content', '') or meta.get('original_content', '')
                    chunk_id = meta.get('chunk_id', content[:50])
                    
                    if chunk_id not in seen_chunks and content and len(content.strip()) > 50:
                        seen_chunks.add(chunk_id)
                        doc_id = meta.get('doc_id', 'unknown')
                        score = r.get('similarity', 0)
                        all_snippets.append({
                            'content': content[:1500],  # Cap per-snippet length
                            'doc_id': doc_id,
                            'score': score
                        })
            except Exception as e:
                print(f"  Search query failed: {e}")
                continue
        
        # Sort by relevance and take top results
        all_snippets.sort(key=lambda x: x['score'], reverse=True)
        top_snippets = all_snippets[:10]
        
        if not top_snippets:
            print("  No relevant RAG results found for sizing")
            return ""
        
        # Format as context block
        rag_context = "\n\n=== RAG CONTEXT FROM RESEARCH PAPERS (use for accurate sizing) ===\n"
        rag_context += "The following excerpts from published papers contain relevant design equations,\n"
        rag_context += "specifications, and sizing approaches. Use these to ground your sizing script.\n\n"
        
        for i, snippet in enumerate(top_snippets, 1):
            rag_context += f"--- Paper Excerpt {i} (source: {snippet['doc_id']}, relevance: {snippet['score']:.3f}) ---\n"
            rag_context += snippet['content'].strip() + "\n\n"
        
        rag_context += "=== END RAG CONTEXT ===\n"
        rag_context += "\nIMPORTANT: Use the above paper data to inform realistic parameter ranges,\n"
        rag_context += "design equations, and specifications in your sizing script.\n"
        
        print(f"  \u2705 RAG pre-search found {len(top_snippets)} relevant paper excerpts for sizing")
        return rag_context
        
    except FileNotFoundError:
        print("  \u26a0\ufe0f Vector DB not built yet, skipping RAG pre-search")
        return ""
    except Exception as e:
        print(f"  \u26a0\ufe0f RAG pre-search failed: {e}")
        return ""


# =============================================================================
# CODE VALIDATION & SELF-REPAIR
# =============================================================================

def _extract_python_code(answer: str) -> list:
    """Extract all ```python ... ``` code blocks from the LLM answer."""
    blocks = re.findall(r'```python\s*\n(.*?)```', answer, re.DOTALL)
    return blocks


def _validate_python_syntax(code: str):
    """
    Check Python code for syntax errors using ast.parse().
    Returns None if valid, or the error message if invalid.
    """
    import ast
    try:
        ast.parse(code)
        return None  # valid
    except SyntaxError as e:
        return f"Line {e.lineno}: {e.msg}"


def _check_bracket_balance(code: str):
    """
    Quick bracket balance check. Returns a description of the first
    imbalance found, or None if balanced.
    """
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    closers = set(pairs.values())
    in_string = None
    prev_char = ''
    
    for i, ch in enumerate(code):
        # Skip string literals (basic handling)
        if ch in ('"', "'") and prev_char != '\\':
            if in_string is None:
                # Check for triple quotes
                if code[i:i+3] in ('"""', "'''"):
                    in_string = code[i:i+3]
                else:
                    in_string = ch
            elif in_string == ch or (len(in_string) == 3 and code[i:i+3] == in_string):
                in_string = None
            prev_char = ch
            continue
        
        if in_string:
            prev_char = ch
            continue
        
        # Skip comments
        if ch == '#':
            # Skip to end of line
            newline = code.find('\n', i)
            if newline == -1:
                break
            prev_char = ch
            continue
        
        if ch in pairs:
            stack.append((ch, i))
        elif ch in closers:
            if not stack:
                return f"Extra closing '{ch}' at position {i}"
            opener, _ = stack.pop()
            if pairs[opener] != ch:
                return f"Mismatched '{opener}' closed with '{ch}' at position {i}"
        prev_char = ch
    
    if stack:
        opener, pos = stack[-1]
        # Find the line number
        line_num = code[:pos].count('\n') + 1
        return f"Unclosed '{opener}' opened at line {line_num}"
    
    return None


def validate_and_repair_code(answer: str, bot, max_retries: int = 2) -> str:
    """
    Extract Python code blocks from the LLM answer, validate syntax,
    and fix if broken.  Strategy:
      1. Try deterministic bracket auto-fix (_auto_fix_brackets)
      2. If still broken, fall back to LLM repair with a focused prompt
    Returns the (possibly repaired) answer.
    """
    code_blocks = _extract_python_code(answer)
    if not code_blocks:
        return answer  # no code to validate
    
    for idx, code in enumerate(code_blocks):
        # Check 1: bracket balance (fast)
        bracket_err = _check_bracket_balance(code)
        # Check 2: full syntax parse
        syntax_err = _validate_python_syntax(code)
        
        error = bracket_err or syntax_err
        if not error:
            continue  # this block is fine
        
        print(f"\n\u26a0\ufe0f Syntax error detected in code block {idx+1}: {error}")
        
        # -- PHASE 1: deterministic bracket fix --------------------------
        fixed_code = _auto_fix_brackets(code)
        if _check_bracket_balance(fixed_code) is None and _validate_python_syntax(fixed_code) is None:
            print(f"   \u2705 Programmatic bracket fix succeeded")
            answer = answer.replace(
                f"```python\n{code}```",
                f"```python\n{fixed_code}\n```"
            )
            continue  # move to next code block
        
        # -- PHASE 2: LLM repair (bracket fix wasn't enough) ------------
        print(f"   Attempting LLM repair (up to {max_retries} retries)...")
        
        # Use the partially-fixed code as starting point
        working_code = fixed_code
        remaining_err = _check_bracket_balance(working_code) or _validate_python_syntax(working_code) or error
        
        for attempt in range(max_retries):
            # Send only ~20 lines around the error for a focused repair
            err_context = _extract_error_context(working_code, remaining_err)
            
            repair_prompt = f"""The Python code you just generated has a syntax error:

ERROR: {remaining_err}

{err_context}

RULES FOR THE FIX:
1. Return the COMPLETE corrected code inside ```python ... ```
2. Fix ONLY the syntax error - do not change the logic
3. Every results.append({{ must end with }})
4. Every df[ filter must end with ]
5. Do NOT add explanation - just output the fixed code block"""

            fixed_response = bot(repair_prompt)
            if fixed_response is None:
                continue
            
            fixed_blocks = _extract_python_code(fixed_response)
            if not fixed_blocks:
                continue
            
            candidate = fixed_blocks[0]
            # Also run programmatic fix on the LLM's attempt
            candidate = _auto_fix_brackets(candidate)
            
            new_bracket_err = _check_bracket_balance(candidate)
            new_syntax_err = _validate_python_syntax(candidate)
            
            if not new_bracket_err and not new_syntax_err:
                print(f"   \u2705 Code repaired on attempt {attempt+1}")
                answer = answer.replace(
                    f"```python\n{code}```",
                    f"```python\n{candidate}\n```"
                )
                working_code = None  # signal success
                break
            else:
                remaining_err = new_bracket_err or new_syntax_err
                working_code = candidate
                print(f"   \u274c Attempt {attempt+1} still has errors: {remaining_err}")
        else:
            # All LLM retries failed -- use best effort (programmatic fix even if imperfect)
            if _validate_python_syntax(fixed_code) is None:
                # Bracket-fixed code at least parses
                print(f"   \u26a0\ufe0f LLM repair failed, using programmatic fix (parses OK)")
                answer = answer.replace(
                    f"```python\n{code}```",
                    f"```python\n{fixed_code}\n```"
                )
            else:
                print(f"   \u26a0\ufe0f Could not auto-repair code block {idx+1} after {max_retries} attempts")
    
    return answer


def _auto_fix_brackets(code: str) -> str:
    """
    Deterministically fix unclosed brackets by inserting the missing
    closer(s) at the correct indentation level.
    
    Handles patterns like:
      results.append({          df[
          'key': val,               (condition1) &
          'key2': val2              (condition2)
                         <-- missing })           <-- missing ]
      except:                   if ...:
    
    Algorithm:
      1. Run bracket check to find "Unclosed '{' opened at line N"
      2. Analyse all unclosed brackets on line N to build the closing sequence
      3. Find insertion point: first non-blank line after N with indent <= line N
      4. Insert the closing sequence
      5. Repeat until balanced (max 10 iterations)
    """
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for _ in range(10):
        err = _check_bracket_balance(code)
        if err is None:
            break  # balanced
        
        m = re.match(r"Unclosed '(.)' opened at line (\d+)", str(err))
        if not m:
            break  # can't parse error
        
        bracket_char = m.group(1)
        line_num = int(m.group(2))  # 1-based
        
        lines = code.split('\n')
        if line_num < 1 or line_num > len(lines):
            break
        
        open_line = lines[line_num - 1]
        open_indent = len(open_line) - len(open_line.lstrip())
        
        # -- Determine closing sequence from unclosed brackets on this line --
        line_stack = []
        in_str = None
        for ci, ch in enumerate(open_line):
            # Basic string skipping (handles 'x' and "x", not triple-quotes on one line)
            if ch in ("'", '"'):
                if in_str is None:
                    in_str = ch
                elif in_str == ch and (ci == 0 or open_line[ci - 1] != '\\'):
                    in_str = None
                continue
            if in_str:
                continue
            if ch == '#':
                break  # rest of line is comment
            if ch in pairs:
                line_stack.append(ch)
            elif ch in pairs.values():
                if line_stack and pairs.get(line_stack[-1]) == ch:
                    line_stack.pop()
        
        if line_stack:
            close_seq = ''.join(pairs[b] for b in reversed(line_stack))
        else:
            # Bracket opened on this line but scanner missed it -- close reported one
            close_seq = pairs.get(bracket_char, '')
        
        if not close_seq:
            break
        
        # -- Find insertion point --
        # First non-blank line AFTER the opening with indent <= open_indent
        insert_pos = len(lines)  # default: end of code
        for i in range(line_num, len(lines)):  # line_num is 1-based -> index starts one line after
            stripped = lines[i].strip()
            if not stripped:
                continue
            curr_indent = len(lines[i]) - len(lines[i].lstrip())
            if curr_indent <= open_indent:
                insert_pos = i
                break
        
        # Insert the closing bracket(s) at the opening line's indentation
        close_line = ' ' * open_indent + close_seq
        lines.insert(insert_pos, close_line)
        code = '\n'.join(lines)
        
        print(f"   \U0001f527 Auto-inserted '{close_seq}' at line {insert_pos + 1}")
    
    return code


def _extract_error_context(code: str, error: str) -> str:
    """
    Extract ~20 lines around the error location for a focused LLM repair prompt,
    instead of sending the entire 130-line code block.
    """
    # Try to parse line number from error
    m = re.search(r'line (\d+)', str(error), re.IGNORECASE)
    if not m:
        # Can't locate -- send full code (truncated)
        return f"Here is the code:\n```python\n{code[:3000]}\n```"
    
    err_line = int(m.group(1))
    lines = code.split('\n')
    start = max(0, err_line - 10)
    end = min(len(lines), err_line + 10)
    
    snippet = '\n'.join(f"{i+1:4d} | {lines[i]}" for i in range(start, end))
    return f"Here is the code around the error (lines {start+1}-{end}):\n```\n{snippet}\n```\n\nFull code length: {len(lines)} lines."


def query(question: str, max_turns: int = 10, system_prompt: str = None, existing_bot: 'Agent' = None) -> Dict[str, Any]:
    """
    Main query function implementing ReAct agent loop.
    Supports multimodal queries with images.
    Supports two-stage workflow with custom system prompts.
    Supports conversation continuation via existing_bot parameter.
    
    Args:
        question: User's question about analog circuits (may contain image paths)
        max_turns: Maximum agent turns
        system_prompt: Optional custom system prompt (for Stage 1/2 modes)
        existing_bot: Optional existing Agent instance to reuse (enables conversation continuation)
        
    Returns:
        Dictionary with answer, images, metrics, and the bot instance for continuation
    """
    i = 0
    
    # Use provided system prompt or default
    active_system_message = system_prompt if system_prompt else system_message
    
    # Extract images from query
    cleaned_query, image_paths = extract_images_from_query(question)
    
    # Load images for multimodal input
    user_images = []
    if image_paths:
        print(f"\n" Found {len(image_paths)} image(s) in query:")
        for path in image_paths:
            img = load_image_for_gemini(path)
            if img:
                user_images.append(img)
    
    # Reuse existing bot (conversation continuation) or create new one
    if existing_bot is not None:
        bot = existing_bot
        print(f"' Continuing conversation (history: {len(bot.chat_history)} messages)")
    else:
        bot = Agent(active_system_message)
    
    # Set user images for multimodal processing
    if user_images:
        bot.set_user_images(user_images)
        # Add image context to the query
        image_context = (
            f"\n\n[CIRCUIT IMAGE PROVIDED -- {len(user_images)} image(s)]\n"
            "IMPORTANT: Analyse the attached circuit schematic image FIRST.\n"
            "Identify the EXACT topology, label every transistor (M1, M2, ...), "
            "determine each transistor's role (input pair, active load, current mirror, "
            "cascode, tail source, output stage, etc.), and design/size THIS topology.\n"
            "Do NOT substitute a different topology. The image is the ground truth."
        )
        next_prompt = cleaned_query + image_context
    else:
        next_prompt = question
    
    image_mappings = {}
    last_observation = None
    final_answer = None
    
    print(f"\n{'='*60}")
    print(f"Query: {question}")
    print(f"{'='*60}\n")
    
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        
        # Handle None or error responses
        if result is None:
            print("  Model returned empty response, retrying...")
            continue
        
        if result.startswith("Error:"):
            print(result)
            return {
                "answer": result,
                "image_paths": None,
                "total_cost": bot.total_cost,
                "total_latency": bot.total_latency,
                "turns": i
            }
        
        final_answer = result
        print(result)
        
        # Check if final answer (but encourage search for paper-specific questions)
        if "Action:" not in result:
            # Determine if the answer is substantial enough to accept without search
            answer_is_substantial = (
                len(result) > 500 or  # Long detailed answer
                "```" in result or    # Contains code block
                "\\boxed{" in result or  # Contains boxed answer (LaTeX)
                any(kw in result.lower() for kw in ['equation', 'formula', 'calculate', 'result:', 'answer:'])
            )
            
            # Check if question explicitly asks for papers/references
            needs_paper_search = any(kw in question.lower() for kw in [
                'paper', 'reference', 'cite', 'publication', 'journal', 'jssc',
                'what paper', 'which paper', 'show me a circuit from'
            ])
            
            # If this is the first turn, no search was done, and we need papers
            if i == 1 and last_observation is None and needs_paper_search and not answer_is_substantial:
                print("\n  Agent tried to answer without searching. Reminding to use database...")
                next_prompt = """You skipped the database search! 

IMPORTANT: You MUST search the database first before answering.
Use: Action: search_db: [relevant query]

The database contains 400+ JSSC research papers with expert knowledge on analog circuit design.
Please search for relevant papers, then provide your answer based on the search results.

Original question: """ + question
                continue
            
            # Accept the answer - either substantial or doesn't need paper search
            image_dir = None
            topology_images = None
            
            if last_observation:
                image_dir = extract_and_save_answer_images(result, image_mappings, last_observation)
                # Extract topology images from search results
                topology_images = extract_topology_images(last_observation, final_answer)
            
            # Format with AnuRAG header for topology analysis responses
            sources_found = []
            seen_doc_ids = set()
            if last_observation and isinstance(last_observation, dict):
                # Extract unique source document IDs and paper titles from search results
                for item in last_observation.get('text', []):
                    if isinstance(item, dict) and 'item' in item:
                        doc_id = item['item'].get('doc_id', '')
                        paper_title = item['item'].get('paper_title', '')
                        if doc_id and doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            if paper_title:
                                # Truncate long titles for the header
                                short_title = paper_title[:60] + '...' if len(paper_title) > 60 else paper_title
                                sources_found.append(f"{short_title} [{doc_id}]")
                            else:
                                sources_found.append(doc_id)
            
            formatted_answer = format_anurag_output(final_answer, sources_found, topology_images)
            
            # === CODE VALIDATION: check generated Python for syntax errors ===
            if _extract_python_code(formatted_answer):
                formatted_answer = validate_and_repair_code(formatted_answer, bot)
            
            return {
                "answer": formatted_answer,
                "image_paths": topology_images.get('output_dir') if topology_images else image_dir,
                "topology_images": topology_images,
                "total_cost": bot.total_cost,
                "total_latency": bot.total_latency,
                "turns": i,
                "bot": bot  # Return bot for conversation continuation
            }
        
        # Parse actions
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        
        if actions:
            action, action_input = actions[0].groups()
            
            if action not in known_actions:
                next_prompt = f"Observation: Unknown action '{action}'. Available: {list(known_actions.keys())}"
                continue
            
            print(f"\n  -> Running {action}({action_input})")
            
            try:
                if action == "load_titles":
                    observation = known_actions[action]()
                elif action_input.startswith('(') and action_input.endswith(')'):
                    parsed_input = eval(action_input)
                    observation = known_actions[action](parsed_input)
                else:
                    observation = known_actions[action](action_input)
                
                if isinstance(observation, dict):
                    last_observation = observation
                    image_mappings.update(create_image_mapping(observation))
                
                obs_str = json.dumps(observation, indent=2, default=str)[:10000] if isinstance(observation, dict) else str(observation)[:10000]
                next_prompt = f"Observation: {obs_str}"
                
                if image_mappings:
                    next_prompt += f"\n\nImage mappings: {json.dumps(image_mappings)}"
                    
            except ConnectionError as e:
                # Database connection failed - fall back to LLM knowledge
                print(f"\n  Database connection failed: {e}")
                print(" Falling back to LLM internal knowledge (no RAG search)")
                next_prompt = f"""Observation: DATABASE OFFLINE - Connection failed.

The database is currently unavailable. Please proceed using your internal knowledge only.
You are an expert analog circuit designer. Answer the question using your training knowledge.

NOTE: Since you cannot search the paper database, your answer will be based on general knowledge only.
Provide the best answer you can without RAG support.

Original question: {question}"""
                # Mark that we've "searched" so we don't loop forever
                last_observation = {"status": "fallback_mode", "reason": str(e)}
                    
            except Exception as e:
                next_prompt = f"Observation: Error - {str(e)}"
    
    # Extract topology images from last observation if available
    topology_images = None
    if last_observation and isinstance(last_observation, dict):
        topology_images = extract_topology_images(last_observation, final_answer or "")
    
    # Format final answer with AnuRAG header for topology analysis
    formatted_answer = format_anurag_output(final_answer or "Max turns reached.", [], topology_images)
    
    return {
        "answer": formatted_answer,
        "image_paths": topology_images.get('output_dir') if topology_images else None,
        "topology_images": topology_images,
        "total_cost": bot.total_cost,
        "total_latency": bot.total_latency,
        "turns": max_turns,
        "bot": bot  # Return bot for conversation continuation
    }


def process_papers(paper_path: str) -> None:
    """Process PDF papers into the database."""
    import glob
    import asyncio
    
    if os.path.isfile(paper_path):
        pdf_files = [paper_path]
    elif os.path.isdir(paper_path):
        pdf_files = glob.glob(os.path.join(paper_path, "*.pdf"))
    else:
        print(f"Error: {paper_path} is not a valid file or directory")
        return
    
    if not pdf_files:
        print("No PDF files found!")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    asyncio.run(pdf2json_chunked(pdf_files))


# Default paper path - change this to your papers location
DEFAULT_PAPER_PATH = "./papers"


def build_database() -> None:
    """Build the vector database from processed documents."""
    import glob
    import asyncio
    
    documents_path = "../finalAgent_db/documents.json"
    
    # Check if documents.json exists
    if not os.path.exists(documents_path):
        print("" documents.json not found. Processing papers first...")
        
        # Use default path or prompt user
        paper_path = DEFAULT_PAPER_PATH
        
        if not os.path.exists(paper_path):
            print(f" Default paper path not found: {paper_path}")
            print("Please run: python main.py --process_papers /path/to/your/pdfs/")
            return
        
        print(f"" Using paper path: {paper_path}")
        process_papers(paper_path)
        
        # Verify documents.json was created
        if not os.path.exists(documents_path):
            print(" Failed to create documents.json")
            return
        
        print("... Papers processed successfully!")
    
    print("\n" Building vector database...")
    search_db(query=None, load_data=True)


def _read_user_input(prompt: str = "> ") -> str:
    """Read user input, supporting multi-line input terminated by '---'."""
    first_line = input(prompt).strip()
    
    if first_line == '---':
        return ''
    
    lines = [first_line]
    if first_line:
        print("  (Continue typing, or enter '---' to submit)")
        while True:
            try:
                line = input("  ")
                if line.strip() == '---':
                    break
                lines.append(line)
            except EOFError:
                break
    
    return '\n'.join(lines).strip()


def interactive_mode() -> None:
    """Run in interactive question-answering mode with two-stage support and conversation continuation."""
    print("\n" + "="*60)
    print("AnuRAG - Analog Design Framework with RAG")
    print(f"LLM Provider: {LLM_PROVIDER.upper()} ({get_active_chat_model()})")
    print("Two-Stage Workflow: Topology Selection -> Sizing")
    print("="*60)
    print("\n" MODES:")
    print("  [1] Topology Selection - Search RAG for circuit architectures")
    print("  [2] Sizing - Generate sizing script with Pareto optimization")
    print("  [0] General - Free-form queries (default)")
    print("\n" COMMANDS:")
    print("  'mode 1' or 'stage 1' - Switch to Topology Selection mode")
    print("  'mode 2' or 'stage 2' - Switch to Sizing mode")
    print("  'mode 0' or 'general' - Switch to General mode")
    print("  'quit' or 'exit' - Exit the program")
    print("  '---' on a new line - End multi-line input and submit")
    print("  'file:<path>' - Load question from a text file")
    print("\n" AFTER EACH ANSWER:")
    print("  [c] Continue - follow-up on same conversation (context preserved)")
    print("  [n] New query - fresh start in same mode")
    print("  [s] Switch mode - change to a different mode")
    print("  [q] Quit")
    print("\n")
    
    current_mode = 0  # 0=General, 1=Topology, 2=Sizing
    mode_names = {0: "General", 1: "Topology Selection", 2: "Sizing"}
    last_topology_result = None  # Store result from Stage 1 for Stage 2
    active_bot = None  # Persistent bot for conversation continuation
    session_cost = 0.0  # Track cumulative cost across conversation
    session_turns = 0
    
    while True:
        try:
            mode_indicator = f"[{mode_names[current_mode]}]"
            
            if active_bot:
                print(f"\n{mode_indicator} ' Conversation active ({len(active_bot.chat_history)} msgs)")
                print(f"  Follow-up question (or 'new' for fresh query, 'mode X' to switch):")
            else:
                print(f"\n{mode_indicator} Question (type '---' on new line to submit):")
            
            # Read first line
            first_line = input("> ").strip()
            
            if first_line.lower() in ['quit', 'exit', 'q']:
                if session_cost > 0:
                    print(f"\n" Session total: ${session_cost:.6f} across {session_turns} turns")
                print("Goodbye!")
                break
            
            # Check for mode switch commands
            if first_line.lower() in ['mode 1', 'stage 1', 'topology', '1'] and not active_bot:
                current_mode = 1
                active_bot = None  # Reset conversation on mode switch
                print("... Switched to TOPOLOGY SELECTION mode")
                print("   Ask: 'Design OTA for 12-bit ADC, VDD=1.2V, CL=2pF, Gain>=70dB, GBW>=500MHz'")
                continue
            elif first_line.lower() in ['mode 2', 'stage 2', 'sizing', '2'] and not active_bot:
                current_mode = 2
                active_bot = None
                print("... Switched to SIZING mode")
                if last_topology_result:
                    print("   Previous topology selection available - reference it or specify new topology")
                print("   Ask: 'Size Two-Stage OTA with VDD=1.2V, CL=2pF, GBW=500MHz using gm/ID=15'")
                continue
            elif first_line.lower() in ['mode 0', 'general', '0'] and not active_bot:
                current_mode = 0
                active_bot = None
                print("... Switched to GENERAL mode")
                continue
            
            # Handle 'new' command to start fresh query (discard conversation)
            if first_line.lower() in ['new', 'n', 'new query']:
                active_bot = None
                print("" Starting fresh conversation.")
                continue
            
            # Handle 's' / 'switch' to switch mode and reset conversation
            if first_line.lower() in ['s', 'switch', 'switch mode']:
                active_bot = None
                print("\nSwitch to which mode?")
                print("  [0] General  [1] Topology Selection  [2] Sizing")
                mode_choice = input("Mode> ").strip()
                if mode_choice in ['0', 'general']:
                    current_mode = 0
                elif mode_choice in ['1', 'topology']:
                    current_mode = 1
                elif mode_choice in ['2', 'sizing']:
                    current_mode = 2
                else:
                    print("Invalid choice, staying in current mode.")
                    continue
                print(f"... Switched to {mode_names[current_mode]} mode (fresh conversation)")
                continue
            
            if not first_line:
                continue
            
            # Check if it's a file reference
            if first_line.startswith('file:'):
                file_path = first_line[5:].strip()
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        question = f.read().strip()
                    print(f"Loaded question from: {file_path}")
                except Exception as e:
                    print(f"Error reading file: {e}")
                    continue
            else:
                # Check if user wants multi-line input
                lines = [first_line]
                
                # If line doesn't end with '---', check for more input
                if first_line != '---':
                    print("  (Continue typing, or enter '---' to submit)")
                    while True:
                        try:
                            line = input("  ")
                            if line.strip() == '---':
                                break
                            lines.append(line)
                        except EOFError:
                            break
                
                question = '\n'.join(lines).strip()
            
            if not question or question == '---':
                continue
            
            print(f"\n" Submitting question ({len(question)} chars)...")
            
            # Select system message based on mode and run query
            if current_mode == 1:
                result = query(
                    question, max_turns=10,
                    system_prompt=system_message_stage1,
                    existing_bot=active_bot
                )
                last_topology_result = result  # Store for Stage 2
            elif current_mode == 2:
                # === RAG PRE-SEARCH: Query vector DB for relevant design info ===
                print("\n\U0001f50d Running RAG pre-search for sizing context...")
                rag_context = rag_presearch_for_sizing(question)
                
                # === BUILD LUT-AWARE STAGE 2 PROMPT ===
                try:
                    lut_info = get_lut_info(LUT_NMOS_PATH, LUT_PMOS_PATH)
                    stage2_prompt = _build_stage2_prompt(lut_info)
                    print("... LUT introspection loaded into Stage 2 prompt")
                except Exception as e:
                    print(f"  Could not introspect LUT files ({e}), using static reference")
                    stage2_prompt = system_message_stage2  # falls back to static
                
                # Prepend context from Stage 1 if available
                if last_topology_result and 'answer' in last_topology_result:
                    context_hint = f"\n\n[Context from Stage 1 Topology Selection - use if relevant]\n"
                    question = question + context_hint
                
                # Inject RAG context into the question
                if rag_context:
                    question = question + "\n" + rag_context
                
                result = query(
                    question, max_turns=10,
                    system_prompt=stage2_prompt,
                    existing_bot=active_bot
                )
            else:
                result = query(question, existing_bot=active_bot)
            
            # Display answer
            print("\n" + "-"*40)
            print(f"ANSWER ({mode_names[current_mode]} Mode):")
            print("-"*40)
            print(result['answer'])
            print(f"\nMetrics: {result['turns']} turns, ${result['total_cost']:.6f} cost, {result['total_latency']:.2f}s latency")
            
            session_cost += result.get('total_cost', 0)
            session_turns += result.get('turns', 0)
            
            if result.get('image_paths'):
                print(f"Images saved to: {result['image_paths']}")
            
            # Suggest next step for Stage 1
            if current_mode == 1 and 'answer' in result and 'PASS' in result['answer']:
                print("\n' Tip: Type 'stage 2' to proceed to sizing with your selected topology")
            
            # === POST-ANSWER: Continue / New / Switch / Quit ===
            active_bot = result.get('bot')  # Preserve bot for potential continuation
            
            print("\n" + "-"*40)
            print("What next?")
            print("  [c] Continue - follow-up on this conversation")
            print("  [n] New query - start fresh in same mode")
            print("  [s] Switch mode - change mode (resets conversation)")
            print("  [q] Quit")
            print("-"*40)
            
            choice = input("Choice [c/n/s/q]: ").strip().lower()
            
            if choice in ['q', 'quit', 'exit']:
                if session_cost > 0:
                    print(f"\n" Session total: ${session_cost:.6f} across {session_turns} turns")
                print("Goodbye!")
                break
            elif choice in ['n', 'new', 'new query']:
                active_bot = None
                print("" Starting fresh conversation.")
            elif choice in ['s', 'switch', 'switch mode']:
                active_bot = None
                print("\nSwitch to which mode?")
                print("  [0] General  [1] Topology Selection  [2] Sizing")
                mode_choice = input("Mode> ").strip()
                if mode_choice in ['0', 'general']:
                    current_mode = 0
                elif mode_choice in ['1', 'topology']:
                    current_mode = 1
                elif mode_choice in ['2', 'sizing']:
                    current_mode = 2
                else:
                    print("Invalid choice, staying in current mode.")
                print(f"... Switched to {mode_names[current_mode]} mode (fresh conversation)")
            elif choice in ['c', 'continue', '']:
                # Keep active_bot -- it already has the conversation history
                print("' Continuing conversation. Type your follow-up:")
            else:
                # Default: treat any other input as a follow-up question directly
                # (in case user just types their next question instead of pressing 'c')
                if len(choice) > 3:
                    # They typed a question directly -- process it
                    question = choice
                    print(f"\n" Submitting follow-up ({len(question)} chars)...")
                    if current_mode == 1:
                        result = query(question, max_turns=10, system_prompt=system_message_stage1, existing_bot=active_bot)
                        last_topology_result = result
                    elif current_mode == 2:
                        rag_context = rag_presearch_for_sizing(question)
                        try:
                            lut_info = get_lut_info(LUT_NMOS_PATH, LUT_PMOS_PATH)
                            stage2_prompt = _build_stage2_prompt(lut_info)
                        except Exception:
                            stage2_prompt = system_message_stage2
                        if rag_context:
                            question = question + "\n" + rag_context
                        result = query(question, max_turns=10, system_prompt=stage2_prompt, existing_bot=active_bot)
                    else:
                        result = query(question, existing_bot=active_bot)
                    
                    print("\n" + "-"*40)
                    print(f"ANSWER ({mode_names[current_mode]} Mode):")
                    print("-"*40)
                    print(result['answer'])
                    print(f"\nMetrics: {result['turns']} turns, ${result['total_cost']:.6f} cost, {result['total_latency']:.2f}s latency")
                    session_cost += result.get('total_cost', 0)
                    session_turns += result.get('turns', 0)
                    active_bot = result.get('bot')
                else:
                    print("Continuing conversation. Type your follow-up:")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='AnuRAG: Multimodal AI for Analog Circuit Design',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process PDF papers
  python main.py --process_papers /path/to/papers/
  
  # Build the vector database
  python main.py --build_db
  
  # Query the system
  python main.py --query "What is the lowest power bandgap reference circuit?"
  
  # Interactive mode
  python main.py --interactive
  
  # Full pipeline: process papers, build DB, and query
  python main.py --process_papers /path/to/papers/ --build_db --query "Show me a PTAT circuit"
        """
    )
    
    parser.add_argument('--process_papers', type=str, 
                        help='Path to PDF file or directory to process')
    parser.add_argument('--build_db', action='store_true',
                        help='Build/rebuild the vector database')
    parser.add_argument('--query', type=str,
                        help='Query to ask the system')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--max_turns', type=int, default=10,
                        help='Maximum agent turns per query (default: 10)')
    
    args = parser.parse_args()
    
    # Check if API key is set based on provider
    llm_prov = LLM_PROVIDER.lower()
    if llm_prov in ("claude", "anthropic"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY not found. Set it in your .env file.")
            print("Get your key from: https://console.anthropic.com/settings/keys")
            sys.exit(1)
    else:
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            print("Error: Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")
            sys.exit(1)
    
    # Process papers if specified
    if args.process_papers:
        print(f"\n" Processing papers from: {args.process_papers}")
        process_papers(args.process_papers)
    
    # Build database if specified
    if args.build_db:
        print("\n" Building vector database...")
        build_database()
    
    # Handle query or interactive mode
    if args.query:
        print(f"\n" Querying: {args.query}")
        result = query(args.query, max_turns=args.max_turns)
        
        print("\n" + "="*60)
        print("FINAL ANSWER")
        print("="*60)
        print(result['answer'])
        print(f"\nMetrics: {result['turns']} turns, ${result['total_cost']:.6f}, {result['total_latency']:.2f}s")
        
    elif args.interactive:
        interactive_mode()
        
    elif not args.process_papers and not args.build_db:
        # No arguments provided, show help
        parser.print_help()
        print("\n' Quick start:")
        print("   1. Set GOOGLE_API_KEY in .env file")
        print("   2. Run: python main.py --process_papers /path/to/papers/")
        print("   3. Run: python main.py --build_db")
        print("   4. Run: python main.py --interactive")


if __name__ == "__main__":
    main()
