"""
AnuRAG: Full Context Module
Uses the full content of the PDF to answer questions using Google Gemini
Uses the new google-genai package (REST API) for better firewall compatibility
"""

import os
import sys
import json
import time
import argparse
from typing import Optional
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

# Import centralized configuration
from config import GEMINI_CHAT_MODEL, GEMINI_VISION_MODEL

# Try the new google-genai package first (REST API, better firewall compatibility)
GENAI_AVAILABLE = False
client = None

try:
    from google import genai
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        client = genai.Client(api_key=api_key)
        GENAI_AVAILABLE = True
    else:
        print("Warning: GOOGLE_API_KEY or GEMINI_API_KEY not found in environment")
except ImportError:
    # Fallback to the old google-generativeai package (gRPC)
    try:
        import google.generativeai as genai_old
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            genai_old.configure(api_key=api_key)
            GENAI_AVAILABLE = True
        print("Using legacy google-generativeai package")
    except ImportError:
        print("Warning: google-genai not installed. Run: pip install google-genai")


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """Extract all text from a PDF file."""
    try:
        text = ""
        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip() if text.strip() else None
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None


def cag_gemini(context: str, question: str) -> str:
    """
    Context-Augmented Generation using Google Gemini.
    
    Args:
        context: The full document text
        question: The user's question
        
    Returns:
        The model's response
    """
    if not GENAI_AVAILABLE:
        return "Error: google-genai library not available"
    
    if context is None:
        return "Error: Could not read PDF content"
    
    max_retries = 3
    retry_delay = 2  # Start with 2 seconds
    
    # Create the prompt
    prompt = f"""You are an expert in analog and mixed-signal circuit design.
            
Use the following document context to answer the question. Be precise and technical.

DOCUMENT CONTEXT:
{context[:100000]}

USER QUESTION:
{question}

Provide a detailed, accurate answer based on the document content. If the information is not in the document, say so clearly.
"""
    
    for attempt in range(max_retries):
        try:
            if client:
                # Use the new google-genai package (REST API)
                response = client.models.generate_content(
                    model=GEMINI_CHAT_MODEL,
                    contents=prompt
                )
                return response.text
            else:
                # Fallback to old package
                import google.generativeai as genai_old
                model = genai_old.GenerativeModel(GEMINI_CHAT_MODEL)
                response = model.generate_content(prompt)
                return response.text
            
        except Exception as e:
            error_str = str(e).lower()
            if ("rate" in error_str or "quota" in error_str) and attempt < max_retries - 1:
                print(f"Rate limit hit, waiting {retry_delay} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Error calling Gemini API: {e}")
                return f"Error: Could not process document - {str(e)}"
    
    return "Error: Max retries reached"


def analyze_with_image(image_path: str, question: str, context: Optional[str] = None) -> str:
    """
    Analyze an image using Gemini's vision capabilities.
    
    Args:
        image_path: Path to the image file
        question: Question about the image
        context: Optional text context from the document
        
    Returns:
        Analysis of the image
    """
    if not GENAI_AVAILABLE:
        return "Error: google-genai library not available"
    
    try:
        import PIL.Image
        import base64
        
        # Load and encode the image
        image = PIL.Image.open(image_path)
        
        # Create prompt
        prompt = f"""Analyze this image from a technical research paper.

{f'DOCUMENT CONTEXT: {context[:5000]}' if context else ''}

QUESTION: {question}

Provide a detailed technical analysis. If this is:
- A circuit schematic: Describe the topology, components, and operation
- A graph: Explain the axes, trends, and performance metrics
- An equation: Extract and explain the mathematical formula
- A block diagram: Describe the system architecture
- A table: Summarize the key data points
"""
        
        if client:
            # Use new google-genai package with image
            # Read image as bytes
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
            
            response = client.models.generate_content(
                model=GEMINI_VISION_MODEL,
                contents=[
                    prompt,
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(image_bytes).decode()
                        }
                    }
                ]
            )
            return response.text
        else:
            # Fallback to old package
            import google.generativeai as genai_old
            model = genai_old.GenerativeModel(GEMINI_VISION_MODEL)
            response = model.generate_content([prompt, image])
            return response.text
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


def main(input_str: str, pdf_path: str) -> str:
    """
    Main function to answer questions using full document context.
    
    Args:
        input_str: The question to answer
        pdf_path: Path to the PDF document
        
    Returns:
        The answer from Gemini
    """
    context = extract_text_from_pdf(pdf_path)
    if context is None:
        return f"Error: Could not extract text from {pdf_path}"
    
    return cag_gemini(context, input_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Answer questions using full document context')
    parser.add_argument('--question', type=str, default="What is the title of the document?",
                        help='Question to ask about the document')
    parser.add_argument('--pdf_path', type=str, required=True,
                        help='Path to the PDF document')
    
    args = parser.parse_args()
    
    result = main(args.question, args.pdf_path)
    print(f"\nAnswer:\n{result}")
