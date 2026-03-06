"""
AnuRAG: PDF Processing Module
Extracts text, images, equations, and tables from research papers
"""

import json
import uuid
import argparse
import os
from typing import List, Dict, Any, Tuple, Optional
import shutil
import glob
import re
import nltk
from nltk.tokenize import sent_tokenize
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import asyncio
import PyPDF2
import fitz  # PyMuPDF for better image extraction

try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("Warning: unstructured library not available, using PyPDF2 fallback")

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


class PDFProcessor:
    """Processes PDF files to extract text, images, equations, and tables."""
    
    def __init__(self, image_dir: str):
        self.image_dir = image_dir
        os.makedirs(image_dir, exist_ok=True)

    def is_meaningful(self, text: str) -> bool:
        """Filter out meaningless text (e.g., single characters, whitespace only)."""
        cleaned_text = re.sub(r'\s', '', text)
        return len(cleaned_text) > 1 and any(char.isalnum() for char in cleaned_text)

    def create_chunks(self, text: str, max_chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """Create overlapping chunks from text for better retrieval.
        
        OPTIMIZED: Increased chunk size from 1000 to 2000 to reduce total chunks.
        This reduces embedding costs while maintaining good retrieval quality.
        """
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback to simple splitting if sent_tokenize fails
            sentences = text.split('. ')
            
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                
                # Add overlap from previous chunk
                if chunks:
                    overlap_text = " ".join(chunks[-1].split()[-overlap//10:])
                    current_chunk = overlap_text + " " + current_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 as primary/fallback method."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                return text.strip()
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
            return ""

    def extract_images_pymupdf(self, pdf_path: str, doc_id: int, start_number: int) -> Tuple[List[Dict], int]:
        """Extract images from PDF using PyMuPDF."""
        images = []
        image_count = 0
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Skip very small images (likely icons/artifacts)
                        if len(image_bytes) < 5000:
                            continue
                        
                        image_num = start_number + image_count
                        new_name = f'image_{image_num}.{image_ext}'
                        new_path = os.path.join(self.image_dir, new_name)
                        json_path = os.path.join("../finalAgent_db/images/", new_name)
                        
                        with open(new_path, 'wb') as img_file:
                            img_file.write(image_bytes)
                        
                        images.append({
                            "image_id": f"doc_{doc_id}_image_{image_num}",
                            "path": json_path,
                            "page": page_num + 1
                        })
                        image_count += 1
                        
                    except Exception as e:
                        print(f"Error extracting image {img_index} from page {page_num}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            print(f"PyMuPDF image extraction failed: {e}")
            
        return images, image_count

    def detect_equations(self, text: str) -> List[Dict[str, Any]]:
        """Detect potential equations in text."""
        equations = []
        
        # Common equation patterns
        patterns = [
            r'\$\$.*?\$\$',  # LaTeX display math
            r'\$.*?\$',  # LaTeX inline math
            r'\\begin\{equation\}.*?\\end\{equation\}',
            r'[A-Za-z]+\s*=\s*[^,.\n]+',  # Simple equation pattern
            r'V_[A-Za-z]+\s*=',  # Voltage equations
            r'I_[A-Za-z]+\s*=',  # Current equations
            r'P_[A-Za-z]+\s*=',  # Power equations
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                equations.append({
                    "type": "equation",
                    "content": match.strip()
                })
        
        return equations

    async def process_pdf(self, pdf_path: str, doc_id: int, start_number: int, 
                         progress_task=None, progress=None) -> Tuple[Dict[str, Any], int]:
        """Process a single PDF file and extract all content."""
        content = ""
        elements = []
        
        # Try unstructured library first for better text extraction
        if UNSTRUCTURED_AVAILABLE:
            try:
                elements = partition_pdf(
                    filename=pdf_path,
                    extract_images_in_pdf=False,  # We'll use PyMuPDF for images
                    infer_table_structure=True,
                    strategy="fast"
                )
                print(f"Unstructured extracted {len(elements)} elements from {os.path.basename(pdf_path)}")
            except Exception as e:
                print(f"Unstructured failed: {e}")
        
        # Extract text from unstructured elements
        if elements:
            for element in elements:
                if hasattr(element, 'text'):
                    element_text = element.text.strip()
                    if element_text and self.is_meaningful(element_text):
                        content += element_text + "\n\n"
        
        # If unstructured didn't work or got no content, try PyPDF2
        if not content or len(content.strip()) < 100:
            print(f"Using PyPDF2 as fallback for {os.path.basename(pdf_path)}...")
            content = self.extract_text_pypdf2(pdf_path)
            
        if progress_task is not None and progress is not None:
            progress.update(progress_task, advance=30)

        # Extract images using PyMuPDF
        images, num_images = self.extract_images_pymupdf(pdf_path, doc_id, start_number)
        
        if progress_task is not None and progress is not None:
            progress.update(progress_task, advance=30)

        # Create text chunks
        chunks = []
        chunk_id = 0
        text_chunks = self.create_chunks(content)
        
        for chunk in text_chunks:
            chunks.append({
                "chunk_id": f"doc_{doc_id}_chunk_{chunk_id}",
                "original_index": chunk_id,
                "content": chunk
            })
            chunk_id += 1

        # Detect equations in text
        equations = self.detect_equations(content)
        
        if progress_task is not None and progress is not None:
            progress.update(progress_task, advance=20)

        # Handle any figures extracted by unstructured to the figures directory
        figures_dir = os.path.join(os.getcwd(), 'figures')
        if os.path.exists(figures_dir):
            image_files = glob.glob(os.path.join(figures_dir, '*'))
            for i, image_file in enumerate(sorted(image_files)):
                _, ext = os.path.splitext(image_file)
                new_num = start_number + num_images + i
                new_name = f'image_{new_num}{ext}'
                new_path = os.path.join(self.image_dir, new_name)
                json_path = os.path.join("../finalAgent_db/images/", new_name)
                
                try:
                    shutil.move(image_file, new_path)
                    images.append({
                        "image_id": f"doc_{doc_id}_image_{new_num}",
                        "path": json_path
                    })
                    num_images += 1
                except Exception as e:
                    print(f"Error moving image {image_file}: {e}")

            try:
                shutil.rmtree(figures_dir)
            except:
                pass

        if progress_task is not None and progress is not None:
            progress.update(progress_task, advance=20)

        # Create document structure
        document = {
            "doc_id": f"doc_{doc_id}",
            "original_uuid": str(uuid.uuid4().hex),
            "content": content.strip(),
            "chunks": chunks,
            "images": images,
            "equations": equations,
            "pdf_path": pdf_path,
            "filename": os.path.basename(pdf_path)
        }

        return document, num_images


async def main(pdf_files: List[str], 
               output_path: str = "../finalAgent_db/documents.json", 
               image_dir: str = "../finalAgent_db/images"):
    """
    Process PDF files into JSON format for the AnuRAG system.
    
    Args:
        pdf_files: List of PDF file paths to process
        output_path: Path to output JSON file
        image_dir: Directory to save extracted images
    """
    # Get next document ID and image number from existing documents
    next_doc_id = 1
    start_number = 1
    existing_docs = []
    
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            try:
                existing_docs = json.load(f)
                if existing_docs:
                    next_doc_id = max([int(doc['doc_id'].split('_')[1]) for doc in existing_docs]) + 1
            except json.JSONDecodeError:
                print(f"Warning: Could not parse existing {output_path}, starting fresh")
    
    # Find highest existing image number
    if os.path.exists(image_dir):
        existing_images = glob.glob(os.path.join(image_dir, 'image_*'))
        if existing_images:
            try:
                start_number = max([
                    int(os.path.splitext(os.path.basename(f))[0].split('_')[1]) 
                    for f in existing_images
                ]) + 1
            except:
                start_number = len(existing_images) + 1

    processor = PDFProcessor(image_dir)
    
    # Process new documents
    new_documents = []
    current_start_number = start_number
    
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn()
    ) as progress:
        pdf_task = progress.add_task("[cyan]Processing PDFs...", total=len(pdf_files))
        
        for i, pdf_file in enumerate(pdf_files, start=next_doc_id):
            process_task = progress.add_task(
                f"[cyan]Processing {os.path.basename(pdf_file)}...", 
                total=100
            )
            
            try:
                document, num_images = await processor.process_pdf(
                    pdf_file, i, current_start_number, process_task, progress
                )
                new_documents.append(document)
                current_start_number += num_images
                print(f"(ok) Processed: {os.path.basename(pdf_file)} - {len(document['chunks'])} chunks, {len(document['images'])} images")
            except Exception as e:
                print(f"-- Error processing {pdf_file}: {e}")
                
            progress.update(pdf_task, advance=1)
            progress.remove_task(process_task)

    # Save documents to output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_documents, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(new_documents)} PDF files")
    print(f"Output saved to: {output_path}")
    print(f"Images saved to: {image_dir}")
    print(f"Document IDs: {next_doc_id} to {next_doc_id + len(new_documents) - 1}")
    print(f"Image numbers start from: {start_number}")
    
    return new_documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process PDF files into JSON format for AnuRAG')
    parser.add_argument('--paper_path', type=str, required=True,
                        help='Path to PDF file or directory containing PDF files')
    parser.add_argument('--output_path', type=str, default="../finalAgent_db/documents.json",
                        help='Path to output JSON file')
    parser.add_argument('--image_dir', type=str, default="../finalAgent_db/images",
                        help='Directory to save extracted images')
    
    args = parser.parse_args()
    
    # Handle both single file and directory
    if os.path.isfile(args.paper_path):
        pdf_files = [args.paper_path]
    elif os.path.isdir(args.paper_path):
        pdf_files = glob.glob(os.path.join(args.paper_path, "*.pdf"))
    else:
        print(f"Error: {args.paper_path} is not a valid file or directory")
        exit(1)
    
    if not pdf_files:
        print("No PDF files found!")
        exit(1)
    
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    asyncio.run(main(pdf_files, args.output_path, args.image_dir))
