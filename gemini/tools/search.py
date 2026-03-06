"""
AnuRAG: Hybrid Search Module
Implements contextual retrieval with semantic search and BM25 using Google Gemini
Uses the new google-genai package (REST API) for better firewall compatibility
"""

import os
import pickle
import json
import numpy as np
import base64
import threading
import time
import sys
import argparse
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Import centralized configuration
from config import (
    GEMINI_CHAT_MODEL,
    GEMINI_CONTEXT_MODEL,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_VISION_MODEL,
    USE_LOCAL_EMBEDDINGS,
    LOCAL_EMBEDDING_MODEL,
    API_DELAY
)

# Try the new google-genai package first (REST API, better firewall compatibility)
GEMINI_AVAILABLE = False
genai_client = None

try:
    from google import genai
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        genai_client = genai.Client(api_key=api_key)
        GEMINI_AVAILABLE = True
except ImportError:
    # Fallback to the old google-generativeai package (gRPC)
    try:
        import google.generativeai as genai_old
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            genai_old.configure(api_key=api_key)
            GEMINI_AVAILABLE = True
        print("Using legacy google-generativeai package")
    except ImportError:
        print("Warning: google-genai not installed")

# Import Cohere for reranking
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    print("Warning: cohere not installed, reranking will be disabled")

# Import Elasticsearch for BM25
try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False
    print("Warning: elasticsearch not installed, using vector search only")

# Try to import sentence-transformers for local embeddings (FREE, no rate limits!)
LOCAL_EMBEDDINGS_AVAILABLE = False
local_embedding_model = None
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    pass

# Configuration: Set to True to use local embeddings (faster, free, no rate limits)
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

# Import fullcontext for title extraction
from fullcontext import main as fullcontext


class ContextualVectorDB:
    """
    Contextual Vector Database using Google Gemini for embeddings and contextualization.
    Implements the contextual retrieval approach from the AnuRAG paper.
    
    Supports two embedding modes:
    1. Gemini API (gemini-embedding-001) - High quality, but rate limited
    2. Local (sentence-transformers) - Free, fast, no rate limits
    """
    
    def __init__(self, name: str = "base_db", use_local_embeddings: bool = None):
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"../finalAgent_db/base_db/vector_db.pkl"
        self.token_counts = {
            'input': 0,
            'output': 0,
        }
        self.token_lock = threading.Lock()
        
        # Store reference to global client
        self.genai_client = genai_client
        
        # Determine embedding mode
        if use_local_embeddings is None:
            use_local_embeddings = USE_LOCAL_EMBEDDINGS
        
        self.use_local_embeddings = use_local_embeddings and LOCAL_EMBEDDINGS_AVAILABLE
        self.local_model = None
        
        if self.use_local_embeddings:
            print("ðŸš€ Using LOCAL embeddings (sentence-transformers) - FREE & FAST!")
            self._load_local_model()
        else:
            print("â˜ï¸  Using Gemini API embeddings (may hit rate limits)")
    
    def _load_local_model(self):
        """Load the local embedding model."""
        global local_embedding_model
        if local_embedding_model is None:
            print("Loading local embedding model (all-MiniLM-L6-v2)...")
            local_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("âœ… Local model loaded!")
        self.local_model = local_embedding_model

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        reraise=True
    )
    def situate_text_context(self, doc: str, chunk: str) -> Tuple[str, Any]:
        """
        Generate contextual description for a text chunk using Gemini.
        This helps improve search retrieval by adding document-level context.
        """
        DOCUMENT_CONTEXT_PROMPT = f"""
        <document>
        {doc[:50000]}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = f"""
        Here is the chunk we want to situate within the whole document:
        <chunk>
        {chunk}
        </chunk>
        
        Start with "This text is from the document titled [TITLE]".
        Please give a short succinct context to situate this chunk within the overall document 
        for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context, mention the title of the document and nothing else.
        """
        
        try:
            prompt = DOCUMENT_CONTEXT_PROMPT + CHUNK_CONTEXT_PROMPT
            
            if self.genai_client:
                # Use new google-genai package (REST API)
                # Use GEMINI_CONTEXT_MODEL (cheap/fast) for offline contextualization
                response = self.genai_client.models.generate_content(
                    model=GEMINI_CONTEXT_MODEL,
                    contents=prompt,
                    config={
                        'temperature': 0.0,
                        'max_output_tokens': 1000
                    }
                )
                return response.text, None
            else:
                # Fallback to old package
                import google.generativeai as genai_old
                model = genai_old.GenerativeModel(GEMINI_CONTEXT_MODEL)
                response = model.generate_content(
                    prompt,
                    generation_config=genai_old.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=1000
                    )
                )
                return response.text, None
            
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "quota" in err_str or "429" in str(e) or "resource_exhausted" in err_str or "getaddrinfo" in err_str:
                # Don't sleep here â€” let tenacity @retry handle backoff timing
                raise
            raise

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=5, max=30),
        reraise=True
    )
    def situate_image_context(self, image_path: str, title: str) -> Tuple[str, Any]:
        """
        Generate contextual description for an image using Gemini's vision capabilities.
        Analyzes circuit diagrams, graphs, equations, and schematics.
        """
        IMAGE_CONTEXT_PROMPT = f"""
        Start by stating: "This image is from the document titled: {title}"

        Then analyze the image using the following format:

        1. Image Type and Summary:
           - "It is a [type of image] that [briefly describes what it shows]."

        2. Detailed Technical Description:
           - For circuit diagrams: Explain components, connections, and operation
           - For graphs: Explain axes, trends, and performance metrics
           - For block diagrams: Describe system architecture
           - For schematics: Describe circuit topology and biasing
           - For equations: Extract and explain the mathematical formula

        3. Contextual Significance:
           - Explain how this image supports the paper's main contributions

        Be comprehensive but concise in your technical analysis.
        """
        
        try:
            import PIL.Image
            
            # Resolve the image path
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.path.dirname(__file__), image_path)
            
            if self.genai_client:
                # Use new google-genai package (REST API)
                # Read image as bytes
                # Use GEMINI_CONTEXT_MODEL for offline contextualization (cheaper)
                with open(image_path, "rb") as img_file:
                    image_bytes = img_file.read()
                
                response = self.genai_client.models.generate_content(
                    model=GEMINI_CONTEXT_MODEL,
                    contents=[
                        IMAGE_CONTEXT_PROMPT,
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(image_bytes).decode()
                            }
                        }
                    ],
                    config={
                        'temperature': 0.0,
                        'max_output_tokens': 1024
                    }
                )
                return response.text, None
            else:
                # Fallback to old package
                import google.generativeai as genai_old
                image = PIL.Image.open(image_path)
                model = genai_old.GenerativeModel(GEMINI_VISION_MODEL)
                response = model.generate_content(
                    [IMAGE_CONTEXT_PROMPT, image],
                    generation_config=genai_old.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=1024
                    )
                )
                return response.text, None
            
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "quota" in err_str or "429" in str(e) or "resource_exhausted" in err_str or "getaddrinfo" in err_str:
                # Don't sleep here â€” let tenacity @retry handle backoff timing
                raise  # Let @retry handle it
            print(f"Error processing image: {str(e)}")
            return f"Error processing image: {str(e)}", None

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using either local model or Gemini API."""
        # Use local embeddings if available and configured
        if self.use_local_embeddings and self.local_model:
            return self.local_model.encode(text, convert_to_numpy=True).tolist()
        
        # Use Gemini API
        try:
            if genai_client:
                # Use new google-genai package (REST API)
                response = genai_client.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    contents=text,
                    config={'output_dimensionality': 768}
                )
                return response.embeddings[0].values
            else:
                # Fallback to old package
                import google.generativeai as genai_old
                result = genai_old.embed_content(
                    model=f"models/{GEMINI_EMBEDDING_MODEL}",
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
        except Exception as e:
            print(f"Error getting embedding: {e}")
            raise

    def get_query_embedding(self, text: str) -> List[float]:
        """Get embedding for query using Gemini embedding model."""
        try:
            if genai_client:
                # Use new google-genai package (REST API)
                response = genai_client.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    contents=text,
                    config={'output_dimensionality': 768}
                )
                return response.embeddings[0].values
            else:
                # Fallback to old package
                import google.generativeai as genai_old
                result = genai_old.embed_content(
                    model=f"models/{GEMINI_EMBEDDING_MODEL}",
                    content=text,
                    task_type="retrieval_query"
                )
                return result['embedding']
        except Exception as e:
            print(f"Error getting query embedding: {e}")
            raise

    def load_data(self, dataset: List[Dict[str, Any]], parallel_threads: int = 1,
                  contextualize: bool = False, text_only: bool = False):
        """
        Load and process documents into the vector database.
        
        Args:
            dataset: List of document dicts with 'chunks', 'images', 'content' fields.
            parallel_threads: Not used (kept for API compat).
            contextualize: If True, generate contextual summaries for each chunk/image
                          using GEMINI_CONTEXT_MODEL before embedding.
            text_only: If True (with contextualize), skip image vision calls.
                       Images get filename-based descriptions instead of vision analysis.
        """
        CONTEXT_CHECKPOINT_PATH = os.path.join(
            os.path.dirname(self.db_path), "context_checkpoint.json"
        )
        
        # Cache for document titles to avoid repeated API calls
        title_cache = {}
        
        def get_cached_title(doc):
            """Get title from cache or extract once per document."""
            doc_id = doc['doc_id']
            if doc_id not in title_cache:
                pdf_path = doc.get("pdf_path", "")
                title_cache[doc_id] = os.path.splitext(os.path.basename(pdf_path))[0]
            return title_cache[doc_id]
        
        def process_item_simple(item):
            """Process single item WITHOUT contextualization (fast, no API calls)."""
            try:
                if 'content' in item[1]:  # It's a text chunk
                    doc, chunk = item
                    return {
                        'text_to_embed': chunk['content'],
                        'metadata': {
                            'doc_id': doc['doc_id'],
                            'original_uuid': doc.get('original_uuid', ''),
                            'chunk_id': chunk['chunk_id'],
                            'original_index': chunk.get('original_index', 0),
                            'original_content': chunk['content'],
                            'contextualized_content': chunk['content']
                        }
                    }
                else:  # It's an image
                    doc, image = item
                    title = get_cached_title(doc)
                    image_text = f"Image from paper: {title}. Path: {image['path']}"
                    return {
                        'text_to_embed': image_text,
                        'metadata': {
                            'doc_id': doc['doc_id'],
                            'image_id': image['image_id'],
                            'path': image['path'],
                            'contextualized_content': image_text
                        }
                    }
            except Exception as e:
                print(f"Error processing item: {e}")
                return None

        def process_item_contextual(item, idx):
            """Process single item WITH LLM contextualization."""
            try:
                if 'content' in item[1]:  # It's a text chunk
                    doc, chunk = item
                    # Call situate_text_context with full document + chunk
                    doc_text = doc.get('content', '')
                    chunk_text = chunk['content']
                    try:
                        context_summary, _ = self.situate_text_context(doc_text, chunk_text)
                        contextualized = f"{context_summary}\n\n{chunk_text}"
                    except Exception as e:
                        err_str = str(e)
                        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                            # Rate limited â€” wait and retry once
                            print(f"  â³ Rate limit at chunk {idx}, waiting 60s...")
                            time.sleep(60)
                            try:
                                context_summary, _ = self.situate_text_context(doc_text, chunk_text)
                                contextualized = f"{context_summary}\n\n{chunk_text}"
                            except Exception as e2:
                                print(f"  âš ï¸ Retry failed for chunk {idx}: {e2}")
                                contextualized = chunk_text
                        else:
                            print(f"  âš ï¸ Context failed for chunk {idx}, using raw: {e}")
                            contextualized = chunk_text
                    
                    return {
                        'text_to_embed': contextualized,
                        'metadata': {
                            'doc_id': doc['doc_id'],
                            'original_uuid': doc.get('original_uuid', ''),
                            'chunk_id': chunk['chunk_id'],
                            'original_index': chunk.get('original_index', 0),
                            'original_content': chunk_text,
                            'contextualized_content': contextualized
                        }
                    }
                else:  # It's an image
                    doc, image = item
                    title = get_cached_title(doc)
                    image_path = image.get('path', '')
                    
                    # Vision-based contextualization (tenacity @retry handles 429 backoff)
                    try:
                        if os.path.exists(image_path):
                            context_summary, _ = self.situate_image_context(image_path, title)
                            contextualized = context_summary
                        else:
                            contextualized = f"Image from paper: {title}. Path: {image_path}"
                    except Exception as e:
                        print(f"  âš ï¸ Image context failed for {image_path}: {e}")
                        contextualized = f"Image from paper: {title}. Path: {image_path}"
                    
                    return {
                        'text_to_embed': contextualized,
                        'metadata': {
                            'doc_id': doc['doc_id'],
                            'image_id': image['image_id'],
                            'path': image_path,
                            'contextualized_content': contextualized
                        }
                    }
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                return None

        # â”€â”€ Prepare all items â”€â”€
        all_items = []
        for doc in dataset:
            for chunk in doc.get("chunks", []):
                all_items.append((doc, chunk))
            for image in doc.get("images", []):
                all_items.append((doc, image))

        print(f"Total items to process: {len(all_items)}")
        
        texts_to_embed = []
        metadata = []
        
        if contextualize:
            # â”€â”€ CONTEXTUAL MODE: LLM generates a summary for each chunk â”€â”€
            print(f"ðŸ§  Contextual mode ENABLED (using model: {GEMINI_CONTEXT_MODEL})")
            if text_only:
                print(f"   ðŸ“ Text-only mode: images will use filename-based descriptions")
            
            # Separate delays for text and image calls
            text_delay = API_DELAY       # From config, default 4.0s
            image_delay = API_DELAY * 2  # Images use vision API, often stricter quota
            min_delay = 1.0
            max_delay = 30.0  # Don't let delays grow absurdly
            
            text_success_streak = 0
            image_fail_streak = 0          # Consecutive image 429 failures
            image_skip_mode = False        # True = auto-skip images (quota exhausted)
            image_skip_until = 0           # timestamp to resume image attempts
            IMAGE_FAIL_THRESHOLD = 3       # After 3 consecutive image 429s, skip images
            IMAGE_SKIP_DURATION = 600      # Skip images for 10 minutes, then retry one
            total_images_skipped = 0
            
            # Load checkpoint if exists (resume from crash/rate-limit)
            start_idx = 0
            if os.path.exists(CONTEXT_CHECKPOINT_PATH):
                try:
                    with open(CONTEXT_CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
                        checkpoint = json.load(f)
                    texts_to_embed = checkpoint.get('texts', [])
                    metadata = checkpoint.get('metadata', [])
                    start_idx = len(texts_to_embed)
                    print(f"ðŸ“‚ Resuming from checkpoint: {start_idx}/{len(all_items)} done")
                except Exception as e:
                    print(f"âš ï¸ Could not load checkpoint: {e}, starting fresh")
            
            # Estimate remaining time
            remaining = len(all_items) - start_idx
            print(f"   Items remaining: {remaining}")
            print(f"   Estimated: ~{remaining * 1.0 / 60:.0f} min (paid tier) "
                  f"or ~{remaining * 4 / 3600:.1f} hrs (free tier)")
            
            for i in tqdm(range(start_idx, len(all_items)), 
                          desc="Contextualizing", initial=start_idx, total=len(all_items)):
                item = all_items[i]
                is_image = 'content' not in item[1]
                
                # Decide whether to skip this image
                skip_this_image = False
                if is_image and not text_only:
                    if image_skip_mode:
                        if time.time() < image_skip_until:
                            # Still in skip window â€” use simple processing
                            skip_this_image = True
                        else:
                            # Cooldown expired â€” try one image to probe quota
                            tqdm.write(f"  ðŸ”„ Probing image API (skip cooldown expired)...")
                            image_skip_mode = False
                
                # In text_only mode OR image-skip mode, use simple processing for images
                if (text_only and is_image) or skip_this_image:
                    result = process_item_simple(item)
                    if skip_this_image:
                        total_images_skipped += 1
                else:
                    result = process_item_contextual(item, i)
                
                if result:
                    texts_to_embed.append(result['text_to_embed'])
                    metadata.append(result['metadata'])
                    
                    # Check if an image call failed (fell back to filename description)
                    is_image_fallback = (is_image and not text_only and not skip_this_image and
                                         (result['text_to_embed'].startswith("Error processing image:") or
                                          result['text_to_embed'].startswith("Image from paper:")))
                    
                    if is_image_fallback:
                        image_fail_streak += 1
                        image_delay = min(max_delay, image_delay * 1.5)
                        
                        if image_fail_streak >= IMAGE_FAIL_THRESHOLD:
                            # Daily quota likely exhausted â€” stop wasting time on images
                            image_skip_mode = True
                            image_skip_until = time.time() + IMAGE_SKIP_DURATION
                            tqdm.write(f"  ðŸš« {image_fail_streak} consecutive image 429s â€” "
                                       f"skipping images for {IMAGE_SKIP_DURATION//60} min "
                                       f"(total skipped: {total_images_skipped})")
                        else:
                            tqdm.write(f"  âš ï¸ Image fail #{image_fail_streak}, "
                                       f"image delay now {image_delay:.1f}s")
                            time.sleep(image_delay)
                    elif is_image and not text_only and not skip_this_image:
                        # Image succeeded! Reset image failure tracking
                        if image_fail_streak > 0:
                            tqdm.write(f"  âœ… Image API recovered! Resetting fail streak.")
                        image_fail_streak = 0
                        image_delay = max(min_delay, image_delay * 0.8)
                    
                    if not is_image:
                        # Text chunk succeeded â€” track text streak separately
                        text_success_streak += 1
                
                # Rate limiting for API calls (skip for simple-processed items)
                if not ((text_only and is_image) or skip_this_image):
                    if is_image:
                        time.sleep(image_delay)
                    else:
                        time.sleep(text_delay)
                        # Gradually speed up text after sustained success
                        if text_success_streak > 50 and text_delay > min_delay:
                            text_delay = max(min_delay, text_delay * 0.95)
                            if text_success_streak % 200 == 0:
                                tqdm.write(f"  âœ¨ Text delay â†’ {text_delay:.2f}s (streak: {text_success_streak})")
                
                # Save checkpoint every 50 items (more frequent for long runs)
                if (i + 1) % 50 == 0:
                    checkpoint = {'texts': texts_to_embed, 'metadata': metadata}
                    os.makedirs(os.path.dirname(CONTEXT_CHECKPOINT_PATH), exist_ok=True)
                    with open(CONTEXT_CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint, f)
                    tqdm.write(f"  âœ… Checkpoint saved: {len(texts_to_embed)}/{len(all_items)} "
                               f"(text_delay={text_delay:.2f}s, img_skip={image_skip_mode}, "
                               f"img_skipped={total_images_skipped})")
            
            # Final checkpoint save
            checkpoint = {'texts': texts_to_embed, 'metadata': metadata}
            os.makedirs(os.path.dirname(CONTEXT_CHECKPOINT_PATH), exist_ok=True)
            with open(CONTEXT_CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f)
            print(f"\nâœ… Contextualization complete! {len(texts_to_embed)} items processed.")
        else:
            # â”€â”€ FAST MODE: no API calls, just raw chunk content â”€â”€
            print("âš¡ Fast mode (no contextualization API calls)...")
            for i, item in enumerate(tqdm(all_items, desc="Processing")):
                result = process_item_simple(item)
                if result:
                    texts_to_embed.append(result['text_to_embed'])
                    metadata.append(result['metadata'])
        
        print(f"\nProcessed {len(texts_to_embed)} items. Now creating embeddings...")
        
        # Batch embed all texts at once
        self._embed_and_store_optimized(texts_to_embed, metadata)
        self.save_db()
        
        print(f"\nâœ… Vector database loaded. Total items: {len(texts_to_embed)}")

    def _embed_and_store_optimized(self, texts: List[str], data: List[Dict[str, Any]]):
        """Generate embeddings with optimized batching and rate limiting."""
        
        # Check for existing checkpoint to resume from
        start_index = 0
        embeddings = []
        
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    existing_data = pickle.load(f)
                existing_count = len(existing_data.get("embeddings", []))
                if existing_count > 0 and existing_count < len(texts):
                    print(f"ðŸ“‚ Found checkpoint with {existing_count}/{len(texts)} embeddings")
                    print(f"â–¶ï¸  Resuming from index {existing_count}...")
                    # Load existing embeddings
                    embeddings = existing_data.get("embeddings", [])
                    start_index = existing_count
                elif existing_count >= len(texts):
                    print(f"âœ… All {existing_count} embeddings already exist. Skipping.")
                    self.embeddings = existing_data.get("embeddings", [])
                    self.metadata = existing_data.get("metadata", [])
                    return
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
        
        # Use larger batches for local embeddings (no rate limits!)
        if self.use_local_embeddings and self.local_model:
            batch_size = 500  # Local model can handle large batches
            print(f"ðŸš€ Creating embeddings for {len(texts) - start_index} texts using LOCAL model (FREE & FAST)...")
        else:
            batch_size = 20  # Reduced batch size for Gemini free tier
            print(f"Creating embeddings for {len(texts) - start_index} texts using Gemini API...")
            print("âš ï¸  Using rate-limited mode for free tier (15 RPM limit)")
        
        failed_indices = []
        
        # Calculate total batches (remaining)
        remaining_texts = len(texts) - start_index
        total_batches = (remaining_texts + batch_size - 1) // batch_size
        
        for i in range(start_index, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = (i - start_index) // batch_size + 1
            
            print(f"Embedding batch {batch_num}/{total_batches} ({len(embeddings)}/{len(texts)} done)")
            
            # Try batch embedding with exponential backoff
            max_retries = 5
            base_delay = 60  # Start with 60 seconds on rate limit
            
            for attempt in range(max_retries):
                try:
                    batch_embeddings = self.get_embeddings_batch(batch_texts)
                    embeddings.extend(batch_embeddings)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"â³ Rate limit hit. Waiting {delay}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(delay)
                    else:
                        print(f"Error in batch: {e}")
                        # Fall back to individual embeddings with delays
                        for idx, text in enumerate(batch_texts):
                            try:
                                time.sleep(4)  # 4 seconds = 15 RPM max
                                embedding = self.get_embedding(text)
                                embeddings.append(embedding)
                            except Exception as e2:
                                print(f"Failed text {i + idx}: {e2}")
                                embeddings.append([0] * 768)
                                failed_indices.append(i + idx)
                        break
            else:
                # All retries failed, use placeholder embeddings
                print(f"âš ï¸ Batch {batch_num} failed after {max_retries} retries. Using placeholders.")
                for _ in batch_texts:
                    embeddings.append([0] * 768)
                    failed_indices.append(len(embeddings) - 1)
            
            # Progress save every 5 batches (save directly, not appending)
            if batch_num % 5 == 0:
                self._save_checkpoint(embeddings, data[:len(embeddings)])
            
            # Rate limiting - only needed for Gemini API, not local embeddings
            if not self.use_local_embeddings and i + batch_size < len(texts):
                time.sleep(4)  # 4 seconds = 15 RPM max
        
        self.embeddings = embeddings
        self.metadata = data
        
        if failed_indices:
            print(f"âš ï¸ {len(failed_indices)} embeddings failed and use placeholders")
        print(f"âœ… Embeddings created successfully! Total: {len(embeddings)}")
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in a single API call."""
        # Use local embeddings if available - FAST and FREE!
        if self.use_local_embeddings and self.local_model:
            embeddings = self.local_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return [emb.tolist() for emb in embeddings]
        
        # Use Gemini API with retries
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                if genai_client:
                    # Use new google-genai package
                    result = genai_client.models.embed_content(
                        model=GEMINI_EMBEDDING_MODEL,
                        contents=texts,
                        config={'output_dimensionality': 768}
                    )
                    return [e.values for e in result.embeddings]
                else:
                    # Fallback to individual embeddings with delay
                    embeddings = []
                    for text in texts:
                        time.sleep(4)  # Rate limit
                        embeddings.append(self.get_embedding(text))
                    return embeddings
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    delay = 60 * (attempt + 1)
                    print(f"Rate limit, waiting {delay}s...")
                    time.sleep(delay)
                else:
                    raise
        print("Embeddings created successfully!")

    def _save_checkpoint(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """Save checkpoint during embedding creation (overwrites, doesn't append)."""
        data = {
            "embeddings": embeddings,
            "metadata": metadata,
            "query_cache": json.dumps(self.query_cache)
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)
        print(f"âœ… Checkpoint saved: {len(embeddings)} embeddings")

    def save_db(self):
        """Save the database to disk (overwrites existing data)."""
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache)
        }
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Database saved with {len(data['embeddings'])} total embeddings")

    def load_db(self):
        """Load the database from disk."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Vector database file not found: {self.db_path}")
        
        with open(self.db_path, "rb") as f:
            data = pickle.load(f)
        
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data.get("query_cache", "{}"))
        
        print(f"Loaded database with {len(self.embeddings)} embeddings")

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search the database using semantic similarity.
        """
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.get_query_embedding(query)
            self.query_cache[query] = query_embedding
        
        if not self.embeddings:
            raise ValueError("Vector database is empty. Please load or embed data first.")
        
        # Check dimension compatibility
        db_dim = len(self.embeddings[0]) if self.embeddings else 0
        query_dim = len(query_embedding)
        if db_dim != query_dim:
            raise ValueError(
                f"Dimension mismatch: Database has {db_dim} dimensions, "
                f"but query has {query_dim}. Please rebuild the database."
            )
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for index in top_indices:
            result = {
                "metadata": self.metadata[index],
                "similarity": float(similarities[index]),
            }
            results.append(result)
        
        return results


class ElasticSearchBM25:
    """BM25 search using Elasticsearch."""
    
    def __init__(self, name: str = "contextual_bm25_index", timeout: int = 5):
        if not ES_AVAILABLE:
            raise ImportError("Elasticsearch is not installed")
        
        # Use short timeout (5 seconds) to fail fast if ES is not running
        # Compatible with elasticsearch 8.x and 9.x
        try:
            self.es_client = Elasticsearch(
                hosts=["http://localhost:9200"],
                request_timeout=timeout,
                max_retries=1,
                retry_on_timeout=False
            )
        except Exception as init_error:
            print(f"âš ï¸ ES init error: {init_error}")
            raise ConnectionError(f"Cannot initialize Elasticsearch client: {init_error}")
            
        self.index_name = name
        
        # Quick health check - fail fast if ES is not running
        try:
            info = self.es_client.info()
            print(f"âœ… Connected to Elasticsearch {info['version']['number']}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Elasticsearch: {e}")
        
        self.create_index()
    
    def create_index(self):
        """Create the Elasticsearch index with BM25 settings."""
        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "contextualized_content": {"type": "text", "analyzer": "english"},
                    "doc_id": {"type": "keyword", "index": False},
                    "chunk_id": {"type": "keyword", "index": False},
                    "image_id": {"type": "keyword", "index": False},
                    "original_index": {"type": "integer", "index": False},
                    "content_type": {"type": "keyword"},
                    "image_path": {"type": "keyword", "index": False}
                }
            }
        }
        
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(index=self.index_name, body=index_settings)
            print(f"Created Elasticsearch index: {self.index_name}")
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for BM25 search."""
        actions = []
        
        for doc in documents:
            is_image = 'image_id' in doc
            
            action = {
                "_index": self.index_name,
                "_source": {
                    "contextualized_content": doc.get("contextualized_content", ""),
                    "doc_id": doc.get("doc_id", ""),
                    "content_type": "image" if is_image else "text"
                }
            }
            
            if is_image:
                action["_source"].update({
                    "image_id": doc.get("image_id", ""),
                    "image_path": doc.get("path", "")
                })
            else:
                action["_source"].update({
                    "content": doc.get("original_content", ""),
                    "chunk_id": doc.get("chunk_id", ""),
                    "original_index": doc.get("original_index", 0)
                })
            
            actions.append(action)
        
        success, _ = bulk(self.es_client, actions)
        self.es_client.indices.refresh(index=self.index_name)
        return success
    
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search using BM25."""
        self.es_client.indices.refresh(index=self.index_name)
        
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content", "contextualized_content"],
                                "type": "best_fields"
                            }
                        }
                    ]
                }
            },
            "size": top_k,
        }
        
        response = self.es_client.search(index=self.index_name, body=search_body)
        
        results = []
        for hit in response["hits"]["hits"]:
            result = {
                "doc_id": hit["_source"]["doc_id"],
                "contextualized_content": hit["_source"]["contextualized_content"],
                "score": hit["_score"],
            }
            
            if hit["_source"]["content_type"] == "text":
                result.update({
                    "original_index": hit["_source"].get("original_index", 0),
                    "content": hit["_source"].get("content", ""),
                })
            else:
                result.update({
                    "image_id": hit["_source"].get("image_id", ""),
                    "image_path": hit["_source"].get("image_path", ""),
                })
            
            results.append(result)
        
        return results


def create_elasticsearch_bm25_index(db: ContextualVectorDB) -> Optional[ElasticSearchBM25]:
    """Create and populate Elasticsearch BM25 index.
    
    Uses fast timeout (5s) to fail quickly if Elasticsearch is not running.
    Falls back gracefully to vector-only search.
    """
    if not ES_AVAILABLE:
        print("â„¹ï¸ Elasticsearch not installed. Using vector search only.")
        return None
    
    try:
        es_bm25 = ElasticSearchBM25(timeout=5)  # 5 second timeout
        es_bm25.index_documents(db.metadata)
        print("âœ… Elasticsearch connected successfully.")
        return es_bm25
    except ConnectionError as e:
        print(f"âš ï¸ Elasticsearch offline: {e}")
        print("â„¹ï¸ Continuing with vector search only (no BM25 hybrid search).")
        return None
    except Exception as e:
        print(f"âš ï¸ Elasticsearch error: {e}")
        print("â„¹ï¸ Continuing with vector search only.")
        return None


def chunk_to_content(result: Dict[str, Any]) -> str:
    """Convert a result to content string for reranking."""
    if 'image_id' in result.get('item', {}):
        return result['item'].get('contextualized_content', '')
    else:
        original = result['item'].get('original_content', '')
        contextualized = result['item'].get('contextualized_content', '')
        return f"{original}\n\nContext: {contextualized}"


def retrieve_advanced(query: str, db: ContextualVectorDB, es_bm25: Optional[ElasticSearchBM25],
                      k: int, semantic_weight: float = 0.8, bm25_weight: float = 0.2) -> Tuple[List, int, int]:
    """
    Advanced hybrid retrieval combining semantic search and BM25.
    """
    num_chunks_to_recall = 150
    
    # Semantic search
    semantic_results = db.search(query, top_k=num_chunks_to_recall)
    
    ranked_ids = []
    for result in semantic_results:
        metadata = result['metadata']
        if 'image_id' in metadata:
            ranked_ids.append(('image', metadata['doc_id'], metadata['image_id']))
        else:
            ranked_ids.append(('text', metadata['doc_id'], metadata['original_index']))
    
    # BM25 search if available
    ranked_bm25_ids = []
    if es_bm25:
        bm25_results = es_bm25.search(query, top_k=num_chunks_to_recall)
        for result in bm25_results:
            if 'content' in result:
                ranked_bm25_ids.append(('text', result['doc_id'], result['original_index']))
            else:
                ranked_bm25_ids.append(('image', result['doc_id'], result['image_id']))
    
    # Combine results
    item_ids = list(set(ranked_ids + ranked_bm25_ids))
    item_id_to_score = {}
    
    for item_id in item_ids:
        score = 0
        if item_id in ranked_ids:
            index = ranked_ids.index(item_id)
            score += semantic_weight * (1 / (index + 1))
        if item_id in ranked_bm25_ids:
            index = ranked_bm25_ids.index(item_id)
            score += bm25_weight * (1 / (index + 1))
        item_id_to_score[item_id] = score
    
    # Sort by score
    sorted_ids = sorted(item_id_to_score.keys(), key=lambda x: item_id_to_score[x], reverse=True)
    
    # Prepare final results
    final_results = []
    semantic_count = 0
    bm25_count = 0
    
    for item_id in sorted_ids[:k]:
        content_type, doc_id, sub_id = item_id
        
        try:
            if content_type == 'text':
                item_metadata = next(
                    (item for item in db.metadata 
                     if 'chunk_id' in item 
                     and item['doc_id'] == doc_id 
                     and item['original_index'] == sub_id),
                    None
                )
            else:
                item_metadata = next(
                    (item for item in db.metadata 
                     if 'image_id' in item 
                     and item['doc_id'] == doc_id 
                     and item['image_id'] == sub_id),
                    None
                )
            
            if item_metadata is None:
                continue
            
            is_from_semantic = item_id in ranked_ids
            is_from_bm25 = item_id in ranked_bm25_ids
            
            final_results.append({
                'item': item_metadata,
                'content_type': content_type,
                'score': item_id_to_score[item_id],
                'from_semantic': is_from_semantic,
                'from_bm25': is_from_bm25
            })
            
            if is_from_semantic and not is_from_bm25:
                semantic_count += 1
            elif is_from_bm25 and not is_from_semantic:
                bm25_count += 1
            else:
                semantic_count += 0.5
                bm25_count += 0.5
                
        except Exception as e:
            print(f"Warning: Error processing item {item_id}: {e}")
            continue
    
    return final_results, semantic_count, bm25_count


def retrieve_rerank(query: str, db: ContextualVectorDB, es_bm25: Optional[ElasticSearchBM25],
                    k: int) -> List[Dict[str, Any]]:
    """Retrieve and rerank results using Cohere."""
    if not COHERE_AVAILABLE:
        results, _, _ = retrieve_advanced(query, db, es_bm25, k)
        return results
    
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    
    # Retrieve more results than needed
    results, _, _ = retrieve_advanced(query, db, es_bm25, k * 10)
    
    if not results:
        return []
    
    # Prepare documents for reranking
    documents = [chunk_to_content(res) for res in results]
    
    try:
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=k
        )
        time.sleep(0.1)
        
        final_results = []
        for r in response.results:
            original_result = results[r.index]
            final_results.append({
                "item": original_result['item'],
                "content_type": original_result['content_type'],
                "score": r.relevance_score,
                "from_semantic": original_result['from_semantic'],
                "from_bm25": original_result.get('from_bm25', False)
            })
        return final_results
        
    except Exception as e:
        print(f"Reranking failed: {e}, returning unranked results")
        return results[:k]


def main(query: str = None, load_data: bool = False, contextualize: bool = False,
         text_only: bool = False) -> Optional[Dict]:
    """
    Main function to search or load data into the vector database.
    
    Args:
        query: The search query string
        load_data: If True, load new data into DB
        contextualize: If True (with load_data), generate contextual summaries
                       for each chunk using GEMINI_CONTEXT_MODEL before embedding.
        text_only: If True (with contextualize), skip image vision calls.
        
    Returns:
        Search results if query is provided
    """
    if load_data:
        # Load existing content
        pdf_content_path = "../finalAgent_db/pdf_content.json"
        existing_content = {}
        
        try:
            if os.path.exists(pdf_content_path) and os.path.getsize(pdf_content_path) > 0:
                with open(pdf_content_path, "r", encoding="utf-8") as f:
                    existing_content = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing content: {e}")
        
        # Load documents
        with open("../finalAgent_db/documents.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        # Get titles for documents
        for i, item in enumerate(dataset):
            try:
                if not os.path.exists(item.get("pdf_path", "")):
                    filename = os.path.basename(item.get("pdf_path", "unknown.pdf"))
                    item["title"] = os.path.splitext(filename)[0]
                else:
                    title = fullcontext(
                        "What is the title of the document? Just give the title, nothing else.",
                        item["pdf_path"]
                    )
                    item["title"] = title if title and not title.startswith("Error:") else \
                                   os.path.splitext(os.path.basename(item["pdf_path"]))[0]
                
                if item["title"] not in existing_content:
                    existing_content[item["title"]] = item.get("content", "")
                
                if i < len(dataset) - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error processing document: {e}")
                filename = os.path.basename(item.get("pdf_path", "unknown.pdf"))
                item["title"] = os.path.splitext(filename)[0]
        
        # Save content
        with open(pdf_content_path, "w", encoding="utf-8") as f:
            json.dump(existing_content, f, indent=4)
        
        # Create vector database
        vector_db = ContextualVectorDB("base_db")
        vector_db.load_data(dataset, contextualize=contextualize, text_only=text_only)
        
        # Save context data
        context_data = []
        for item in vector_db.metadata:
            context_entry = {"doc_id": item["doc_id"]}
            
            if "chunk_id" in item:
                context_entry.update({
                    "type": "text",
                    "content": item["original_content"],
                    "contextualized_content": item["contextualized_content"],
                    "chunk_id": item["chunk_id"]
                })
            else:
                context_entry.update({
                    "type": "image",
                    "image_id": item["image_id"],
                    "path": item["path"],
                    "contextualized_content": item["contextualized_content"]
                })
            
            context_data.append(context_entry)
        
        with open("../finalAgent_db/context.json", "w", encoding="utf-8") as f:
            json.dump(context_data, f, indent=4)
        print("Context data saved to context.json")
    
    if query:
        vector_db = ContextualVectorDB("base_db")
        vector_db.load_db()
        
        # Try to create Elasticsearch index
        es_bm25 = None
        if ES_AVAILABLE:
            try:
                es_bm25 = create_elasticsearch_bm25_index(vector_db)
            except Exception as e:
                print(f"Warning: Elasticsearch not available: {e}")
        
        # Search
        if es_bm25 and COHERE_AVAILABLE:
            results = retrieve_rerank(query, vector_db, es_bm25, 10)
        elif es_bm25:
            results, _, _ = retrieve_advanced(query, vector_db, es_bm25, 10)
        else:
            # Use only vector search
            raw_results = vector_db.search(query, top_k=10)
            results = [
                {
                    'item': r['metadata'],
                    'content_type': 'image' if 'image_id' in r['metadata'] else 'text',
                    'score': r['similarity'],
                    'from_semantic': True,
                    'from_bm25': False
                }
                for r in raw_results
            ]
        
        # Build doc_id â†’ paper title mapping from documents.json
        doc_title_map = {}
        try:
            docs_path = os.path.join(os.path.dirname(__file__), '..', 'finalAgent_db', 'documents.json')
            with open(docs_path, 'r', encoding='utf-8') as f:
                all_docs = json.load(f)
            for doc in all_docs:
                did = doc.get('doc_id', '')
                fname = doc.get('filename', '')
                ppath = doc.get('pdf_path', '')
                raw_name = fname or (os.path.basename(ppath) if ppath else '')
                clean_title = raw_name.replace('.pdf', '').replace('_', ' ').strip()
                if did and clean_title:
                    doc_title_map[did] = clean_title
        except Exception as e:
            print(f"Warning: Could not load paper titles: {e}")
        
        # Enrich each result with paper_title
        for result in results:
            meta = result.get('item', {})
            did = meta.get('doc_id', '')
            if did and did in doc_title_map:
                meta['paper_title'] = doc_title_map[did]
        
        # Prepare response
        response = {
            'text': results,
            'images': {}
        }
        
        # Extract image information
        for idx, result in enumerate(results, 1):
            if result.get('content_type') == 'image' and 'item' in result and 'path' in result['item']:
                response['images'][idx] = result['item']['path']
        
        # Save results
        with open("../finalAgent_db/search_results.json", "w", encoding="utf-8") as f:
            json.dump(response, f, indent=4)
        
        # Print results
        print("\n" + "=" * 60)
        print("SEARCH RESULTS")
        print("=" * 60)
        print(f"Query: {query}")
        print(f"Found {len(results)} relevant items:")
        
        for i, result in enumerate(results[:3]):
            print(f"\n--- Result {i+1} ---")
            meta = result.get('item', {})
            doc_id = meta.get('doc_id', 'Unknown')
            paper_title = meta.get('paper_title', '')
            content = (
                meta.get('content') or 
                meta.get('original_content') or 
                meta.get('contextualized_content', 'No content')
            )
            print(f"Document: {doc_id}")
            if paper_title:
                print(f"Paper: {paper_title[:80]}")
            print(f"Type: {result.get('content_type', 'unknown')}")
            print(f"Score: {result.get('score', 0):.4f}")
            print(f"Content: {content[:200]}...")
        
        print("=" * 60)
        
        return response
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the AnuRAG vector database")
    parser.add_argument("--load_data", action="store_true", 
                        help="Load data from documents.json and create vector database")
    parser.add_argument("--contextualize", action="store_true",
                        help="Enable LLM contextualization when loading data (one-time offline pass)")
    parser.add_argument("--text_only", action="store_true",
                        help="With --contextualize: skip image vision calls, only contextualize text chunks")
    parser.add_argument("--query", type=str, 
                        default="show me the schematic diagram of a PTAT voltage generator",
                        help="Search query")
    
    args = parser.parse_args()
    main(query=args.query, load_data=args.load_data, 
         contextualize=args.contextualize, text_only=args.text_only)
