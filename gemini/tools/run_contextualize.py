"""
Run contextualization directly on documents.json, bypassing the slow title-extraction step.
Usage: python run_contextualize.py [--text_only]

This script:
1. Loads documents.json (already has chunks + images)
2. Calls ContextualVectorDB.load_data(contextualize=True) 
3. Saves the contextualized vector DB + context.json
4. Supports checkpoint/resume if interrupted
"""
import json
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

from config import GEMINI_CONTEXT_MODEL, GEMINI_CHAT_MODEL
from search import ContextualVectorDB, create_elasticsearch_bm25_index, ES_AVAILABLE

def main():
    parser = argparse.ArgumentParser(description="Run contextualization on existing documents")
    parser.add_argument("--text_only", action="store_true",
                        help="Only contextualize text chunks, skip image vision calls")
    args = parser.parse_args()
    
    print("=" * 60)
    print("AnuRAG: Contextual Embedding Pipeline")
    print("=" * 60)
    print(f"Context model:  {GEMINI_CONTEXT_MODEL}")
    print(f"Chat model:     {GEMINI_CHAT_MODEL}")
    print(f"Text only:      {args.text_only}")
    print("=" * 60)
    
    # Load documents
    docs_path = "../finalAgent_db/documents.json"
    print(f"\nLoading documents from {docs_path}...")
    with open(docs_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    total_chunks = sum(len(d.get("chunks", [])) for d in dataset)
    total_images = sum(len(d.get("images", [])) for d in dataset)
    print(f"  Documents: {len(dataset)}")
    print(f"  Text chunks: {total_chunks}")
    print(f"  Images: {total_images}")
    print(f"  Total items: {total_chunks + total_images}")
    
    # Assign titles from filenames (fast, no API calls)
    for doc in dataset:
        if "title" not in doc:
            pdf_path = doc.get("pdf_path", "unknown.pdf")
            doc["title"] = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Create vector database and run contextualization
    print(f"\n🚀 Starting contextualization...")
    t_start = time.time()
    
    vector_db = ContextualVectorDB("base_db")
    vector_db.load_data(dataset, contextualize=True, text_only=args.text_only)
    
    elapsed = time.time() - t_start
    print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    
    # Save context data for inspection
    print("\nSaving context.json...")
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
                "image_id": item.get("image_id", ""),
                "path": item.get("path", ""),
                "contextualized_content": item["contextualized_content"]
            })
        
        context_data.append(context_entry)
    
    with open("../finalAgent_db/context.json", "w", encoding="utf-8") as f:
        json.dump(context_data, f, indent=2)
    print(f"✅ context.json saved ({len(context_data)} entries)")
    
    # Re-index Elasticsearch if available
    if ES_AVAILABLE:
        print("\nRe-indexing Elasticsearch BM25...")
        try:
            es_bm25 = create_elasticsearch_bm25_index(vector_db)
            if es_bm25:
                print("✅ Elasticsearch re-indexed with contextualized content")
        except Exception as e:
            print(f"⚠️ ES indexing failed: {e}")
    
    # Print some sample contextualizations
    print("\n" + "=" * 60)
    print("SAMPLE CONTEXTUALIZATIONS")
    print("=" * 60)
    text_items = [m for m in vector_db.metadata if "chunk_id" in m]
    for i, item in enumerate(text_items[:3]):
        ctx = item["contextualized_content"]
        orig = item["original_content"]
        # Extract just the context preamble (before the original content)
        preamble = ctx[:len(ctx) - len(orig)].strip() if ctx != orig else "(no context)"
        print(f"\n--- Chunk {i+1} (doc: {item['doc_id'][:30]}) ---")
        print(f"  Context: {preamble[:200]}")
        print(f"  Content: {orig[:150]}...")
    
    print("\n" + "=" * 60)
    print("✅ CONTEXTUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
