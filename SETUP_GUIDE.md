# MuaLLM-Gemini: Complete Setup & Architecture Guide

## 📋 Table of Contents
1. [Conda Environment Setup](#1-conda-environment-setup)
2. [Gemini API Key Setup](#2-gemini-api-key-setup)
3. [Which Model to Use](#3-which-model-to-use)
4. [Project Architecture](#4-project-architecture)
5. [Code Flow Explanation](#5-code-flow-explanation)
6. [Step-by-Step Usage](#6-step-by-step-usage)

---

## 1. Conda Environment Setup

### Step 1: Open Anaconda Prompt or Terminal

```bash
# Navigate to project directory
cd C:\Users\VikasKumar\Documents\github_repos\MuaLLM-Gemini
```

### Step 2: Create Environment from YAML

```bash
# Create the environment (this may take 5-10 minutes)
conda env create -f environment.yml
```

### Step 3: Activate the Environment

```bash
conda activate muallm-gemini
```

### Step 4: Verify Installation

```bash
python -c "import google.generativeai as genai; print('Gemini SDK installed successfully!')"
```

### Troubleshooting

If you encounter issues:
```bash
# Remove and recreate
conda env remove -n muallm-gemini
conda env create -f environment.yml

# Or update existing
conda env update -f environment.yml --prune
```

---

## 2. Gemini API Key Setup

### Step 1: Get Your API Key

1. Go to **Google AI Studio**: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Select a Google Cloud project (or create one)
5. **Copy the API key** (it looks like: `AIzaSy...`)

### Step 2: Create the .env File

In the project root folder (`MuaLLM-Gemini/`), create a file named `.env`:

```bash
# Windows (PowerShell)
cd C:\Users\VikasKumar\Documents\github_repos\MuaLLM-Gemini
notepad .env
```

### Step 3: Add Your API Key

Paste this into the `.env` file:

```
GOOGLE_API_KEY=AIzaSy_YOUR_ACTUAL_API_KEY_HERE
```

**⚠️ Important**: 
- Replace `AIzaSy_YOUR_ACTUAL_API_KEY_HERE` with your actual key
- No quotes around the key
- No spaces around the `=` sign

### Step 4: Verify the Key Works

```bash
cd gemini\tools
python -c "
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
print(f'API Key loaded: {api_key[:10]}...' if api_key else 'ERROR: No API key found!')

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content('Say hello!')
    print(f'Gemini says: {response.text}')
"
```

---

## 3. Which Model to Use

### Recommended: `gemini-2.0-flash`

This project uses **Gemini 2.0 Flash** because:

| Feature | Gemini 2.0 Flash | Gemini 1.5 Pro |
|---------|-----------------|----------------|
| **Speed** | ⚡ Very Fast | 🐢 Slower |
| **Cost** | 💰 Cheap ($0.075/1M input) | 💸 Expensive ($3.50/1M) |
| **Vision** | ✅ Yes | ✅ Yes |
| **Context** | 1M tokens | 2M tokens |
| **Best For** | This project | Complex reasoning |

### Models Used in the Code

```python
# Text generation & reasoning (agent.py, main.py)
model = genai.GenerativeModel('gemini-2.0-flash')

# Embeddings for vector search (search.py)
genai.embed_content(model="models/text-embedding-004", ...)

# Image analysis (fullcontext.py, search.py)
model = genai.GenerativeModel('gemini-2.0-flash')  # Has vision capabilities
```

### Alternative Models

If you need different capabilities:
```python
# For more complex reasoning (costs more)
model = genai.GenerativeModel('gemini-1.5-pro')

# For simple tasks (even cheaper)
model = genai.GenerativeModel('gemini-1.5-flash')
```

---

## 4. Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
│              "Which BGR circuit has lowest power?"               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      main.py (Entry Point)                       │
│  - Parses command line arguments                                 │
│  - Routes to appropriate function                                │
│  - Initializes the Agent                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    agent.py (ReAct Agent)                        │
│  - Implements Thought → Action → Observation loop                │
│  - Uses Gemini for reasoning                                     │
│  - Calls tools based on the query                                │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌─────────────────┐ ┌──────────────────┐
│  load_titles.py  │ │   search.py     │ │ fullcontext.py   │
│                  │ │                 │ │                  │
│ Returns list of  │ │ Hybrid search:  │ │ Full document    │
│ paper titles     │ │ - Vector search │ │ Q&A using        │
│                  │ │ - BM25 search   │ │ complete PDF     │
│                  │ │ - Reranking     │ │ context          │
└──────────────────┘ └─────────────────┘ └──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vector Database                               │
│  - Stores embeddings (Gemini text-embedding-004)                 │
│  - Stores metadata (doc_id, chunk_id, content)                   │
│  - Stores contextualized descriptions                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 pdf2json_chunked.py                              │
│  - Extracts text from PDFs                                       │
│  - Extracts images from PDFs                                     │
│  - Creates overlapping chunks                                    │
│  - Saves to documents.json                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PDF Papers                                  │
│         C:\Users\VikasKumar\Documents\github_repos\Filtered_01_06│
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Code Flow Explanation

### File-by-File Breakdown

#### 📄 `main.py` - The Entry Point
```
ROLE: Command center of the application

WHAT IT DOES:
1. Parses command line arguments (--process_papers, --build_db, --query)
2. Initializes the Gemini-powered Agent
3. Routes user requests to appropriate handlers
4. Provides interactive mode for Q&A

KEY FUNCTIONS:
- main()           → CLI argument parsing
- query()          → ReAct agent loop
- process_papers() → Triggers PDF processing
- build_database() → Triggers embedding creation
```

#### 🤖 `agent.py` - The Brain (ReAct Agent)
```
ROLE: Intelligent reasoning engine

WHAT IT DOES:
1. Receives user question
2. THINKS about what to do
3. Takes ACTION (calls a tool)
4. OBSERVES the result
5. Repeats until answer is found

KEY COMPONENTS:
- GeminiAgent class  → Wraps Gemini API
- query() function   → Main reasoning loop
- known_actions      → Available tools (search_db, load_titles)

EXAMPLE FLOW:
Question: "Which BGR has lowest power?"
  → Thought: "I need to find all BGR circuits first"
  → Action: load_titles: True
  → Observation: [list of paper titles]
  → Thought: "Now I'll search each paper for power consumption"
  → Action: search_db: "power consumption BGR circuit Paper1"
  → Observation: "100nW power consumption..."
  → ... (continues until answer is complete)
```

#### 🔍 `search.py` - Hybrid Search Engine
```
ROLE: Finds relevant information from the database

WHAT IT DOES:
1. Semantic Search: Uses Gemini embeddings to find similar content
2. BM25 Search: Keyword-based search using Elasticsearch
3. Fusion: Combines both results with weighted scores
4. Reranking: Uses Cohere to reorder by relevance

KEY CLASSES:
- ContextualVectorDB    → Manages embeddings and vector search
- ElasticSearchBM25     → Manages keyword search
- retrieve_advanced()   → Hybrid retrieval
- retrieve_rerank()     → Adds Cohere reranking

EMBEDDINGS USED:
- text-embedding-004 (768 dimensions)
- Contextual: Each chunk gets document-level context added
```

#### 📑 `pdf2json_chunked.py` - PDF Processor
```
ROLE: Converts PDFs into searchable data

WHAT IT DOES:
1. Extracts text from PDFs (PyPDF2 + unstructured)
2. Extracts images from PDFs (PyMuPDF)
3. Chunks text into ~1000 character pieces with overlap
4. Detects equations and formulas
5. Saves everything to documents.json

OUTPUT FORMAT:
{
  "doc_id": "doc_1",
  "content": "Full extracted text...",
  "chunks": [
    {"chunk_id": "doc_1_chunk_0", "content": "First chunk..."},
    {"chunk_id": "doc_1_chunk_1", "content": "Second chunk..."}
  ],
  "images": [
    {"image_id": "doc_1_image_1", "path": "../finalAgent_db/images/image_1.png"}
  ],
  "pdf_path": "path/to/original.pdf"
}
```

#### 💬 `messages.py` - System Prompts
```
ROLE: Defines how the AI agent behaves

WHAT IT CONTAINS:
- system_message: Instructions for the ReAct agent
- Action definitions: What tools are available
- Example sessions: Shows the agent how to respond
- Specialized prompts: For images, equations, graphs
```

#### 📖 `fullcontext.py` - Full Document Q&A
```
ROLE: Answers questions using complete document context

WHAT IT DOES:
1. Loads entire PDF content
2. Sends to Gemini with the question
3. Returns answer based on full context

USE CASE:
When you need information that spans the whole document,
not just a specific chunk (e.g., "What is the title?")
```

#### 📚 `load_titles.py` - Title Manager
```
ROLE: Manages paper titles for the agent

WHAT IT DOES:
1. Loads titles.json
2. Returns list of all paper titles
3. Used by agent to know what papers are available
```

---

## 6. Step-by-Step Usage

### Phase 1: Setup (One Time)

```bash
# 1. Create conda environment
conda env create -f environment.yml

# 2. Activate environment
conda activate muallm-gemini

# 3. Create .env file with API key
# (See Section 2 above)
```

### Phase 2: Process Papers (One Time per Dataset)

```bash
cd gemini\tools

# Process all PDFs in the database folder
python main.py --process_papers "C:\Users\VikasKumar\Documents\github_repos\Filtered_01_06"
```

This creates:
- `finalAgent_db/documents.json` - Extracted text and chunks
- `finalAgent_db/images/` - Extracted images

### Phase 3: Build Vector Database (One Time)

```bash
python main.py --build_db
```

This creates:
- `finalAgent_db/base_db/vector_db.pkl` - Embeddings
- `finalAgent_db/context.json` - Contextualized content
- `finalAgent_db/titles.json` - Paper titles

### Phase 4: Query the System (Anytime)

```bash
# Interactive mode
python main.py --interactive

# Single query
python main.py --query "What is the lowest power bandgap reference?"

# Using the agent directly
python agent.py --query "Show me a PTAT circuit schematic"
```

---

## 🎯 Quick Reference

| Command | Purpose |
|---------|---------|
| `conda activate muallm-gemini` | Activate environment |
| `python main.py --process_papers PATH` | Process PDFs |
| `python main.py --build_db` | Create embeddings |
| `python main.py --interactive` | Q&A mode |
| `python main.py --query "..."` | Single question |

---

## ❓ Common Questions

**Q: How long does processing take?**
A: ~5-10 seconds per PDF for text extraction, ~2-5 seconds per chunk for embedding.

**Q: Can I add more papers later?**
A: Yes! Run `--process_papers` again, then `--build_db` to update.

**Q: What if I hit rate limits?**
A: The code has automatic retry with exponential backoff. Wait and retry.

**Q: How do I change the model?**
A: Edit the model name in `agent.py`, `search.py`, and `fullcontext.py`.
