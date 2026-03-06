# MuaLLM-Gemini Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Get Your Gemini API Key

1. Go to https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key

### Step 2: Create Environment File

Create a file named `.env` in the `MuaLLM-Gemini` folder:

```
GOOGLE_API_KEY=paste_your_key_here
```

### Step 3: Install Dependencies

Open a terminal in the `MuaLLM-Gemini` folder:

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install packages
pip install -r requirements.txt
```

### Step 4: Process Your Papers

```bash
cd gemini\tools

# Process papers from the Filtered_01_06 folder
python main.py --process_papers "C:\Users\VikasKumar\Documents\github_repos\Filtered_01_06"
```

This will extract text and images from all PDFs.

### Step 5: Build the Database

```bash
python main.py --build_db
```

This creates embeddings for all content (may take a while for many papers).

### Step 6: Ask Questions!

```bash
# Interactive mode
python main.py --interactive

# Or single query
python main.py --query "Which BGR circuit has the lowest power?"
```

---

## 📝 Example Workflow

```bash
# Terminal 1: Navigate to project
cd C:\Users\VikasKumar\Documents\github_repos\MuaLLM-Gemini

# Activate environment
venv\Scripts\activate

# Go to tools directory
cd gemini\tools

# Process 5 papers first (for testing)
python pdf2json_chunked.py --paper_path "C:\Users\VikasKumar\Documents\github_repos\Filtered_01_06" --output_path "../finalAgent_db/documents.json"

# Build database
python search.py --load_data

# Start asking questions!
python main.py --interactive
```

---

## 🔍 Sample Questions to Try

1. "What is the power consumption of the lowest power bandgap reference?"
2. "Show me the schematic of a PTAT current generator"
3. "Compare the temperature coefficients of voltage references in the database"
4. "Which paper uses a 0.5V supply voltage?"
5. "Explain the operation of the sub-bandgap reference circuit"

---

## ⚠️ Troubleshooting

**Error: API key not found**
- Make sure `.env` file exists in the root folder
- Check the key is correct

**Error: Module not found**
- Run `pip install -r requirements.txt` again
- Make sure virtual environment is activated

**Rate limit errors**
- The system will automatically retry
- For large datasets, process papers in batches

**No results found**
- Make sure you ran `--build_db` after processing papers
- Check that `documents.json` exists in `finalAgent_db/`

---

## 📁 Important Paths

- **Project Root**: `C:\Users\VikasKumar\Documents\github_repos\MuaLLM-Gemini`
- **PDF Papers**: `C:\Users\VikasKumar\Documents\github_repos\Filtered_01_06`
- **Database**: `gemini\finalAgent_db\`
- **Main Script**: `gemini\tools\main.py`

---

## 🎯 Next Steps

1. Process all papers (takes time but improves results)
2. Install Elasticsearch for better BM25 search
3. Get a Cohere API key for reranking
4. Try different types of questions

Enjoy using MuaLLM-Gemini! 🎉
