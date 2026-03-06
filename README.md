# AnuRAG

**An**alog Design Framework with **u**nified **R**etrieval-**A**ugmented **G**eneration

A two-stage "white-box" assistant for analog integrated-circuit design that **augments the human designer** rather than replacing them -- combining retrieval-augmented generation with physics-grounded reasoning.

---

## Overview

AnuRAG addresses the fundamental challenge of analog IC design automation: high-dimensional continuous design spaces coupled with the need for rigorous physical intuition. Unlike black-box approaches (RL, Bayesian Optimization) that require hundreds of SPICE simulations and yield uninterpretable decisions, AnuRAG operates as a **transparent, literature-grounded assistant**.

### Two-Stage Architecture

The framework diagram ([AnURAG_flowchart.tex](AnURAG_flowchart.tex)) illustrates the full pipeline:

```
+-----------------------------------------------------------------+
|              OFFLINE -- Knowledge-Base Construction              |
|  Raw Sources --> Embedding Pipeline --> Vector Database         |
|  (Papers, Textbooks,     (text-embedding-004,       (ChromaDB/  |
|   Lecture Notes)          Google Vision captions)     FAISS)     |
+-----------------------------------------------------------------+
                              | retrieval
          +-------------------+-------------------+
          v                                       v
+---------------------+                +-------------------------+
| STAGE 1              |                | STAGE 2                  |
| Topology Selection   |                | Design-Space Exploration |
|                      |                |                          |
| User Specs           |                | Trade-off Specs          |
|   v                  |  ranked        |   v                      |
| RAG Engine --> Ranked|--------------->| RAG Engine (Eq. Code Gen)|
| (LLM + Vector        |  topologies    |   v                      |
|  Retrieval +          |                | Pareto: 10,000+ pts     |
|  Feasibility Checks)  |                |   v                      |
|                      |                | Sparse SPICE Verification|
| Output:              |                |   v                      |
|  Folded Cascode,     |                | Error < 10%? -->Yes-->  |
|  Two-Stage OTA, ...  |                |   |                Output|
|  with citations      |                |   No--> Re-calibrate     |
+---------------------+                +-------------------------+
```

**Stage 1 -- Topology Selection**: A ReAct-style agent performs hybrid retrieval (semantic + BM25) against a curated knowledge base. Physics-grounded feasibility checks (voltage headroom, swing constraints, intrinsic-gain limits) prune infeasible candidates, returning a ranked shortlist with literature citations.

**Stage 2 -- Design-Space Exploration**: The RAG engine switches to equation-driven code generation, emitting executable Python scripts that sweep gm/ID look-up tables, producing Pareto frontiers over 10,000+ sizing points -- **without a single SPICE call**. A sparse verification loop samples ~5 anchor points for SPICE simulation; if analytical-vs-simulated error exceeds 10%, the framework auto-recalibrates.

### LLM Provider Support

AnuRAG supports multiple LLM backends via a unified abstraction layer:

| Provider | Models | Use Case |
|----------|--------|----------|
| **Google Gemini** | gemini-2.0-flash, gemini-2.5-pro, etc. | Default; also handles embeddings & vision |
| **Anthropic Claude** | claude-sonnet-4-20250514, claude-opus-4-20250514 | Alternative reasoning engine |

Switch providers with a single environment variable (`LLM_PROVIDER`).

---

## Project Structure

```
AnuRAG/
|-- .env.example              # Template -- fill in your API keys
|-- .gitignore
|-- README.md                 # This file
|-- AnURAG_flowchart.tex      # LaTeX TikZ diagram of the full pipeline
|-- requirements.txt          # pip dependencies
|-- environment.yml           # Conda environment spec
|-- run_anurag.bat            # Windows launcher
|-- setup_conda_env.bat       # Windows conda setup
|-- setup_conda_env.sh        # Linux/Mac conda setup
|
|-- QUICKSTART.md             # 5-minute getting started
|-- SETUP_GUIDE.md            # Detailed setup & architecture
|-- API_CONFIGURATION.md      # API key configuration guide
|-- ELASTICSEARCH_SETUP.md    # Optional BM25 search backend
|
+-- gemini/
    |-- Design_Question.txt   # Example design prompt
    |-- finalAgent_db/        # Vector DB (auto-generated, git-ignored)
    |   +-- README.md
    +-- tools/
        |-- main.py           # Entry point -- CLI for all workflows
        |-- agent.py          # ReAct agent loop
        |-- search.py         # Hybrid search (semantic + BM25 + rerank)
        |-- messages.py       # Stage 1 & 2 system prompts
        |-- fullcontext.py    # Full-document Q&A with vision
        |-- load_titles.py    # Paper title management
        |-- pdf2json_chunked.py  # Async PDF -> chunked JSON
        |-- config.py         # Centralized configuration
        |-- llm_provider.py   # LLM abstraction (Gemini / Claude)
        |-- run_contextualize.py # Batch contextual embedding
        +-- web_scraper.py    # Web/arXiv scraping utilities
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/vkrahul77/AnuRAG.git
cd AnuRAG
```

### 2. Environment Setup

**Option A -- Conda (recommended)**:
```bash
conda env create -f environment.yml
conda activate anurag
```

**Option B -- pip**:
```bash
python -m venv venv
# Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

At minimum you need:
- `GOOGLE_API_KEY` -- required for embeddings & vision (get from [Google AI Studio](https://makersuite.google.com/app/apikey))
- `LLM_PROVIDER` -- set to `gemini` or `claude`
- `ANTHROPIC_API_KEY` -- only if using Claude

Optional:
- `COHERE_API_KEY` -- enables reranking for better retrieval precision
- Elasticsearch -- see [ELASTICSEARCH_SETUP.md](ELASTICSEARCH_SETUP.md) for BM25 hybrid search

### 4. Process Your Papers

```bash
cd gemini/tools

# Process PDFs from a directory
python main.py --process_papers "/path/to/your/papers"

# Build the vector database
python main.py --build_db

# (Optional) Add contextual embeddings for better retrieval
python run_contextualize.py
```

### 5. Query

```bash
# Single query
python main.py --query "Which topology achieves >70dB gain with 500MHz GBW?"

# Interactive mode
python main.py --interactive
```

---

## Example Workflow

### Stage 1 -- Topology Selection

```
User: I need an OTA with Gain >= 70 dB, GBW >= 500 MHz, CL = 500 fF,
      Swing >= 1.0 Vpp,diff in IHP SG13G2 130nm.

AnuRAG: Based on retrieval from 406 sources (402 JSSC papers + 4 textbooks):

  1. Folded Cascode OTA -- meets gain/swing, proven at 130nm [Smith2018, ...]
  2. Two-Stage Miller OTA -- highest gain margin, needs compensation [Lee2021, ...]
  3. Telescopic OTA -- best GBW/power but limited swing [Chen2019, ...]

  Infeasible: Single-stage CS (gain < 70 dB at this node)
```

### Stage 2 -- Design-Space Exploration

```
User: Explore the Folded Cascode design space with gm/ID methodology.

AnuRAG: [Generates executable Python script]
  -> Sweeps gm/ID  in  [5, 25], L, bias points
  -> Produces Pareto frontier: Power vs. Bandwidth (10,000+ points)
  -> Sparse SPICE verification at 5 anchor points
  -> Analytical vs. simulated error: 8.3% (ok) (< 10% threshold)
```

---

## Bringing Your Own Papers

AnuRAG is designed to work with **your own** paper collection. You supply PDFs (e.g., IEEE JSSC, ISSCC, textbook chapters), and the system builds a private knowledge base. No papers are included in this repository.

Recommended sources:
- IEEE JSSC / ISSCC papers in your research area
- Textbooks (Razavi, Allen & Holberg, Murmann gm/ID lectures, etc.)
- Your own lab's publications and notes

---

## Documentation

| Guide | Description |
|-------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | 5-minute setup |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Full architecture & walkthrough |
| [API_CONFIGURATION.md](API_CONFIGURATION.md) | API keys & model selection |
| [ELASTICSEARCH_SETUP.md](ELASTICSEARCH_SETUP.md) | Optional BM25 hybrid search |

---

## References

- AnuRAG Flowchart: see [AnURAG_flowchart.tex](AnURAG_flowchart.tex) (compile with `pdflatex`)
- Google Gemini: [Documentation](https://ai.google.dev/docs)
- Contextual Retrieval: [Anthropic Blog](https://www.anthropic.com/news/contextual-retrieval)

---

## License

MIT License
