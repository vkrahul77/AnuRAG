# AnuRAG

**An**alog Design Framework with **u**nified **R**etrieval-**A**ugmented **G**eneration

A two-stage "white-box" assistant for analog integrated-circuit design that **augments the human designer** rather than replacing them â€” combining retrieval-augmented generation with physics-grounded reasoning.

---

## Overview

AnuRAG addresses the fundamental challenge of analog IC design automation: high-dimensional continuous design spaces coupled with the need for rigorous physical intuition. Unlike black-box approaches (RL, Bayesian Optimization) that require hundreds of SPICE simulations and yield uninterpretable decisions, AnuRAG operates as a **transparent, literature-grounded assistant**.

### Two-Stage Architecture

The framework diagram ([AnURAG_flowchart.tex](AnURAG_flowchart.tex)) illustrates the full pipeline:

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚              OFFLINE â€” Knowledge-Base Construction              â”‚
â”‚  Raw Sources â”€â”€â–º Embedding Pipeline â”€â”€â–º Vector Database         â”‚
â”‚  (Papers, Textbooks,     (text-embedding-004,       (ChromaDB/  â”‚
â”‚   Lecture Notes)          Google Vision captions)     FAISS)     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â”‚ retrieval
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
          â–¼                                       â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ STAGE 1              â”‚                â”‚ STAGE 2                  â”‚
â”‚ Topology Selection   â”‚                â”‚ Design-Space Exploration â”‚
â”‚                      â”‚                â”‚                          â”‚
â”‚ User Specs           â”‚                â”‚ Trade-off Specs          â”‚
â”‚   â–¼                  â”‚  ranked        â”‚   â–¼                      â”‚
â”‚ RAG Engine â”€â”€â–º Rankedâ”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–ºâ”‚ RAG Engine (Eq. Code Gen)â”‚
â”‚ (LLM + Vector        â”‚  topologies    â”‚   â–¼                      â”‚
â”‚  Retrieval +          â”‚                â”‚ Pareto: 10,000+ pts     â”‚
â”‚  Feasibility Checks)  â”‚                â”‚   â–¼                      â”‚
â”‚                      â”‚                â”‚ Sparse SPICE Verificationâ”‚
â”‚ Output:              â”‚                â”‚   â–¼                      â”‚
â”‚  Folded Cascode,     â”‚                â”‚ Error < 10%? â”€â”€â–ºYesâ”€â”€â–º  â”‚
â”‚  Two-Stage OTA, ...  â”‚                â”‚   â”‚                Outputâ”‚
â”‚  with citations      â”‚                â”‚   Noâ”€â”€â–º Re-calibrate     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

**Stage 1 â€” Topology Selection**: A ReAct-style agent performs hybrid retrieval (semantic + BM25) against a curated knowledge base. Physics-grounded feasibility checks (voltage headroom, swing constraints, intrinsic-gain limits) prune infeasible candidates, returning a ranked shortlist with literature citations.

**Stage 2 â€” Design-Space Exploration**: The RAG engine switches to equation-driven code generation, emitting executable Python scripts that sweep gm/ID look-up tables, producing Pareto frontiers over 10,000+ sizing points â€” **without a single SPICE call**. A sparse verification loop samples ~5 anchor points for SPICE simulation; if analytical-vs-simulated error exceeds 10%, the framework auto-recalibrates.

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
â”œâ”€â”€ .env.example              # Template â€” fill in your API keys
â”œâ”€â”€ .gitignore
â”œâ”€â”€ README.md                 # This file
â”œâ”€â”€ AnURAG_flowchart.tex      # LaTeX TikZ diagram of the full pipeline
â”œâ”€â”€ requirements.txt          # pip dependencies
â”œâ”€â”€ environment.yml           # Conda environment spec
â”œâ”€â”€ run_anurag.bat            # Windows launcher
â”œâ”€â”€ setup_conda_env.bat       # Windows conda setup
â”œâ”€â”€ setup_conda_env.sh        # Linux/Mac conda setup
â”‚
â”œâ”€â”€ QUICKSTART.md             # 5-minute getting started
â”œâ”€â”€ SETUP_GUIDE.md            # Detailed setup & architecture
â”œâ”€â”€ API_CONFIGURATION.md      # API key configuration guide
â”œâ”€â”€ ELASTICSEARCH_SETUP.md    # Optional BM25 search backend
â”‚
â””â”€â”€ gemini/
    â”œâ”€â”€ Design_Question.txt   # Example design prompt
    â”œâ”€â”€ finalAgent_db/        # Vector DB (auto-generated, git-ignored)
    â”‚   â””â”€â”€ README.md
    â””â”€â”€ tools/
        â”œâ”€â”€ main.py           # Entry point â€” CLI for all workflows
        â”œâ”€â”€ agent.py          # ReAct agent loop
        â”œâ”€â”€ search.py         # Hybrid search (semantic + BM25 + rerank)
        â”œâ”€â”€ messages.py       # Stage 1 & 2 system prompts
        â”œâ”€â”€ fullcontext.py    # Full-document Q&A with vision
        â”œâ”€â”€ load_titles.py    # Paper title management
        â”œâ”€â”€ pdf2json_chunked.py  # Async PDF â†’ chunked JSON
        â”œâ”€â”€ config.py         # Centralized configuration
        â”œâ”€â”€ llm_provider.py   # LLM abstraction (Gemini / Claude)
        â”œâ”€â”€ run_contextualize.py # Batch contextual embedding
        â””â”€â”€ web_scraper.py    # Web/arXiv scraping utilities
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/vkrahul77/AnuRAG.git
cd AnuRAG
```

### 2. Environment Setup

**Option A â€” Conda (recommended)**:
```bash
conda env create -f environment.yml
conda activate anurag
```

**Option B â€” pip**:
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
- `GOOGLE_API_KEY` â€” required for embeddings & vision (get from [Google AI Studio](https://makersuite.google.com/app/apikey))
- `LLM_PROVIDER` â€” set to `gemini` or `claude`
- `ANTHROPIC_API_KEY` â€” only if using Claude

Optional:
- `COHERE_API_KEY` â€” enables reranking for better retrieval precision
- Elasticsearch â€” see [ELASTICSEARCH_SETUP.md](ELASTICSEARCH_SETUP.md) for BM25 hybrid search

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

### Stage 1 â€” Topology Selection

```
User: I need an OTA with Gain â‰¥ 70 dB, GBW â‰¥ 500 MHz, CL = 500 fF,
      Swing â‰¥ 1.0 Vpp,diff in IHP SG13G2 130nm.

AnuRAG: Based on retrieval from 406 sources (402 JSSC papers + 4 textbooks):

  1. Folded Cascode OTA â€” meets gain/swing, proven at 130nm [Smith2018, ...]
  2. Two-Stage Miller OTA â€” highest gain margin, needs compensation [Lee2021, ...]
  3. Telescopic OTA â€” best GBW/power but limited swing [Chen2019, ...]

  Infeasible: Single-stage CS (gain < 70 dB at this node)
```

### Stage 2 â€” Design-Space Exploration

```
User: Explore the Folded Cascode design space with gm/ID methodology.

AnuRAG: [Generates executable Python script]
  â†’ Sweeps gm/ID âˆˆ [5, 25], L, bias points
  â†’ Produces Pareto frontier: Power vs. Bandwidth (10,000+ points)
  â†’ Sparse SPICE verification at 5 anchor points
  â†’ Analytical vs. simulated error: 8.3% âœ“ (< 10% threshold)
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
