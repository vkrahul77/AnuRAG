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
|-- README.md                 # This file
|-- CITATION.cff              # Citation metadata (GitHub "Cite this repository")
|-- LICENSE                   # Apache License 2.0
|-- AnURAG_flowchart.tex      # LaTeX TikZ diagram of the full pipeline
|-- .gitignore
|
|-- QUICKSTART.md             # 5-minute getting started
|-- SETUP_GUIDE.md            # Detailed setup & architecture
|-- API_CONFIGURATION.md      # API key configuration guide
+-- ELASTICSEARCH_SETUP.md    # Optional BM25 search backend
```

> **Note:** The full implementation source code (RAG pipeline, agent loop,
> gm/ID-based design-space exploration) will be released after the
> accompanying journal paper is published. For early access, please
> contact the author at vikas@hawaii.edu.

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

## Documentation

| Guide | Description |
|-------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | 5-minute setup |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Full architecture & walkthrough |
| [API_CONFIGURATION.md](API_CONFIGURATION.md) | API keys & model selection |
| [ELASTICSEARCH_SETUP.md](ELASTICSEARCH_SETUP.md) | Optional BM25 hybrid search |

---

## Attribution

> This repository accompanies work presented at the **NSF-funded Analog Design Automation Workshop 2026**.
> The full manuscript is in preparation. If you use ideas, methodology, or code from this
> repository, please cite the work using the information below or contact the author.

### Citing This Work

If you use AnuRAG in your research, please cite:

```bibtex
@software{kumar2026anurag,
  author       = {Kumar, Vikas},
  title        = {{AnuRAG}: Retrieval-Augmented Analog Design Assistant},
  year         = {2026},
  url          = {https://github.com/vkrahul77/AnuRAG},
  note         = {Presented at the NSF-funded Analog Design Automation Workshop 2026.
                  Manuscript in preparation.}
}
```

GitHub will also show a **"Cite this repository"** button automatically from the [CITATION.cff](CITATION.cff) file.

---

## References

- AnuRAG Flowchart: see [AnURAG_flowchart.tex](AnURAG_flowchart.tex) (compile with `pdflatex`)
- Google Gemini: [Documentation](https://ai.google.dev/docs)
- Contextual Retrieval: [Anthropic Blog](https://www.anthropic.com/news/contextual-retrieval)

---

## License

Copyright 2026 Vikas Kumar. Licensed under the [Apache License 2.0](LICENSE).

> **Note:** The full implementation code will be released after journal publication.
> This public repository currently contains architecture documentation, example
> workflows, and configuration templates. For access to the full pipeline code,
> please contact the author at vikas@hawaii.edu.
