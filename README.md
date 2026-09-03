# Multi-Agent Scientific Claim Verification System (MASCV)

[![CI Pipeline](https://github.com/Beshoy-Zaki/Multi-Agent-Scientific-Claim-Verification-System/actions/workflows/ci.yml/badge.svg)](https://github.com/Beshoy-Zaki/Multi-Agent-Scientific-Claim-Verification-System/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](pyproject.toml)

> **An Evidence-Grounded Multi-Agent System for Adversarial Analysis of Scientific Claims**

---

## 📌 Executive Summary

Scientific understanding requires more than document summarization. Research papers communicate findings through a combination of experimental results, methodological choices, interpretations, and claims. A claim may appear convincing inside the original paper while the broader literature contains:
- Independent studies that support the result
- Failed replications or weaker findings
- Unstated experimental limitations or alternative explanations
- Methodological criticisms or contradictory findings

**MASCV** is an agentic artificial intelligence system designed to analyze and critically evaluate scientific claims made in research papers. Rather than merely summarizing papers, MASCV:
1. Extracts meaningful and testable propositions.
2. Formulates adversarial literature queries across academic and web sources.
3. Retrieves and bundles claim-aware evidence units (text, tables, experimental contexts).
4. Pits a **Support Agent** (proponent) against an **Attack Agent** (adversary) in structured dialectic debate.
5. Employs a **Critic Agent** to validate citations, inspect premise-conclusion validity, and detect overgeneralizations.
6. Uses an adaptive **Supervisor Agent** to trigger additional search cycles when evidence is insufficient.

---

## 🏗️ System Architecture

```text
Target Paper (PDF / LaTeX)
           │
           ▼
    Supervisor Agent ◄────────────────────────┐
           │                                  │
           ▼                                  │
     Claim Analyst                            │
           │                                  │
           ▼                                  │
   Paper Search Agent                         │
     (Adversarial)                            │
           │                                  │
           ├──────────────────────────┐       │
           ▼                          ▼       │
      Target Paper               Web Papers   │
           │                          │       │
           └────────────┬─────────────┘       │
                        ▼                     │
               RAG / Evidence Agent           │
           (Claim-Aware & Agentic)            │
                        │                     │
                        ▼                     │
                  Evidence Store              │
                        │                     │
              ┌─────────┴─────────┐           │
              ▼                   ▼           │
        Support Agent        Attack Agent     │
              │                   │           │
              └─────────┬─────────┘           │
                        ▼                     │
                  Critic Agent                │
                        │                     │
              ┌─────────┴─────────┐           │
              ▼                   ▼           │
         Final Verdict       Need More        │
        / Full Report         Evidence        │
                                  └───────────┘
```

---

## 🤖 The Seven Specialized Agents

| Agent | Core Question / Responsibility |
| :--- | :--- |
| **1. Supervisor Agent** | *"What should happen next?"* Manages investigation state, monitors confidence, and controls iteration loops. |
| **2. Claim Analyst** | *"What exactly are we testing?"* Formalizes vague statements into structured testable propositions. |
| **3. Paper Search Agent** | *"What other research exists?"* Executes adversarial queries (proponent vs. opponent). |
| **4. RAG / Evidence Agent** | *"What evidence inside these papers is relevant?"* Performs claim-aware parsing and bundles evidence. |
| **5. Support Agent** | *"Why might this claim be true?"* Constructs the strongest evidence-grounded affirmative argument. |
| **6. Attack Agent** | *"Why might this claim be wrong or overstated?"* Uncovers contradictions, limitations, and biases. |
| **7. Critic Agent** | *"Which evidence and arguments actually hold up?"* Validates citations, comparisons, and issues verdicts. |

---

## 📂 Repository Structure

```
Multi-Agent-Scientific-Claim-Verification-System/
├── .github/
│   ├── ISSUE_TEMPLATE/               # GitHub issue templates for bug reports & features
│   ├── workflows/                    # GitHub Actions CI & documentation workflows
│   └── PULL_REQUEST_TEMPLATE.md      # Standard PR template
├── assets/
│   └── diagrams/                     # Architectural and workflow schematics
├── config/
│   ├── agents/                       # Agent prompt templates, model selections, & configs
│   ├── evaluation/                   # Baseline and ablation matrix configs
│   ├── default_config.yaml           # Global system hyperparameters and loop thresholds
│   └── logging.yaml                  # Application logger settings
├── data/
│   ├── benchmarks/                   # Standard scientific verification datasets (SciFact, etc.)
│   ├── raw_papers/                   # Ingested PDF & source documents
│   ├── processed_cache/              # Cached parsed papers and embeddings
│   └── sample_inputs/                # Sample papers for pipeline demonstration
├── docs/
│   ├── architecture/                 # High-level architecture & interaction loop specifications
│   ├── agents/                       # Detailed specifications for all 7 agents
│   ├── rag_and_evidence/             # Agentic chunking and evidence bundle documentation
│   ├── evaluation/                   # Evaluation metrics, baselines, and ablations
│   └── api/                          # REST API specification
├── scripts/
│   ├── setup_env.sh / .bat           # Automated environment setup scripts
│   ├── run_pipeline.py               # Command-line pipeline execution script
│   ├── run_evaluation.py             # Evaluation benchmark runner
│   ├── run_ablations.py              # Ablation experiments runner
│   └── export_report.py              # Scientific report exporter (MD/LaTeX/JSON)
├── src/
│   └── mascv/                        # Core MASCV Python Package
│       ├── agents/                   # Implementations of the 7 specialized agents
│       ├── core/                     # Workflow state machine, graph orchestration, exceptions
│       ├── models/                   # Pydantic schemas (Claims, EvidenceBundles, Debates, Verdicts)
│       ├── rag/                      # Document parsers, agentic chunking, embeddings, hybrid retrieval
│       ├── search/                   # Academic (arXiv, Semantic Scholar, Crossref) and Web clients
│       ├── storage/                  # EvidenceStore, vector database managers, provenance graph
│       ├── evaluation/               # Metrics, single-agent/standard RAG baselines, ablations
│       ├── reporting/                # JSON, Markdown, and LaTeX report generators
│       └── utils/                    # Config loaders, logging, text processing utilities
├── ui/
│   ├── backend/                      # FastAPI REST application exposing backend routes
│   └── frontend/                     # Streamlit research dashboard
├── tests/
│   ├── conftest.py                   # Pytest fixtures and mock state data
│   ├── unit/                         # Unit tests for agents, RAG, models, and storage
│   └── integration/                  # End-to-end and iterative feedback loop integration tests
├── .env.example                      # Template for API keys and environment variables
├── .gitignore                        # Standard Python, cache, and artifact exclusions
├── CITATION.cff                      # Academic citation metadata
├── LICENSE                           # MIT License
├── Makefile                          # Convenient development commands
├── pyproject.toml                    # Modern PEP 621 package and dependency definitions
└── requirements.txt                  # Pinned dependency requirements
```

---

## 🔬 Evidence Relationships & Verdict Taxonomy

### Evidence Classifications
- **`SUPPORTS`**: Confirms claim findings under comparable conditions.
- **`CONTRADICTS`**: Conflicts with the claim under comparable conditions.
- **`QUALIFIES`**: Indicates the claim holds only within narrow boundary constraints.
- **`REPLICATES`**: Independent reproduction of results on identical or analogous setups.
- **`CHALLENGES`**: Identifies methodological flaws or conceptual weaknesses.
- **`ALTERNATIVE`**: Provides an alternative hypothesis explaining the reported result.

### Verdict Outputs
- **`Supported`**: Literature demonstrates robust, reproducible affirmative evidence.
- **`Partially Supported`**: Valid under specific conditions, but lacks claimed generality.
- **`Unsupported`**: Substantial contradictory findings or severe methodological flaws.
- **`Inconclusive`**: Available literature is sparse, inaccessible, or contradictory without clear resolution.

---

## 📊 Evaluation & Baselines

MASCV includes an empirical evaluation framework comparing against:
1. **Single-Agent Baseline:** Monolithic LLM with literature search and RAG tools.
2. **Standard RAG Baseline:** Traditional fixed-token sliding window chunking with similarity retrieval.
3. **Ablation Studies (8 configurations):** Systematic evaluation isolating the effect of adversarial debate, critic validation, claim-aware chunking, and the adaptive supervisor loop.

**Key Metrics:** Claim Verification Accuracy, Evidence Retrieval Quality (Precision/Recall), Citation Faithfulness, Contradiction Detection, Overgeneralization Detection, Inference Cost, and Latency.

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Beshoy-Zaki/Multi-Agent-Scientific-Claim-Verification-System.git
cd Multi-Agent-Scientific-Claim-Verification-System

# Create and activate virtual environment
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

# Install dependencies in editable mode
pip install -e ".[dev]"
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

### 3. Launch Dashboard or CLI
```bash
# Launch Streamlit Research Dashboard
streamlit run ui/frontend/streamlit_app.py

# Run pipeline via CLI
python -m mascv --paper "data/sample_inputs/sample_paper.pdf"
```

### 4. Run Tests
```bash
pytest tests/
```

---

## 📜 Citation

If you reference or build upon this project, please cite:
```bibtex
@misc{zaki2026mascv,
  author = {Beshoy Zaki},
  title = {Multi-Agent Scientific Claim Verification System (MASCV): An Evidence-Grounded Multi-Agent System for Adversarial Analysis of Scientific Claims},
  year = {2026},
  url = {https://github.com/Beshoy-Zaki/Multi-Agent-Scientific-Claim-Verification-System}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
