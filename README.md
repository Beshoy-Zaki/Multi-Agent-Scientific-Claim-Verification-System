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
│   └── agents/                       # Agent prompt templates, model configurations & hyperparameters
├── data/
│   ├── raw_papers/                   # Ingested PDF & source documents
│   ├── processed_cache/              # Cached parsed papers and extraction intermediate state
│   └── sample_inputs/                # Sample research papers (e.g. LoRA 2106.09685.pdf)
├── docs/
│   ├── architecture/                 # System architecture & iterative interaction loop specifications
│   ├── agents/                       # Detailed specifications for all 7 specialized agents
│   └── rag_and_evidence/             # Document parsing and hybrid retrieval documentation
├── scripts/
│   ├── setup_env.sh / .bat           # Environment setup scripts
│   ├── test_pipeline_live.py         # End-to-end multi-agent pipeline integration test runner
│   ├── test_paper_search.py          # Grounded literature search validation script
│   └── test_claim_extraction.py      # Claim extraction verification script
├── src/
│   └── mascv/                        # Core MASCV Python Package
│       ├── agents/                   # Implementations of the 7 specialized agents (Gemma 4 powered)
│       ├── core/                     # InvestigationState, workflow state machine, exceptions
│       ├── models/                   # Pydantic schemas (Claim, EvidenceBundle, Argument, Verdict)
│       ├── rag/                      # PDF parser, section extraction, chunking, hybrid retrieval
│       ├── tools/                    # Grounded evidence search & safe arithmetic validation
│       └── utils/                    # Google GenAI LLM client, config loader, logging, text processing
├── ui/
│   ├── backend/                      # FastAPI REST application exposing backend routes
│   └── frontend/                     # Streamlit research dashboard
├── tests/
│   ├── conftest.py                   # Pytest fixtures and mock state data
│   ├── unit/                         # Unit tests for all 7 agents, RAG pipeline, and schemas
│   └── integration/                  # End-to-end pipeline integration tests
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

## 🧪 Verification & Quality Assurance

MASCV is built with a test-driven, evidence-grounded verification workflow:
1. **Automated Unit Testing:** 44+ comprehensive unit tests covering all 7 agents, PDF parsing, hybrid BM25 + dense retrieval, safe AST arithmetic validation, and verdict synthesis.
2. **Adversarial Dialectic Architecture:** Pits the Support Agent directly against the Attack Agent using live Google Search Grounding to unearth real-world empirical limitations.
3. **Impartial Critic Adjudication:** Formulates defensible verdicts with confidence scores, identifying potential overgeneralizations and boundary conditions.
4. **End-to-End Live Integration:** Verified end-to-end across full research publications using Google Gemma 4 (`gemma-4-26b-a4b-it`).

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
Copy `.env.example` to `.env` and set your Google Gemini API key:
```bash
cp .env.example .env
```
In `.env`:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Run Live Pipeline or Research Dashboard
```bash
# Run live end-to-end multi-agent verification pipeline
python scripts/test_pipeline_live.py

# Launch Streamlit Research Dashboard
streamlit run ui/frontend/streamlit_app.py
```

### 4. Run Test Suite
```bash
pytest tests/
```

---

## 👥 Contributors & Core Team

MASCV was designed, engineered, and evaluated through the collaborative contributions of:

* **Beshoy Zaki** ([@Beshoy-Zaki](https://github.com/Beshoy-Zaki)) — System Architecture, PDF Parser, Supervisor & Orchestration Workflow, Live Streaming UI & FastAPI Dashboard, Non-Circularity Framework & Model Contracts.
* **Verina Hany** ([@VerinaHany21](https://github.com/VerinaHany21)) — Supervisor, Claim Analyst, and Paper Search Agent Frameworks, Academic Search Client Integration, Multi-Agent Schema Integration & Configuration.
* **Zeina Mostafa** ([@zeinamostafa254](https://github.com/zeinamostafa254)) — Initial multi-agent modules, evidence retrieval & verification tooling, testing suites, and core agent implementations.
* **Jana Kassem** ([@janaosmaneng-cyber](https://github.com/janaosmaneng-cyber)) — Adversarial Attack & Boundary Testing Agent, Critic Agent Adjudication & Evidence Verification Tools, Epistemic Rigor Guidelines.

---

## 📜 Citation

If you reference or build upon this project, please cite:
```bibtex
@misc{zaki2026mascv,
  author = {Beshoy Zaki and Jana Kassem and Verina Hany and Zeina Mostafa},
  title = {Multi-Agent Scientific Claim Verification System (MASCV): An Evidence-Grounded Multi-Agent System for Adversarial Analysis of Scientific Claims},
  year = {2026},
  url = {https://github.com/Beshoy-Zaki/Multi-Agent-Scientific-Claim-Verification-System}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

