# System Architecture

The Multi-Agent Scientific Claim Verification System (MASCV) is an evidence-grounded multi-agent artificial intelligence framework engineered for adversarial analysis of scientific claims in academic literature.

## Core Conceptual Flow
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
