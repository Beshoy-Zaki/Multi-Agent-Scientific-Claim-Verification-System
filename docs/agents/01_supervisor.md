# Supervisor Agent
- **Purpose:** Manages the overall investigation, tracks state across claims, and dynamically determines whether more evidence is needed or if a claim can be finalized.
- **Inputs:** Claim statuses, Critic feedback, investigation history.
- **Outputs:** Next system action (e.g. Search, RAG extraction, Critic review, Finalize).
