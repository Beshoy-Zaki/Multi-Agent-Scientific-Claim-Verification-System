# Agent Interaction & State Loop

Unlike static sequential RAG pipelines, MASCV utilizes an adaptive state-driven iteration loop:

1. **Claim Extraction:** The Claim Analyst formalizes testable propositions.
2. **Literature Discovery:** The Paper Search Agent queries both confirmation and disconfirmation hypotheses.
3. **Evidence Extraction:** The RAG Agent parses both original and retrieved documents into evidence bundles.
4. **Adversarial Analysis:**
   - Proponent (Support Agent) builds the strongest affirmative case.
   - Adversary (Attack Agent) searches for counter-evidence, hidden assumptions, and limits.
5. **Judicial Critique:** Critic Agent examines citations, compares conditions, and identifies logical fallacies.
6. **Supervisor Decision:**
   - If confidence meets the threshold -> generate Final Report.
   - If evidence is conflicting, sparse, or overgeneralized -> dispatch targeted secondary searches back into the cycle.
