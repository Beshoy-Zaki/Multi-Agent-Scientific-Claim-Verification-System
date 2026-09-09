"""Critic Agent: Evaluates arguments, validates citations, and issues final verdicts."""

from typing import Any, Dict
from mascv.agents.base import BaseAgent
from mascv.models.argument import Argument
from mascv.models.verdict import Verdict
from pydantic import BaseModel, Field 
from langchain_openai import ChatOpenAI 
from langchain.agents import create_agent 
from dotenv import load_dotenv 
from tools import calculate 
from typing import Literal
import os 
 
 
load_dotenv() 
 
 
 
class CriticResult(BaseModel): 
    citation_grounding: str = Field(description="Explain whether the evidence supports the Support and Attack arguments.") 
    experimental_parity: str = Field(description="Explain whether the compared studies had sufficiently similar conditions.") 
    generalization: str = Field(description="Explain whether the claim can be generalized beyond the studied conditions.") 
    comparison: str = Field(description="Explain whether the Support and Attack evidence is genuinely comparable.") 
 
    verdict: Literal["Supported", "Partially Supported", "Unsupported", "Inconclusive"] = Field(description="Choose: Supported, Partially Supported, Unsupported, or Inconclusive.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the verdict from 0.0 to 1.0.") 
    key_issue: str = Field(description="State the most important issue affecting the verdict.") 
    winner: Literal["Support", "Attack", "Neither"] = Field(description="Choose: Support, Attack, or Neither.") 
    overall_summary: str = Field(description="Compare the Support and Attack arguments and explain which is stronger.") 
    final_assessment: str = Field(description="Give the final scientific assessment and explain the verdict.") 
    sources: list[str] = Field(description="Exact URLs of scientific sources used by the Attack Agent.") 
 
   
 
llm = ChatOpenAI( 
    model="openai/gpt-4o-mini", 
    temperature=0.1, 
    api_key=os.getenv("OPENROUTER_API_KEY"), 
    base_url=os.getenv("OPENROUTER_BASE_URL"), 
) 
 
 
 
 
 
 
critic_agent = create_agent( 
    model=llm, 
    tools=[calculate], 
    response_format=CriticResult, 
    system_prompt=""" 
You are Agent 7, the Scientific Critic. 
 
You are the final judge of a scientific claim. 
 
You receive: 
- the original claim 
- supporting evidence 
- the Attack Agent's findings 
- external evidence found by the Attack Agent 
 
Evaluate the claim carefully. 
 
Perform these four checks: 
 
1. Citation Grounding 
Does the evidence actually support the arguments being made? 
 
2. Experimental Parity 
Were the compared studies conducted under reasonably equivalent 
conditions? 
 
3. Generalization 
Does the evidence justify the scope of the original claim? 
 
4. Comparison 
Are the studies and evidence being compared actually comparable? 
 
Then determine exactly one verdict: 
 
Supported 
Partially Supported 
Unsupported 
Inconclusive 
 
Also compare the Support Agent and Attack Agent. 
 
Choose exactly one winner: 
Support 
Attack 
Neither 
 
IMPORTANT: 
You MUST provide a value for EVERY field in CriticResult. 
 
You MUST provide: 
- citation_grounding 
- experimental_parity 
- generalization 
- comparison 
- verdict 
- confidence 
- key_issue 
- winner 
- overall_summary 
- final_assessment 
- sources 
 
For sources: 
 
Extract the exact URLs from the evidence provided by the Attack Agent. 
 
Only include URLs that actually appear in the provided evidence. 
 
Do NOT invent, guess, modify, or fabricate URLs. 
 
If no URLs are available, return an empty list. 
 
Do not invent scientific evidence. 
 
Your final_assessment must clearly state your final judgement 
of the original scientific claim. 
""" 
    ) 
 
 
def run_critic_agent(paper_text,support, attack): 
 
    print("\n" + "=" * 60) 
    print("AGENT 7 — CRITIC AGENT") 
    print("=" * 60) 
 
    try:
        critic_response = critic_agent.invoke({ 
            "messages": [{ 
                "role": "user", 
                "content": f""" 
Original research paper:   {paper_text}                
Original scientific claim: {support.claim} 
 
Supporting evidence from the paper: {support.supporting_evidence} 
 
Attack points: {attack.attack_points} 
 
External evidence found by the Attack Agent: {attack.evidence_found} 
 
Vulnerabilities identified by the Attack Agent: {attack.vulnerabilities} 
 
Attack strength: {attack.strength} 
 
Evaluate these inputs according to your instructions and return the complete CriticResult. 
""" 
        }] 
    }) 
 
        critic = critic_response["structured_response"] 
 
        if critic is None:
            raise ValueError( 
                "Agent 7 did not return a structured response." 
            )

    except Exception as e:
        print(f"\nAgent 7 failed: {e}")
        return None
 
    print("\nVERDICT:") 
    print(critic.verdict) 
 
    print("\nCONFIDENCE:") 
    print(critic.confidence) 
 
    print("\nCITATION GROUNDING:") 
    print(critic.citation_grounding) 
 
    print("\nEXPERIMENTAL PARITY:") 
    print(critic.experimental_parity) 
 
    print("\nGENERALIZATION:") 
    print(critic.generalization) 
 
    print("\nCOMPARISON:") 
    print(critic.comparison) 
 
    print("\nKEY ISSUE:") 
    print(critic.key_issue) 
 
    print("\nWINNER:") 
    print(critic.winner) 
 
    print("\nOVERALL SUMMARY:") 
    print(critic.overall_summary) 
 
    print("\nFINAL ASSESSMENT:") 
    print(critic.final_assessment) 
 
    print("\nSOURCES:") 
    for source in critic.sources: 
        print(f"- {source}") 
 
    return critic
