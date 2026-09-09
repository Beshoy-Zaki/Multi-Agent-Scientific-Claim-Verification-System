from pydantic import BaseModel, Field 
from langchain_openai import ChatOpenAI 
from langchain.agents import create_agent 
from langchain.tools import tool 
from langchain_community.tools import DuckDuckGoSearchRun 
from dotenv import load_dotenv 
from tools import search_scientific_evidence, calculate 
from typing import Literal
import os 
 
 
load_dotenv() 
 
class AttackResult(BaseModel): 
    attack_points: list[str] = Field( 
        description="Specific reasons why the scientific claim may be wrong, overstated, or conditional." 
    ) 
 
    evidence_found: list[str] = Field( 
        description="Evidence from the search that supports the attack." 
    ) 
 
    vulnerabilities: list[str] = Field( 
        description="Scientific or methodological weaknesses found." 
    ) 
 
    strength: Literal["Strong", "Moderate", "Weak"] = Field( 
        description="Overall strength of the attack: Strong, Moderate, or Weak." 
    ) 
 
 
 
tools=[ 
    search_scientific_evidence, 
    calculate 
] 
 
llm = ChatOpenAI( 
    model="openai/gpt-4o-mini", 
    temperature=0.1, #more predictable and less creative. 
    api_key=os.getenv("OPENROUTER_API_KEY"), 
    base_url=os.getenv("OPENROUTER_BASE_URL"), 
) 
 
attack_agent = create_agent( 
    model=llm, 
    tools=tools, 
    response_format=AttackResult, 
    system_prompt=""" 
    You are Agent 6, the Attack Agent in a scientific claim 
    verification system. 
 
    Your job is to challenge the scientific claim. 
 
    Look for: 
    - contradictory evidence 
    - failed or weak replications 
    - methodological weaknesses 
    - overgeneralization 
    - alternative explanations 
    - hidden experimental conditions 
    - numerical inconsistencies 
 
    Choose the appropriate search tool when additional 
    evidence is needed. 
 
    Do not invent evidence. 
 
    Distinguish between: 
    - evidence that directly contradicts the claim 
    - evidence that only limits the claim 
    - evidence that is unrelated 
 
    When using search tools, pay attention to the source URLs 
    included in the search results. 
 
    When reporting evidence_found, include the relevant finding 
    together with its source URL so that another agent can verify 
    the evidence later. 

    When reporting evidence_found, copy the exact URL from the search results.
    Never replace URLs with placeholders such as "[link to evidence]".
    
    Do not invent or guess URLs. 
 
    If evidence is insufficient, say so clearly. 
 
Your response must contain ONLY the fields required by AttackResult. 
""" 
) 
 
def run_attack_agent(paper_text, support): 
 
    print("\n" + "=" * 60) 
    print("AGENT 6 — ATTACK AGENT") 
    print("=" * 60) 
      
      
    try:
        attack_response = attack_agent.invoke({ 
            "messages": [{ 
                "role": "user", 
                "content": f""" 
                        Original research paper: {paper_text} 
 
                        Main scientific claim: {support.claim} 
 
                        Supporting evidence identified from the paper: {support.supporting_evidence} 
 
                        Challenge this claim according to your instructions and return the complete AttackResult. 
                        """ 
            }] 
        }) 
   
        attack = attack_response["structured_response"] 
 
        if attack is None: 
            raise ValueError( 
                "Agent 6 did not return a structured response." 
            )
            
    except Exception as e:
        print(f"\nAgent 6 failed: {e}")
        return None
     
     
    print("\nATTACK POINTS:") 
    for point in attack.attack_points: 
        print(f"- {point}") 
 
    print("\nEVIDENCE FOUND:") 
    for evidence in attack.evidence_found: 
        print(f"- {evidence}") 
 
    print("\nVULNERABILITIES:") 
    for vulnerability in attack.vulnerabilities: 
        print(f"- {vulnerability}") 
 
    print(f"\nATTACK STRENGTH: {attack.strength}") 
 
    return attack