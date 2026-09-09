"""Scientific evidence verification tools: Google Search Grounding with relaxed criteria and arithmetic validation."""

import ast
import logging
import operator
import os
import re
from typing import Optional
from langchain.tools import tool

from mascv.utils.llm import LLMClient

logger = logging.getLogger(__name__)

# Optional DuckDuckGo fallback
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None


def _google_search_evidence(query: str) -> str:
    """Use Google Search Grounding via Gemma 4 to find reasonable counter-evidence, criticisms, and limitations."""
    client = LLMClient(
        model_name="gemma-4-26b-a4b-it",
        temperature=0.2,
        enable_grounding=True,
        thinking_level="minimal",
    )
    prompt = f"""Search Google for credible critiques, limitations, drawbacks, empirical benchmarks, or counter-evidence concerning this proposition:
"{query}"

GUIDELINES:
- Relaxed filtering: the results do NOT need to be strictly peer-reviewed academic papers. They can be technical blog posts (e.g., Hugging Face, Weights & Biases, OpenAI, Towards Data Science), empirical benchmarks, engineering postmortems, GitHub issues/discussions, or papers.
- As long as the source is reasonable, technical, and presents a credible counter-perspective, empirical limit, or critique, include it.
- Return 2 to 4 distinct findings.
- For each finding, include:
  * Title / Source Name
  * URL (canonical direct link)
  * Summary of the critique, limitation, or conflicting observation.

Format each result cleanly:
Result 1:
Title: <title>
URL: <url>
Snippet: <summary>
"""
    system_prompt = (
        "You are an Adversarial Research Assistant using live Google Search. "
        "Find real, practical, and reasonable critiques, edge cases, failure modes, "
        "and empirical limitations for the provided scientific or technical proposition."
    )
    return client.generate(prompt=prompt, system_prompt=system_prompt)


def _duckduckgo_fallback(query: str) -> str:
    """Fallback search using DuckDuckGo with simplified keywords if Google Search is unavailable."""
    if DDGS is None:
        return (
            f"External web search unavailable (DDGS client not installed). "
            f"Could not retrieve external counter-evidence for: '{query}'."
        )

    try:
        clean_q = re.sub(r"[^\w\s]", " ", query)
        words = [w for w in clean_q.split() if len(w) > 2][:5]
        search_query = " ".join(words) + " limitations"
        results = DDGS().text(search_query.strip(), max_results=3)
        if not results:
            return "No search results found via fallback."

        output = []
        for i, result in enumerate(results, start=1):
            output.append(
                f"Result {i}:\n"
                f"Title: {result.get('title', 'N/A')}\n"
                f"URL: {result.get('href', 'N/A')}\n"
                f"Snippet: {result.get('body', 'N/A')}"
            )
        return "\n\n".join(output)
    except Exception as exc:
        logger.warning("DuckDuckGo fallback search error: %s", exc)
        return f"External web search failed ({exc}). No external critique findings retrieved."


_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast_node(node: ast.AST) -> float:
    """Safely evaluate AST arithmetic nodes without executing arbitrary code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("division by zero")
        return _SAFE_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        operand = _eval_ast_node(node.operand)
        return _SAFE_OPERATORS[type(node.op)](operand)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


@tool
def search_scientific_evidence(query: str) -> str:
    """
    Search for scientific and technical evidence, engineering blogs, benchmarks,
    or discussions that may contradict, weaken, or limit a research claim.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            logger.info("Executing Google Search Grounding for counter-evidence: '%s'", query[:60])
            result = _google_search_evidence(query)
            if result and result.strip():
                return result
        except Exception as exc:
            logger.warning("Google search evidence failed (%s); falling back to DuckDuckGo.", exc)

    return _duckduckgo_fallback(query)


@tool
def calculate(expression: str) -> str:
    """
    Perform a basic mathematical calculation (+, -, *, /).
    Useful for checking percentages, differences,
    ratios, and numerical claims.
    """
    allowed = set("0123456789+-*/(). ")
    if not all(char in allowed for char in expression):
        return "Error: Invalid mathematical expression."

    if "**" in expression or len(expression) > 100:
        return "Error: Exponentiation or excessively long expressions are not allowed."

    try:
        parsed = ast.parse(expression.strip(), mode="eval")
        result = _eval_ast_node(parsed.body)
        if result.is_integer():
            return str(int(result))
        return f"{result:.6g}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception:
        return "Error: Could not calculate expression."
