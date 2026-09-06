"""Unified LLM client for Google Gemma 4 31B and multi-model routing."""

import os
from typing import Optional
from dotenv import load_dotenv
import requests

from mascv.utils.logger import get_logger

logger = get_logger(__name__)

# Load variables from .env file
load_dotenv()


class LLMClient:
    """Unified LLM client interface for Google Gemma 4 and Gemini models."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize client with model name, temperature, and API key."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not found in environment or .env file.")

        raw_model = (
            model_name
            or os.getenv("DEFAULT_MODEL")
            or os.getenv("CLAIM_ANALYST_MODEL")
            or os.getenv("PAPER_SEARCH_MODEL", "gemma-4-31b-it")
        )
        self.model_name = raw_model.replace("models/", "")
        self.temperature = temperature

        logger.info(
            "Initialized LLMClient (model=%s, temperature=%.2f)",
            self.model_name,
            self.temperature,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call Google's API to generate text completion using Gemma 4."""
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required to call the model API.")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
            },
        }

        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=60)

        if response.status_code != 200:
            logger.error("API error %d: %s", response.status_code, response.text)
            raise RuntimeError(f"API error ({response.status_code}): {response.text}")

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""

        parts = candidates[0].get("content", {}).get("parts", [])
        text_chunks = [part.get("text", "") for part in parts if "text" in part]
        return "".join(text_chunks)
