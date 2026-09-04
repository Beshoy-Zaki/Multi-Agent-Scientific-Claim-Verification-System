"""Unified LLM client for Google Gemma 4 31B and multi-model routing."""

import os
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
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
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not found in environment or .env file.")

        genai.configure(api_key=self.api_key)

        # Normalize model identifier (e.g. 'gemma-4-31b-it' -> 'models/gemma-4-31b-it')
        raw_model = model_name or os.getenv("CLAIM_ANALYST_MODEL") or os.getenv("DEFAULT_MODEL", "gemma-4-31b-it")
        self.model_name = raw_model if raw_model.startswith("models/") else f"models/{raw_model}"
        self.temperature = temperature
        logger.info(f"Initialized LLMClient with model: {self.model_name}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text completion from prompt with optional system prompt."""
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt if system_prompt else None,
                generation_config=genai.GenerationConfig(temperature=self.temperature),
            )
            response = model.generate_content(prompt)
            return response.text or ""
        except Exception as exc:
            logger.error(f"LLM generation failed for '{self.model_name}': {exc}")
            raise
