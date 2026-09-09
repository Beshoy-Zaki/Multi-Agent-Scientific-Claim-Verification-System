"""Unified LLM client for Google Gemma 4 using the official google-genai SDK."""

import logging
import os
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

from mascv.utils.logger import get_logger

logger = get_logger(__name__)

# Load variables from .env file
load_dotenv()


class LLMClient:
    """Unified LLM client interface for Google Gemma 4 with Search Grounding support."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        enable_grounding: bool = False,
        thinking_level: Optional[str] = None,
    ) -> None:
        """Initialize client with model name, temperature, API key, search grounding flag, and thinking level."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not found in environment or .env file.")

        raw_model = (
            model_name
            or os.getenv("PAPER_SEARCH_MODEL")
            or os.getenv("DEFAULT_MODEL", "gemma-4-26b-a4b-it")
        )
        self.model_name = raw_model.replace("models/", "")
        self.temperature = temperature
        self.enable_grounding = enable_grounding
        
        # Gemma 4 strictly supports thinking_level as 'minimal' (disabled/fast) or 'high' (deep reasoning)
        raw_thinking = thinking_level or os.getenv("THINKING_LEVEL", "minimal")
        self.thinking_level = "high" if str(raw_thinking).lower() in ["high", "true", "1"] else "minimal"

        self.client = (
            genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=120_000))
            if self.api_key
            else None
        )

        logger.info(
            "Initialized LLMClient (model=%s, temp=%.2f, grounding=%s, thinking=%s)",
            self.model_name,
            self.temperature,
            self.enable_grounding,
            self.thinking_level,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        enable_grounding: Optional[bool] = None,
        thinking_level: Optional[str] = None,
    ) -> str:
        """Generate text completion using official google-genai SDK with Search Grounding."""
        if not self.client:
            raise ValueError("GOOGLE_API_KEY is required to call the model API.")

        use_grounding = (
            self.enable_grounding if enable_grounding is None else enable_grounding
        )
        
        t_level = self.thinking_level
        if thinking_level is not None:
            t_level = "high" if str(thinking_level).lower() in ["high", "true", "1"] else "minimal"

        config_args = {
            "temperature": self.temperature,
            "thinking_config": types.ThinkingConfig(thinking_level=t_level),
        }

        if system_prompt:
            config_args["system_instruction"] = system_prompt

        if use_grounding:
            config_args["tools"] = [{"google_search": {}}]

        config = types.GenerateContentConfig(**config_args)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return response.text or ""

        except Exception as exc:
            logger.error("Gemma 4 generation failed: %s", exc)
            raise

