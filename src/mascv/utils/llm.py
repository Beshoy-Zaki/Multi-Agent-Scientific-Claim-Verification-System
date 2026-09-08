"""Unified LLM client for Google Gemma 4 (a4b/31b) with optional Google Search Grounding."""

import json
import logging
import os
from typing import Optional
from dotenv import load_dotenv
import requests

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
        self.thinking_level = (
            thinking_level
            if thinking_level is not None
            else os.getenv("THINKING_LEVEL", "HIGH")
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
        """Generate text completion with optional live Google Search Grounding and configurable thinking level."""
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required to call the model API.")

        use_grounding = (
            self.enable_grounding if enable_grounding is None else enable_grounding
        )
        active_thinking = (
            self.thinking_level if thinking_level is None else thinking_level
        )

        # Using streamGenerateContent for fast streaming and resilience against timeouts
        endpoint = "streamGenerateContent" if use_grounding else "generateContent"
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:{endpoint}?key={self.api_key}"
        )

        generation_config = {
            "temperature": self.temperature,
        }
        if active_thinking:
            generation_config["thinkingConfig"] = {
                "thinkingLevel": active_thinking.upper()
            }

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        if use_grounding:
            # Enables live Google Search Grounding on Gemma 4 a4b
            payload["tools"] = [{"googleSearch": {}}]

        headers = {"Content-Type": "application/json"}
        timeout = 120 if use_grounding else 60

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
                stream=use_grounding,
            )

            if response.status_code != 200:
                logger.error("API error %d: %s", response.status_code, response.text)
                raise RuntimeError(f"API error ({response.status_code}): {response.text}")

            # Parse response chunks (streamGenerateContent returns a JSON array)
            data = response.json()
            chunks = data if isinstance(data, list) else [data]

            full_text = []
            for chunk in chunks:
                for candidate in chunk.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        # Skip internal reasoning thoughts; collect only the output text
                        if not part.get("thought") and "text" in part:
                            full_text.append(part["text"])

            return "".join(full_text)

        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raise
