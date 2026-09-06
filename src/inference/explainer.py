"""
Gemini-powered plain-English clause explanations.

Separation of concerns (this is intentional, per the project architecture):
  - The fine-tuned LegalBERT model is responsible for CLASSIFICATION only.
  - Gemini is used ONLY for EXPLANATION -- turning an already-predicted
    category + clause text into a plain-English description a non-lawyer
    can understand. Gemini never decides the category itself; it always
    receives the category as an input, already decided by our own model.
"""

import os

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_PROMPT_TEMPLATE = """You are explaining a legal contract clause to someone with no legal training.

Clause category (already predicted by a machine learning model): "{category}"
Clause text: "{text}"

In 2-4 short sentences, plain English, no legal jargon:
1. Explain what this type of clause generally means and why it matters in a contract.
2. Briefly note anything specific about THIS clause's wording that stands out.
3. If the clause text does not actually seem to match the stated category, say so plainly instead of forcing an explanation.

Do not give legal advice. Do not start with phrases like "This clause is about" -- explain directly.
"""


class GeminiExplainer:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self._available = bool(api_key)
        self._client = None

        if self._available:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            logger.info("Gemini explainer initialized.")
        else:
            logger.warning("GEMINI_API_KEY not set. Explanations will be unavailable.")

    def is_available(self) -> bool:
        return self._available

    def explain(self, category: str, text: str) -> str:
        if not self._available:
            raise RuntimeError("Gemini is not configured (GEMINI_API_KEY missing).")

        prompt = _PROMPT_TEMPLATE.format(category=category, text=text)
        response = self._client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        return response.text.strip()
