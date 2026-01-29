"""
LLM-based judge for semantic evaluation of test results.
Provider-agnostic by design.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from framework.llm.provider import LLMProvider


class LLMJudge:
    """
    LLM-based judge for semantic evaluation of test results.

    An LLMProvider must be injected explicitly.
    """

    def __init__(self, llm_provider: LLMProvider | None = None):
        self._provider = llm_provider

    def evaluate(
        self,
        constraint: str,
        source_document: str,
        summary: str,
        criteria: str,
    ) -> dict[str, Any]:
        """Evaluate whether a summary meets a constraint."""
        if not self._provider:
            raise RuntimeError("LLMProvider must be provided to LLMJudge")

        prompt = f"""You are evaluating whether a summary meets a specific constraint.

CONSTRAINT: {constraint}
CRITERIA: {criteria}

SOURCE DOCUMENT:
{source_document}

SUMMARY TO EVALUATE:
{summary}

Respond with JSON: {{"passes": true/false, "explanation": "..."}}"""

        try:
            response = self._provider.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                json_mode=True,
            )
            return self._parse_json_result(response.content.strip())

        except Exception as e:
            return {"passes": False, "explanation": f"LLM judge error: {e}"}

    def _parse_json_result(self, text: str) -> dict[str, Any]:
        """Parse JSON output from the LLM."""
        try:
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()

            result = json.loads(text)
            return {
                "passes": bool(result.get("passes", False)),
                "explanation": result.get("explanation", "No explanation provided"),
            }

        except Exception as e:
            raise ValueError(f"LLM judge error: Failed to parse JSON: {e}") from e
