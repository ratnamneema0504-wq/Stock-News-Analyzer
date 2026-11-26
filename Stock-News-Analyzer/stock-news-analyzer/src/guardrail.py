"""Dataset-grounding guardrail: decides whether the LLM may be called at all."""
from __future__ import annotations

from .config import DISCLAIMER, INSUFFICIENT_EVIDENCE, SIMILARITY_THRESHOLD


def has_sufficient_evidence(results: list[dict], threshold: float | None = None) -> bool:
    """True only if at least one retrieved row meets the similarity threshold."""
    limit = SIMILARITY_THRESHOLD if threshold is None else threshold
    if not results:
        return False
    best = max(float(r.get("__score", 0.0)) for r in results)
    return best >= limit


def insufficient_evidence_message() -> str:
    return f"{INSUFFICIENT_EVIDENCE}\n\n{DISCLAIMER}"
