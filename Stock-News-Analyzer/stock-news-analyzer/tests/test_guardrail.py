from pathlib import Path

from src.config import DISCLAIMER, INSUFFICIENT_EVIDENCE
from src.guardrail import has_sufficient_evidence, insufficient_evidence_message


def test_empty_results_are_insufficient():
    assert not has_sufficient_evidence([])


def test_below_threshold_is_insufficient():
    assert not has_sufficient_evidence([{"__score": 0.10}, {"__score": 0.29}], threshold=0.30)


def test_at_or_above_threshold_is_sufficient():
    assert has_sufficient_evidence([{"__score": 0.10}, {"__score": 0.30}], threshold=0.30)


def test_insufficient_message_text():
    msg = insufficient_evidence_message()
    assert msg.startswith("The available evidence does not contain enough information to answer that.")
    assert INSUFFICIENT_EVIDENCE in msg and DISCLAIMER in msg


def test_app_checks_guardrail_before_calling_llm():
    text = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert text.index("has_sufficient_evidence(results)") < text.index("stream_answer(prompt, results, history=history, stats_text=stats_text)")


def test_system_prompt_contains_core_instruction():
    text = (Path(__file__).parents[1] / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")
    assert "evidence-grounded financial news analysis assistant" in text
    assert "<dataset>...</dataset>" in text
    assert "not investment advice" in text
