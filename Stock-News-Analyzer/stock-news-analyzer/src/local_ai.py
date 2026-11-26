from __future__ import annotations

import ollama

from .config import get_model


def is_available() -> bool:
    """Return True when the configured local Ollama model is available."""
    try:
        ollama.show(model=get_model())
        return True
    except Exception:
        return False


def _request(system_prompt: str, user_prompt: str):
    return ollama.chat(
        model=get_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )


def stream_chat(system_prompt: str, user_prompt: str):
    """Stream text from the configured local Ollama model.

    Retries once when the request fails before producing any output
    (e.g. transient connection error while Ollama loads the model).
    """
    response = None
    for attempt in (0, 1):
        try:
            response = _request(system_prompt, user_prompt)
            break
        except Exception:
            if attempt:
                raise
    yielded = False
    for chunk in response:
        try:
            text = chunk["message"]["content"]
        except (KeyError, TypeError):
            text = ""
        if text:
            yielded = True
            yield text
