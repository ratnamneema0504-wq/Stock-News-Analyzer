from __future__ import annotations

import pandas as pd

from .config import COLUMN_MAP


def get_sentiment_column(df: pd.DataFrame) -> str | None:
    column = COLUMN_MAP.get("sentiment")
    return column if column and column in df.columns else None


def batch_compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    # The supplied dataset already contains sentiment. This function is retained
    # for compatibility with replacement datasets that do not.
    existing = get_sentiment_column(df)
    if existing:
        return df
    raise RuntimeError(
        "This dataset has no configured sentiment column. Add one to COLUMN_MAP "
        "or implement an LLM sentiment batch job for the replacement dataset."
    )
