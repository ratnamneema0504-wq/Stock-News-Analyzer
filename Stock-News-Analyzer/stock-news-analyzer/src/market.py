"""Historical market-impact statistics computed directly from the dataset.

These are descriptive statistics of past observations, never predictions.
"""
from __future__ import annotations

import pandas as pd

from .config import COLUMN_MAP


def _col(name: str, df: pd.DataFrame) -> str | None:
    column = COLUMN_MAP.get(name)
    return column if column and column in df.columns else None


def market_impact_available(df: pd.DataFrame) -> bool:
    col = _col("price_change_percent", df)
    return bool(col) and df[col].notna().any()


def market_impact_stats(df: pd.DataFrame) -> dict | None:
    col = _col("price_change_percent", df)
    if not col:
        return None
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return None
    n = len(values)
    return {
        "observations": n,
        "average": float(values.mean()),
        "median": float(values.median()),
        "positive_pct": float((values > 0).sum() / n * 100),
        "negative_pct": float((values < 0).sum() / n * 100),
        "flat_pct": float((values == 0).sum() / n * 100),
    }


def impact_by_group(df: pd.DataFrame, group: str) -> pd.DataFrame | None:
    """Mean/median/count of price change per group ('company', 'topic', 'sentiment', 'impact')."""
    logical = {"company": "ticker", "topic": "event_type"}.get(group, group)
    gcol = _col(logical, df)
    ccol = _col("price_change_percent", df)
    if not gcol or not ccol:
        return None
    out = (
        df.groupby(gcol)[ccol]
        .agg(observations="count", mean_change="mean", median_change="median",
             mean_abs_change=lambda s: s.abs().mean())
        .reset_index()
        .rename(columns={gcol: group})
    )
    return out.sort_values("mean_abs_change", ascending=False)


def news_profile_text(df: pd.DataFrame, companies: list[str] | None = None, min_rows: int = 5) -> str | None:
    """Aggregate the dataset into a plain-language statistics block for the LLM prompt.

    Returns None when there is too little data to summarise meaningfully.
    """
    d = df
    tcol = _col("ticker", d)
    if companies and tcol:
        sel = d[d[tcol].isin(companies)]
        if len(sel) >= min_rows:
            d = sel
    if len(d) < min_rows:
        return None

    ccol = _col("price_change_percent", d)
    scol = _col("sentiment", d)
    tcol = _col("event_type", d)
    icol = _col("impact", d)
    dcol = _col("date", d)
    if not ccol:
        return None
    changes = pd.to_numeric(d[ccol], errors="coerce")
    d = d.assign(__change=changes)
    if dcol and pd.api.types.is_datetime64_any_dtype(d[dcol]):
        d = d.assign(__year=d[dcol].dt.year)
    elif "year" in d.columns:
        d = d.assign(__year=pd.to_numeric(d["year"], errors="coerce"))
    else:
        d = d.assign(__year=None)
    d = d[d["__change"].notna()]
    if len(d) < min_rows:
        return None

    lines = [f"Rows analysed: {len(d):,}"]
    yv = d["__year"].dropna()
    if len(yv):
        lines.append(f"Year range: {int(yv.min())} to {int(yv.max())}")

    if scol:
        n = len(d)
        mix = d[scol].value_counts()
        parts = [f"{name} {cnt / n * 100:.0f}% ({cnt:,} news items)" for name, cnt in mix.items()]
        lines.append("Sentiment mix: " + ", ".join(parts))
        for name, grp in d.groupby(scol):
            ch = grp["__change"]
            up = (ch > 0).sum() / len(ch) * 100
            lines.append(
                f"After {name} sentiment news: average price change {ch.mean():+.2f}%, "
                f"median {ch.median():+.2f}%, share price rose in {up:.0f}% of cases (n={len(ch):,})"
            )
    if icol:
        counts = d[d[icol].notna()][icol].value_counts()
        if len(counts):
            lines.append("Recorded impact size: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))

    if tcol:
        grouped = (
            d[d[tcol].notna()]
            .assign(__abs=d["__change"].abs())
            .groupby(tcol)
            .agg(mean_change=("__change", "mean"), abs_change=("__abs", "mean"), n=("__change", "count"))
            .sort_values("abs_change", ascending=False)
            .head(3)
        )
        if len(grouped):
            top = "; ".join(
                f"{topic}: avg {row.mean_change:+.2f}%, moves of about {row.abs_change:.2f}% on average ({int(row.n):,} items)"
                for topic, row in grouped.iterrows()
            )
            lines.append("Topics with the largest average price movement: " + top)

    overall = d["__change"]
    up = (overall > 0).sum() / len(overall) * 100
    lines.append(
        f"Overall: average price change {overall.mean():+.2f}%, median {overall.median():+.2f}%, "
        f"price rose after {up:.0f}% of these news items and fell after {(overall < 0).sum() / len(overall) * 100:.0f}%"
    )
    return "\n".join(lines)
