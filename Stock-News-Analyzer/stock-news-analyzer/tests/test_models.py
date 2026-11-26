import numpy as np
import pandas as pd
import pytest

from src import models


def _toy_df(n=120):
    """Sentiment fully determines direction so the tree should learn it reliably."""
    rng = np.random.default_rng(0)
    sentiment = ["Positive", "Negative", "Neutral"] * (n // 3)
    direction = ["Up", "Down", "Up"] * (n // 3)
    topic = ["Earnings", "Costs", "Outlook"] * (n // 3)
    company = ["A", "B", "C"] * (n // 3)
    year = [2020, 2021, 2022] * (n // 3)
    volume = rng.integers(100_000, 2_000_000, size=n)
    news = [f"{c} reports on {t} with {s} tone number {i}" for i, (c, t, s) in enumerate(zip(company, topic, sentiment))]
    df = pd.DataFrame({
        "news": news, "body": [n + " Extra context sentence." for n in news],
        "company": company, "topic": topic, "sentiment": sentiment,
        "year": year, "price_change_percent": [5, -5, 1] * (n // 3), "price_direction": direction,
        "volume": volume,
    })
    df["__date"] = pd.to_datetime(df["year"].astype(str), format="%Y")
    return df


def test_availability():
    df = _toy_df()
    assert models.direction_model_available(df)
    assert models.clustering_available(df)
    assert not models.direction_model_available(df.drop(columns=["price_direction"]))


def test_direction_model_learns_sentiment():
    bundle = models.train_direction_model(_toy_df())
    # Should clearly beat the majority baseline on a learnable signal.
    assert bundle["metrics"]["accuracy"] >= bundle["metrics"]["baseline_majority"]
    assert "sentiment" in set(bundle["field_importance"]["field"])
    assert bundle["rules"], "expected at least one extracted rule"


def test_rule_conditions_are_readable():
    bundle = models.train_direction_model(_toy_df())
    joined = " ".join(r["conditions"] for r in bundle["rules"])
    assert "cat__" not in joined and "num__" not in joined


def test_predict_direction_runs():
    bundle = models.train_direction_model(_toy_df())
    out = models.predict_direction(bundle, {"sentiment": "Positive", "topic": "Earnings",
                                            "company": "A", "year": 2021, "log_volume": 15.0})
    assert out["predicted"] in bundle["labels"]
    assert abs(sum(out["probs"].values()) - 1.0) < 1e-6


def test_clusters_shape_and_stats():
    cb = models.train_clusters(_toy_df(150), n_clusters=3)
    assert cb["k"] == 3
    assert len(cb["points"]) == 150
    distinct = {int(c) for c in cb["points"]["cluster"]}
    assert distinct.issubset({0, 1, 2})
    # stats are summarised per cluster that actually received points.
    assert len(cb["stats"]) == len(distinct)


def test_cluster_quality_monotonic_keys():
    q = models.cluster_quality(_toy_df(150))
    assert list(q.columns) == ["k", "silhouette"]
    assert q["k"].is_monotonic_increasing
