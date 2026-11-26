from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from .config import COLUMN_MAP

SENTIMENT_COLORS = {"Positive": "#22c55e", "Neutral": "#94a3b8", "Negative": "#ef4444"}
IMPACT_COLORS = {"High": "#6366f1", "Medium": "#38bdf8", "Low": "#94a3b8"}
ACCENT = "#6366f1"


def _c(name: str, df: pd.DataFrame) -> str | None:
    col = COLUMN_MAP.get(name)
    return col if col and col in df.columns else None


def _style(fig, title: str, height: int = 340):
    """Transparent, theme-neutral styling that works in light and dark mode."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15), x=0.0, xanchor="left"),
        height=height,
        margin=dict(l=8, r=8, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(title=None, orientation="h", y=-0.2),
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.2)", title=None)
    return fig


def news_count_by_ticker(df: pd.DataFrame):
    col = _c("ticker", df)
    if not col:
        return None
    data = df[col].value_counts().rename_axis(col).reset_index(name="articles")
    fig = px.bar(data, x=col, y="articles", color_discrete_sequence=[ACCENT])
    fig.update_xaxes(tickangle=-40)
    return _style(fig, "News by company")


def sentiment_split(df: pd.DataFrame):
    col = _c("sentiment", df)
    if not col:
        return None
    data = df[col].value_counts().rename_axis(col).reset_index(name="articles")
    fig = px.pie(data, names=col, values="articles", hole=0.6, color=col, color_discrete_map=SENTIMENT_COLORS)
    fig.update_traces(textinfo="percent", sort=False)
    return _style(fig, "Sentiment distribution")


def sentiment_over_time(df: pd.DataFrame):
    col = _c("sentiment", df)
    if not col or "__date" not in df.columns or df["__date"].isna().all():
        return None
    data = df.assign(year=df["__date"].dt.year).groupby(["year", col]).size().reset_index(name="articles")
    fig = px.line(data, x="year", y="articles", color=col, markers=True, color_discrete_map=SENTIMENT_COLORS)
    fig.update_xaxes(dtick=1)
    return _style(fig, "Sentiment over time")


def price_impact(df: pd.DataFrame):
    """Distribution of historical price changes (%)."""
    col = _c("price_change_percent", df)
    if not col:
        return None
    fig = px.histogram(df, x=col, nbins=40, color_discrete_sequence=[ACCENT])
    fig.update_layout(bargap=0.05)
    return _style(fig, "Historical price change distribution (%)")


def impact_by_topic(df: pd.DataFrame):
    topic, change = _c("event_type", df), _c("price_change_percent", df)
    if not topic or not change:
        return None
    data = df.groupby(topic)[change].mean().reset_index().sort_values(change)
    data["direction"] = data[change].apply(lambda v: "Positive" if v >= 0 else "Negative")
    fig = px.bar(data, x=change, y=topic, orientation="h", color="direction", color_discrete_map=SENTIMENT_COLORS)
    fig.update_layout(showlegend=False)
    return _style(fig, "Average historical price change by topic (%)", height=420)


def impact_distribution(df: pd.DataFrame):
    col = _c("impact", df)
    if not col:
        return None
    order = [v for v in ("High", "Medium", "Low") if v in set(df[col])]
    data = df[col].value_counts().reindex(order).rename_axis(col).reset_index(name="articles")
    fig = px.bar(data, x=col, y="articles", color=col, color_discrete_map=IMPACT_COLORS)
    fig.update_layout(showlegend=False)
    return _style(fig, "Impact level distribution", height=300)


def price_by_company(df: pd.DataFrame):
    ticker, change = _c("ticker", df), _c("price_change_percent", df)
    if not ticker or not change:
        return None
    fig = px.box(df, x=ticker, y=change, points=False, color_discrete_sequence=[ACCENT])
    fig.update_xaxes(tickangle=-40)
    return _style(fig, "Historical price change by company (%)")


def sector_impact(df: pd.DataFrame):
    sector, change = _c("sector", df), _c("price_change_percent", df)
    if not sector or not change:
        return None
    data = df.groupby(sector)[change].agg(["mean", "count"]).reset_index()
    data["direction"] = data["mean"].apply(lambda v: "Positive" if v >= 0 else "Negative")
    fig = px.bar(data, x="mean", y=sector, orientation="h", color="direction",
                 color_discrete_map=SENTIMENT_COLORS,
                 hover_data={"count": True})
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Mean price change %")
    return _style(fig, "Average price change by sector (%)", height=340)


def sector_news_volume(df: pd.DataFrame):
    sector = _c("sector", df)
    if not sector:
        return None
    data = df[sector].value_counts().rename_axis(sector).reset_index(name="articles")
    fig = px.bar(data, x=sector, y="articles", color=sector,
                 color_discrete_sequence=px.colors.qualitative.Vivid,
                 hover_data={"articles": True})
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=-40)
    return _style(fig, "News coverage by sector", height=340)


# Backwards-compatible aliases for the earlier chart names.
topic_split = impact_by_topic
price_vs_news = price_by_company


# --------------------------------------------------------------- AI model charts
CLUSTER_COLORS = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#38bdf8",
                  "#a855f7", "#14b8a6", "#eab308", "#fb7185", "#94a3b8"]


def feature_importance(field_importance: pd.DataFrame):
    data = field_importance[field_importance["importance"] > 0]
    if data.empty:
        data = field_importance
    data = data.sort_values("importance")
    fig = px.bar(data, x="importance", y="field", orientation="h",
                 color_discrete_sequence=[ACCENT])
    fig.update_traces(texttemplate="%{x:.2f}", textposition="outside")
    return _style(fig, "Feature importance (which news attributes drive the decision)", height=300)


def confusion_heatmap(cm, labels: list[str]):
    fig = px.imshow(np.asarray(cm), text_auto=True, aspect="auto",
                    x=labels, y=labels, color_continuous_scale="Purples")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title="Predicted", showgrid=False)
    fig.update_yaxes(title="Actual", showgrid=False)
    return _style(fig, "Confusion matrix (held-out test set)", height=320)


def cluster_scatter(points: pd.DataFrame):
    if points is None or points.empty:
        return None
    fig = px.scatter(points, x="x", y="y", color="cluster", hover_data=["news"],
                     color_discrete_sequence=CLUSTER_COLORS, opacity=0.75)
    fig.update_traces(marker=dict(size=6))
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.2)")
    return _style(fig, "K-means clusters of news (TF-IDF, 2-D projection)", height=460)


def silhouette_curve(quality: pd.DataFrame):
    if quality is None or quality.empty:
        return None
    fig = px.line(quality, x="k", y="silhouette", markers=True,
                  color_discrete_sequence=["#a855f7"])
    fig.update_xaxes(dtick=1, gridcolor="rgba(148,163,184,0.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.2)")
    return _style(fig, "Silhouette score by number of clusters (k)", height=300)
