import pandas as pd

from src.market import impact_by_group, market_impact_available, market_impact_stats, news_profile_text


def _df():
    return pd.DataFrame({
        "company": ["A", "A", "B", "B"],
        "topic": ["Earnings", "Costs", "Earnings", "Costs"],
        "price_change_percent": [4.0, -2.0, 2.0, -4.0],
    })


def test_stats_values():
    s = market_impact_stats(_df())
    assert s["average"] == 0.0 and s["median"] == 0.0
    assert s["positive_pct"] == 50.0 and s["negative_pct"] == 50.0 and s["observations"] == 4


def test_missing_price_column_hides_feature():
    df = _df().drop(columns=["price_change_percent"])
    assert not market_impact_available(df)
    assert market_impact_stats(df) is None
    assert impact_by_group(df, "topic") is None


def test_group_summary():
    out = impact_by_group(_df(), "topic").set_index("topic")
    assert out.loc["Earnings", "mean_change"] == 3.0


def test_news_profile_aggregates():
    df = pd.DataFrame({
        "company": ["A"] * 5 + ["B"] * 3,
        "sentiment": ["Positive", "Positive", "Negative", "Positive", "Negative",
                      "Negative", "Positive", "Negative"],
        "topic": ["Earnings", "Costs", "Earnings", "Costs", "Earnings",
                  "Earnings", "Costs", "Earnings"],
        "price_change_percent": [3.0, 1.0, -2.0, 4.0, -1.0, -1.0, 2.0, -3.0],
        "impact": ["Medium", "Low", "Low", "High", "Low", "Low", "Low", "Medium"],
    })
    text = news_profile_text(df)
    assert "Rows analysed: 8" in text
    assert "Sentiment mix" in text
    assert "After Positive sentiment news" in text
    assert "Overall: average price change" in text
    assert "Rows analysed: 5" in news_profile_text(df, ["A"])
    assert "Rows analysed: 8" in news_profile_text(df, ["C"])  # too few rows -> falls back to full frame


def test_news_profile_needs_price_column():
    assert news_profile_text(pd.DataFrame({"company": ["A"] * 6})) is None
    assert news_profile_text(pd.DataFrame({"price_change_percent": [1.0, 2.0, 3.0]})) is None
