import pandas as pd


def apply_year_filter(df, start, end):
    return df[(df["__date"] >= start) & (df["__date"] <= end)]


def test_year_filter_matches_only_selected_year():
    df = pd.DataFrame({
        "company": ["A", "B", "A"],
        "__date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
    })
    filtered = apply_year_filter(
        df,
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-12-31 23:59:59"),
    )
    assert filtered["company"].tolist() == ["B"]
