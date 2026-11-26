import pandas as pd

from src.data_loader import load_and_clean


def test_real_dataset_loads_without_missing_values():
    df, info, path = load_and_clean()
    assert len(df) == 15000
    assert info.date_min == "2019"
    assert info.date_max == "2025"
    assert len(info.tickers) == 30
    assert not info.missing
    assert "__content" in df.columns


def test_duplicate_cleaning():
    original = pd.DataFrame({"news": [" A ", " A "], "year": [2020, 2020], "company": ["X", "X"]})
    cleaned = original.drop_duplicates().copy()
    for column in ["news", "company"]:
        cleaned[column] = cleaned[column].astype(str).str.strip()
    assert len(cleaned) == 1


def test_all_expected_columns_present_and_extensions_available():
    from src.config import EXPECTED_COLUMNS
    df, info, _ = load_and_clean()
    assert set(EXPECTED_COLUMNS) <= set(info.columns)
    for extension in ("sector", "source", "date", "body"):
        assert extension in info.columns
    assert info.features["price"] and info.features["topic"]
    assert info.features["sector"] and info.features["source"]
    assert info.sectors and info.sources


def test_missing_required_column_raises_clean_error(tmp_path, monkeypatch):
    import pytest
    from src import data_loader
    bad = tmp_path / "bad.csv"
    bad.write_text("headline,foo\nx,1\n")
    monkeypatch.setattr(data_loader, "resolve_data_file", lambda: bad)
    with pytest.raises(data_loader.DatasetError) as exc:
        data_loader.load_and_clean()
    assert "missing required columns" in str(exc.value) and "Traceback" not in str(exc.value)


def test_corrupt_and_empty_csv_raise_dataset_error(tmp_path, monkeypatch):
    import pytest
    from src import data_loader
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    monkeypatch.setattr(data_loader, "resolve_data_file", lambda: empty)
    with pytest.raises(data_loader.DatasetError):
        data_loader.load_and_clean()


def test_clean_csv_bytes_matches_file_load():
    from src.data_loader import clean_csv_bytes
    df, info = clean_csv_bytes(
        pd.DataFrame({
            "news": ["A", "B"], "date": ["2021-01-05", "2021-02-05"], "year": [2021, 2021],
            "company": ["X", "Y"], "topic": ["Earnings", "Costs"], "sentiment": ["Positive", "Negative"],
        }).to_csv(index=False).encode()
    )
    assert len(df) == 2 and info.rows == 2
    assert df["__date"].dt.year.tolist() == [2021, 2021]
