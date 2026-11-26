from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import COLUMN_MAP, DATA_FILE


class DatasetError(Exception):
    """Raised when the bundled dataset cannot be loaded or validated."""


@dataclass(frozen=True)
class DatasetInfo:
    rows: int
    columns: list[str]
    date_min: str | None
    date_max: str | None
    tickers: list[str]
    sectors: list[str]
    topics: list[str]
    missing: dict[str, int]
    file_hash: str
    duplicates_removed: int
    sources: list[str] = field(default_factory=list)
    features: dict[str, bool] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_file() -> Path:
    if DATA_FILE.exists():
        return DATA_FILE
    candidates = sorted(DATA_FILE.parent.glob("*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise DatasetError(
            "Dataset CSV not found. Place the bundled CSV in the data/ folder "
            "or set DATA_FILE in .env."
        )
    raise DatasetError(
        "Multiple CSV files were found in data/. Set DATA_FILE in .env to the intended filename."
    )


def _mapped(logical: str) -> str | None:
    return COLUMN_MAP.get(logical)


def _require_columns(df: pd.DataFrame) -> None:
    required = []
    for logical in ("headline", "date", "ticker"):
        column = _mapped(logical)
        if column:
            required.append(column)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DatasetError(
            "The dataset is missing required columns: " + ", ".join(missing) + "."
        )


def load_and_clean() -> tuple[pd.DataFrame, DatasetInfo, Path]:
    path = resolve_data_file()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("The dataset file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError("The dataset file appears to be corrupt or is not a valid CSV.") from exc
    except OSError as exc:
        raise DatasetError("The dataset file could not be opened.") from exc
    except Exception as exc:
        raise DatasetError("The dataset file could not be read.") from exc
    df, info = prepare_dataset(df, sha256_file(path))
    return df, info, path


def clean_csv_bytes(data: bytes) -> tuple[pd.DataFrame, DatasetInfo]:
    """Validate and clean an uploaded CSV, returning the same shape as load_and_clean."""
    import io
    try:
        df = pd.read_csv(io.BytesIO(data))
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("The uploaded file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError("The uploaded file is not a valid CSV.") from exc
    except Exception as exc:
        raise DatasetError("The uploaded file could not be read.") from exc
    return prepare_dataset(df, hashlib.sha256(data).hexdigest())


def prepare_dataset(df: pd.DataFrame, file_hash: str) -> tuple[pd.DataFrame, DatasetInfo]:
    if df.empty:
        raise DatasetError("The dataset file contains no rows.")

    df.columns = [str(column).strip() for column in df.columns]
    _require_columns(df)

    before = len(df)
    df = df.drop_duplicates().copy()
    duplicates_removed = before - len(df)

    # Strip whitespace from string fields without destroying numeric values.
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].astype("string").str.strip()

    date_column = _mapped("date")
    if date_column and date_column in df.columns:
        # The supplied dataset has YEAR, not a full calendar date. Converting
        # it to Jan-01 preserves year-level filtering while avoiding invented
        # month/day information in the UI.
        if pd.api.types.is_numeric_dtype(df[date_column]):
            df["__date"] = pd.to_datetime(df[date_column].astype("Int64").astype(str), format="%Y", errors="coerce")
        else:
            df["__date"] = pd.to_datetime(df[date_column], errors="coerce")
        if df["__date"].isna().all():
            raise DatasetError(f"Could not parse the configured date/year column '{date_column}'.")
    else:
        df["__date"] = pd.NaT

    # Numeric optional fields are coerced when present.
    for logical in ("price_before", "price_after", "price_change_percent", "volume", "open", "high", "low", "close"):
        column = _mapped(logical)
        if column and column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Fill text used by embeddings with safe empty strings rather than literal NaN.
    headline = _mapped("headline")
    text = _mapped("text")
    ticker = _mapped("ticker")
    topic = _mapped("event_type")
    sentiment = _mapped("sentiment")
    parts = []
    for column in (headline, text, ticker, topic, sentiment):
        if column and column in df.columns:
            parts.append(df[column].fillna("").astype(str))
    if not parts:
        raise DatasetError("No usable text columns were found for retrieval.")
    df["__content"] = parts[0]
    for series in parts[1:]:
        df["__content"] = df["__content"] + " | " + series

    df["__row_id"] = range(len(df))

    ticker_values = []
    if ticker and ticker in df.columns:
        ticker_values = sorted(df[ticker].dropna().astype(str).unique().tolist())

    sector = _mapped("sector")
    sector_values = []
    if sector and sector in df.columns:
        sector_values = sorted(df[sector].dropna().astype(str).unique().tolist())

    topic_values = []
    if topic and topic in df.columns:
        topic_values = sorted(df[topic].dropna().astype(str).unique().tolist())

    source = _mapped("source")
    source_values = []
    if source and source in df.columns:
        source_values = sorted(df[source].dropna().astype(str).unique().tolist())

    missing = {column: int(count) for column, count in df.isna().sum().items() if count}
    valid_dates = df["__date"].dropna()
    date_min = valid_dates.min().strftime("%Y") if not valid_dates.empty else None
    date_max = valid_dates.max().strftime("%Y") if not valid_dates.empty else None

    def _has(logical: str) -> bool:
        column = _mapped(logical)
        return bool(column) and column in df.columns

    features = {
        "company": _has("ticker"),
        "topic": _has("event_type"),
        "sentiment": _has("sentiment"),
        "sector": _has("sector"),
        "source": _has("source"),
        "year": bool(df["__date"].notna().any()),
        "price": bool(_has("price_change_percent") and df[_mapped("price_change_percent")].notna().any()),
        "impact": _has("impact"),
        "direction": _has("price_direction"),
        "volume": _has("volume"),
    }

    info = DatasetInfo(
        rows=len(df),
        columns=[str(column) for column in df.columns if not str(column).startswith("__")],
        date_min=date_min,
        date_max=date_max,
        tickers=ticker_values,
        sectors=sector_values,
        topics=topic_values,
        missing=missing,
        file_hash=file_hash,
        duplicates_removed=duplicates_removed,
        sources=source_values,
        features=features,
    )
    return df, info
