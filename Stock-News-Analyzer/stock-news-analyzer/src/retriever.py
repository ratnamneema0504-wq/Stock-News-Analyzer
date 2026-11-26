from __future__ import annotations

import json

import pandas as pd

from .config import COLUMN_MAP, EMBEDDING_MODEL, INDEX_DIR
from .data_loader import DatasetInfo


class RetrievalError(Exception):
    """Raised when the embedding model or FAISS index cannot be prepared or queried."""


def load_embedding_model():
    """Load the sentence-transformer once (callers cache the result)."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as exc:
        raise RetrievalError("The semantic search model could not be loaded.") from exc


class Retriever:
    """FAISS-backed semantic search. Build once, reuse for every question."""

    def __init__(self, df: pd.DataFrame, info: DatasetInfo, model=None):
        self.df = df.reset_index(drop=True)
        self.info = info
        self.model = model if model is not None else load_embedding_model()
        try:
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RetrievalError("The search index folder is not writable.") from exc
        self.index_path = INDEX_DIR / "news.faiss"
        self.meta_path = INDEX_DIR / "metadata.json"
        self.index = self._load_or_build()

    def _load_or_build(self):
        try:
            import faiss
        except Exception as exc:
            raise RetrievalError("The FAISS search library is not available.") from exc

        if self.index_path.exists() and self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if (
                    meta.get("file_hash") == self.info.file_hash
                    and meta.get("rows") == len(self.df)
                    and meta.get("model") == EMBEDDING_MODEL
                ):
                    index = faiss.read_index(str(self.index_path))
                    if index.ntotal == len(self.df):
                        return index
            except Exception:
                pass  # stale or corrupt index -> rebuild below

        try:
            embeddings = self.model.encode(
                self.df["__content"].tolist(),
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
            ).astype("float32")
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
        except Exception as exc:
            raise RetrievalError("The search index could not be built.") from exc

        try:
            faiss.write_index(index, str(self.index_path))
            self.meta_path.write_text(
                json.dumps({"file_hash": self.info.file_hash, "rows": len(self.df), "model": EMBEDDING_MODEL}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass  # index still works in memory; just not persisted
        return index

    def search(
        self,
        query: str,
        k: int,
        ticker: str | None = None,
        topic: str | None = None,
        sector: str | None = None,
        start=None,
        end=None,
    ) -> list[dict]:
        if not query.strip() or self.index.ntotal == 0:
            return []
        try:
            query_vector = self.model.encode([query], normalize_embeddings=True).astype("float32")
            has_filter = any(v is not None for v in (ticker, topic, sector, start, end))
            candidate_count = self.index.ntotal if has_filter else min(max(k * 20, 100), self.index.ntotal)
            scores, ids = self.index.search(query_vector, candidate_count)
        except Exception as exc:
            raise RetrievalError("Semantic search failed.") from exc

        ticker_col = COLUMN_MAP.get("ticker")
        topic_col = COLUMN_MAP.get("event_type")
        sector_col = COLUMN_MAP.get("sector")
        rows: list[dict] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            row = self.df.iloc[int(idx)]
            if ticker and ticker_col and ticker_col in self.df.columns and str(row[ticker_col]) != ticker:
                continue
            if topic and topic_col and topic_col in self.df.columns and str(row[topic_col]) != topic:
                continue
            if sector and sector_col and sector_col in self.df.columns and str(row[sector_col]) != sector:
                continue
            row_date = row.get("__date")
            if start is not None and pd.notna(row_date) and row_date < start:
                continue
            if end is not None and pd.notna(row_date) and row_date > end:
                continue
            item = row.to_dict()
            item["__score"] = float(score)
            rows.append(item)
            if len(rows) >= k:
                break
        return rows
