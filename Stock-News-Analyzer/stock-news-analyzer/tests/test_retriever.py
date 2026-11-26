import numpy as np
import pandas as pd

from src.retriever import Retriever


class FakeIndex:
    def __init__(self, vectors):
        self.v, self.ntotal = vectors, len(vectors)

    def search(self, q, n):
        sims = self.v @ q[0]
        order = np.argsort(-sims)[:n]
        return np.array([sims[order]]), np.array([order])


class FakeModel:
    def encode(self, texts, **kw):
        return np.array([[1.0, 0.0] for _ in texts], dtype="float32")


def _retriever():
    df = pd.DataFrame({
        "news": ["a", "b", "c"], "company": ["X", "Y", "X"], "topic": ["Earnings", "Costs", "Costs"],
        "__date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
    })
    r = Retriever.__new__(Retriever)
    r.df, r.model = df, FakeModel()
    r.index = FakeIndex(np.array([[1, 0], [0.9, 0.1], [0.5, 0.5]], dtype="float32"))
    return r


def test_search_returns_scores_sorted():
    rows = _retriever().search("q", k=3)
    assert [r["news"] for r in rows] == ["a", "b", "c"]
    assert "__score" in rows[0]


def test_company_topic_and_year_filters():
    r = _retriever()
    assert [x["news"] for x in r.search("q", 5, ticker="X")] == ["a", "c"]
    assert [x["news"] for x in r.search("q", 5, topic="Costs")] == ["b", "c"]
    assert [x["news"] for x in r.search("q", 5, start=pd.Timestamp("2021-01-01"), end=pd.Timestamp("2021-12-31"))] == ["b"]


def test_blank_query_returns_nothing():
    assert _retriever().search("   ", 3) == []
