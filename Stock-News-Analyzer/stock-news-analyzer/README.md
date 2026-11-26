# Stock News Analyzer

A Streamlit chat application that answers financial-news questions using **only the bundled CSV dataset**. It uses Sentence Transformers + FAISS for retrieval and Ollama for grounded natural-language answers.

## Bundled dataset (extended)

`data/stock_market_news_extended_15000.csv` is generated deterministically by `tools/generate_extended_dataset.py`:

- 15,000 rows and 15 columns
- Columns: `news`, `date`, `year`, `company`, `sector`, `source`, `topic`, `sentiment`, `price_before`, `price_after`, `price_change_percent`, `price_direction`, `impact`, `volume`, `body`
- Year range: 2019-2025 with full calendar dates
- 30 companies across 7 sectors (Banking, IT, Energy, Automobiles, FMCG, Pharmaceuticals, Financial Services)
- 7 news sources (Reuters, Economic Times, Moneycontrol, Bloomberg, Livemint, Business Standard, CNBC-TV18)
- Sentiment labels: Positive, Negative, Neutral; direction: Up/Down; impact: High/Medium/Low
- `body` is a short article body; optional at load time (uploaded CSVs without a body still work)
- The original 5,000-row CSV stays in `data/`; set `DATA_FILE` in `.env` to switch back.
- Regenerate anytime: `python tools/generate_extended_dataset.py` (same output every time).

## Structure

```text
stock-news-analyzer/
├── app.py
├── data/
│   └── stock_market_news_analyzer_5000_project (1).csv
├── index/
├── prompts/
│   └── system_prompt.txt
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── retriever.py
│   ├── analyzer.py
│   ├── sentiment.py
│   └── charts.py
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Windows / VS Code setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set your Ollama key and model:

```env
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=your_supported_ollama_model
```

Then run:

```powershell
streamlit run app.py
```

Open `http://localhost:8501`.

## Dataset configuration

`src/config.py` contains `COLUMN_MAP`. For the supplied dataset it maps:

```python
COLUMN_MAP = {
    "headline": "news",
    "text": "body",
    "date": "date",
    "ticker": "company",
    "sector": "sector",
    "source": "source",
    "sentiment": "sentiment",
    "event_type": "topic",
    "open": None,
    "high": None,
    "low": None,
    "close": None,
    "volume": "volume",
    "price_before": "price_before",
    "price_after": "price_after",
    "price_change_percent": "price_change_percent",
    "price_direction": "price_direction",
    "impact": "impact",
}
```

For a replacement dataset, swap the CSV, update `COLUMN_MAP`, and delete `index/news.faiss` and `index/metadata.json`. The next run rebuilds the index.

## Retrieval/index behavior

The application computes a SHA-256 hash of the CSV. If the file hash, row count, or embedding model changes, FAISS is rebuilt. Otherwise the saved index is reused.

## Guardrails

- Retrieval similarity is checked before calling Ollama.
- Below the configured threshold, the app answers: `The provided dataset doesn't contain information on this.`
- News content is passed as untrusted data; instructions inside it are ignored.
- Uploaded CSVs are validated (required columns: news, date or year, company) and only replace the dataset for the current session; invalid files fall back to the bundled dataset with a visible error.
- The app does not fetch external prices, news, or company facts.
- The disclaimer is appended to every answer.

## Recent upgrades

- **Dark theme** via `.streamlit/config.toml` plus restyled cards, tabs, chat bubbles and badges.
- **Chat like a modern AI app**: avatars, streaming with a cursor, multi-turn context (follow-up questions such as "and what was the market impact?" are answered with the previous turns passed to the model), Regenerate button, follow-up suggestion chips after every answer, and a Clear conversation button.
- **Add your own CSV**: sidebar file uploader swaps the active dataset for the session (validation + graceful fallback).
- **New filters**: Sector and News source (when the data provides them); sector-aware retrieval.
- **Top movers**: biggest gains/drops tables in Market Impact.
- **Sector analytics**: news volume by sector and average price change by sector on the Dashboard.

## Tests

```powershell
python -m pytest -q
```

## Known limitations

- The supplied dataset has year-level time resolution, not exact publication dates.
- There is no sector or source column.
- There are no OHLC columns; the supplied `price_before`, `price_after`, and `price_change_percent` fields are used for historical price analysis.
- Semantic retrieval quality depends on the dataset and embedding model.
- local Ollama AI access is required for generated answers.
- Historical price movement is observational and should not be interpreted as causal attribution or a forecast.


## AI financial research assistant (update)

- Put `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` in a `.env` file in the project root (see `.env.example`). The key is never displayed or logged. Without it, chat shows "AI analysis is currently unavailable. Please configure the local Ollama AI key." while the Dashboard, Market Impact and retrieval keep working.
- Guardrail: if the best retrieval similarity is below `SIMILARITY_THRESHOLD`, the LLM is not called and the app answers "The available evidence does not contain enough information to answer that."
- Market Impact figures are historical statistics from the CSV, never predictions.
- Dataset, embedding model and FAISS index are cached; the index is rebuilt only when the CSV hash changes.
- Run: `pip install -r requirements.txt`, then `streamlit run app.py`. Tests: `pytest`.


## Local AI setup

This version uses Ollama instead of a paid cloud API. Install Ollama, then run `ollama pull llama3.2:3b`. The app connects to the local model through the Ollama Python library. No Anthropic API key is required.

## AI Models view (course policy mapping)

A fourth view, **AI Models**, implements the AI concepts from the *Course Policy — Artificial Intelligence (2025-26)* directly against the bundled dataset using scikit-learn. It runs independently of Ollama.

- **CO4 · Unit 5 Learning, Exp. 9 — Decision Tree (supervised).** A decision-tree classifier predicts the post-news price *direction* (Up/Down) from sentiment, topic, company, year and trading volume. Shows test accuracy vs. the majority baseline, balanced accuracy, a confusion matrix and per-field feature importance, plus an interactive what-if predictor. Rare classes are dropped for balance and reported.
- **CO3 · Unit 3/6 — Knowledge Representation / Expert Systems.** The same trained tree is unwrapped into human-readable IF-THEN production rules with confidence and support — the forward-chaining expert-system view of the learned model.
- **CO4 · Unit 5 Learning, Exp. 10 — K-means (unsupervised).** K-means over TF-IDF text (2-D SVD projection) clusters the news without labels, with a silhouette-by-k chart for choosing k and a per-cluster profile table (size, dominant sentiment, average historical change).

New modules: `src/models.py` (fit + predict + rules), model charts in `src/charts.py`, and styling helpers in `src/ui.py`. Dependency `scikit-learn` was added to `requirements.txt`. All output is descriptive modelling of past observations, not investment advice.
