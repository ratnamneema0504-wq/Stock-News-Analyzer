# Stock News Analyzer

**An AI-powered financial news analysis and market intelligence application.**

Stock News Analyzer is an interactive financial research application built with Python and Streamlit. It combines Natural Language Processing (NLP), semantic search, machine learning, and locally hosted Large Language Models (LLMs) to analyze historical financial news and generate meaningful market insights.

The application helps users explore financial news, understand market sentiment, identify significant financial events, and examine their relationship with historical stock price movements.

## Key Features

### 1. AI-Powered News Analysis
- Ask financial and stock-market-related questions through an interactive AI chat interface.
- Generate context-aware responses using Ollama and Llama 3.2.
- Retrieve relevant financial news to support AI-generated answers.

### 2. Semantic Search
- Uses Sentence Transformers to generate text embeddings.
- Implements FAISS for efficient similarity-based retrieval.
- Retrieves relevant news articles based on contextual meaning rather than exact keyword matching.

### 3. Sentiment Analysis
- Categorizes financial news into Positive, Negative, and Neutral sentiment.
- Analyzes sentiment distribution across companies and sectors.
- Examines relationships between historical sentiment and stock price movements.

### 4. Market Impact Analysis
- Examines historical stock price movements.
- Identifies top gainers and losers.
- Evaluates sector-wise performance.
- Analyzes historical price changes associated with financial news.

### 5. Interactive Dashboard
- Displays financial statistics and market insights.
- Visualizes historical financial news trends.
- Provides interactive charts and data filtering.
- Supports CSV dataset uploads.

### 6. Machine Learning Models

The application incorporates multiple machine learning techniques:

| Model | Application |
|---|---|
| Decision Tree Classifier | Historical stock price direction classification |
| K-Means Clustering | Grouping similar financial news articles |
| TF-IDF | Text feature extraction |
| Production Rules | Extracting interpretable IF-THEN rules |

### 7. Custom Dataset Upload
- Upload custom financial news datasets in CSV format.
- Validate dataset structure and required columns.
- Perform analysis using uploaded data.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| NLP | Sentence Transformers |
| Vector Search | FAISS |
| Generative AI | Ollama, Llama 3.2 |
| Visualization | Plotly |
| Testing | Pytest |

---

## Dataset

The project includes a historical financial news dataset containing:

- **15,000 records**
- **30 companies**
- **7 industry sectors**
- Historical coverage from **2019–2025**

### Dataset Attributes

The extended dataset contains 15 columns:

| Column | Description |
|---|---|
| news | Financial news headline |
| body | News article content |
| date | Historical news date |
| year | Publication year |
| company | Associated company |
| sector | Industry sector |
| source | News source |
| topic | Financial event category |
| sentiment | Positive, Negative, or Neutral |
| price_before | Recorded price before the news event |
| price_after | Recorded price after the news event |
| price_change_percent | Historical percentage price change |
| price_direction | Recorded price direction |
| impact | Historical impact classification |
| volume | Associated trading volume |

The project also includes the original 5,000-record dataset.

---

## Project Architecture

```text
Stock-News-Analyzer/
│
├── app.py
│
├── data/
│   ├── stock_market_news_extended_15000.csv
│   └── stock_market_news_analyzer_5000_project (1).csv
│
├── index/
│   ├── news.faiss
│   └── metadata.json
│
├── prompts/
│   └── system_prompt.txt
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── retriever.py
│   ├── analyzer.py
│   ├── sentiment.py
│   ├── market.py
│   ├── models.py
│   ├── charts.py
│   ├── ui.py
│   ├── local_ai.py
│   └── guardrail.py
│
├── tests/
│   ├── test_analyzer.py
│   ├── test_data_loader.py
│   ├── test_filters.py
│   ├── test_guardrail.py
│   ├── test_market.py
│   ├── test_models.py
│   └── test_retriever.py
│
├── tools/
│   └── generate_extended_dataset.py
│
├── Stock_News_Analyzer.ipynb
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- Ollama (for local AI functionality)

### 1. Clone the Repository

```bash
git clone https://github.com/ratnamneema0504-wq/Stock-News-Analyzer.git
```

Navigate to the project directory:

```bash
cd Stock-News-Analyzer
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

**Windows PowerShell:**

```powershell
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Ollama

Download the required model:

```bash
ollama pull llama3.2:3b
```

Start the Ollama service:

```bash
ollama serve
```

### 6. Configure Environment Variables

Create a `.env` file in the project root using `.env.example` as a reference.

Default configuration:

```env
OLLAMA_MODEL=llama3.2:3b
DATA_FILE=stock_market_news_extended_15000.csv
TOP_K_DEFAULT=8
SIMILARITY_THRESHOLD=0.30
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_ARTICLE_CHARS=6000
```

### 7. Run the Application

```bash
streamlit run app.py
```

Open your browser:

```text
http://localhost:8501
```

---

## Testing

The project includes automated tests covering data loading, semantic retrieval, market analysis, machine learning models, and application guardrails.

Run the test suite:

```bash
python -m pytest -q
```

---

## AI Grounding and Reliability

The application incorporates several mechanisms to improve the reliability of AI-generated responses:

- **Evidence-based retrieval:** Responses are grounded in relevant records retrieved from the available dataset.
- **Similarity threshold:** Queries with insufficient retrieval similarity are prevented from reaching the LLM.
- **Source transparency:** Retrieved news records are displayed alongside AI-generated responses.
- **Input validation:** Uploaded datasets are checked for required fields.
- **Prompt protection:** Retrieved news content is treated as untrusted input.
- **Local inference:** AI responses are generated using a locally hosted Ollama model.

---

## Limitations

- The application operates on historical datasets and does not fetch live financial news.
- Real-time stock prices and market feeds are not supported.
- Machine learning outputs describe historical patterns and should not be interpreted as reliable future predictions.
- AI-generated responses depend on the quality and coverage of the available dataset.
- Ollama is required for AI-generated chat responses in the standard local setup.

---

## Future Improvements

- Integration with live financial news APIs.
- Real-time stock market data integration.
- Advanced financial sentiment models.
- Improved predictive analytics.
- Support for additional financial datasets.
- Expanded company and sector intelligence.

---

## Disclaimer

This project is developed for educational, analytical, and research purposes.

Historical price movements and machine learning outputs represent observations from the available dataset. They do not establish causal relationships, guarantee future performance, or constitute financial or investment advice.

---

## Author

**Ratnam Neema**

**GitHub Repository:** https://github.com/ratnamneema0504-wq/Stock-News-Analyzer

**Deployed Application:** https://stock-newz-analyzer.streamlit.app/

---

**Stock News Analyzer — Turning Financial News into Actionable Insights.**
