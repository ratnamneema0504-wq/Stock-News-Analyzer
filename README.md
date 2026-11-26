# Stock News Analyzer

**An AI-powered financial news intelligence and market analysis platform.**

Stock News Analyzer is an interactive financial research application built with Python and Streamlit. It combines Natural Language Processing (NLP), semantic search, machine learning, and locally hosted Large Language Models (LLMs) to transform historical financial news into meaningful market insights.

The platform allows users to explore financial news, investigate sentiment trends, examine historical stock price movements, and interact with an AI assistant that generates context-aware answers grounded in the available dataset.

---

## Key Features

### 1. AI-Powered Financial Chat
- Ask natural-language questions about companies, financial events, and market trends.
- Generate context-aware responses using Ollama and Llama 3.2.
- Maintain conversational context for follow-up questions.
- Display supporting news records alongside generated answers.
- Regenerate responses and explore suggested follow-up questions.

### 2. Semantic News Search
- Uses Sentence Transformers to generate text embeddings.
- Implements FAISS for efficient similarity-based retrieval.
- Retrieves relevant financial news based on contextual meaning rather than exact keyword matching.
- Supports company, sector, topic, and date-based filtering.

### 3. Sentiment Analysis
- Examines Positive, Negative, and Neutral sentiment classifications.
- Explores sentiment distribution across companies and sectors.
- Connects historical sentiment observations with recorded price movements.

### 4. Interactive Financial Dashboard
- Displays article volume and company statistics.
- Visualizes historical financial news trends.
- Provides sector-wise news analysis.
- Examines average historical price changes.
- Supports interactive filtering and CSV export.

### 5. Market Impact Analysis
- Identifies historical stock price movements associated with financial news.
- Highlights top gainers and losers.
- Examines the relationship between news sentiment and historical market performance.
- Provides sector-level market impact comparisons.

### 6. Machine Learning Models

The application incorporates multiple machine learning techniques:

| Model | Application |
|---|---|
| Decision Tree Classifier | Historical stock price direction classification |
| K-Means Clustering | Grouping similar financial news articles |
| TF-IDF | Text feature extraction for news clustering |
| Production Rules | Extracting interpretable IF-THEN rules from the trained decision tree |

The application also provides model evaluation metrics, confusion matrices, feature importance, and interactive prediction functionality.

### 7. Custom Dataset Support
- Upload custom financial news datasets in CSV format.
- Validate dataset structure and required columns.
- Perform analysis using uploaded data within the current session.
- Automatically rebuild the semantic search index when the underlying dataset changes.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Natural Language Processing | Sentence Transformers |
| Vector Search | FAISS |
| Generative AI | Ollama, Llama 3.2 |
| Data Visualization | Plotly |
| Environment Management | Python Virtual Environment |
| Testing | Pytest |

---

## Dataset

The project includes a historical financial news dataset containing:

- **15,000 records**
- **30 companies**
- **7 industry sectors**
- **7 financial news sources**
- Historical coverage from **2019 to 2025**

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

The project also retains the original 5,000-record dataset.

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

Make sure the following are installed:

- Python 3.10 or later
- Git
- Ollama

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

Install Ollama and download the required model:

```bash
ollama pull llama3.2:3b
```

Make sure the Ollama service is running:

```bash
ollama serve
```

The application uses the locally hosted model for AI-generated responses. No paid cloud API key is required.

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

Open the following address in your browser:

```text
http://localhost:8501
```

---

## Testing

The project includes automated tests covering data loading, semantic retrieval, market analysis, machine learning models, and application guardrails.

Run the test suite using:

```bash
python -m pytest -q
```

---

## AI Grounding and Reliability

Stock News Analyzer incorporates several mechanisms to improve the reliability of AI-generated responses:

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
- Ollama and the configured local model are required for AI-generated chat responses.

---

## Future Improvements

Potential areas for further development include:

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

GitHub: [@ratnamneema0504-wq](https://github.com/ratnamneema0504-wq)

---

**Stock News Analyzer — Turning Financial News into Actionable Insights.**
