"""Generate the extended stock-news dataset used by the app.

Deterministic (fixed seed) so regenerating always yields the same CSV.
Style follows the bundled 5,000-row dataset (simple, declarative headlines)
and adds detail columns: full date, year, sector, source, body.

Usage (from the project root):
    python tools/generate_extended_dataset.py
"""
from __future__ import annotations

import datetime as dt
import random
from pathlib import Path

import pandas as pd

SEED = 20261011
ROWS = 15000
OUT = Path(__file__).resolve().parent.parent / "data" / "stock_market_news_extended_15000.csv"

# company -> (sector, base price level in INR)
COMPANIES = {
    "HDFC Bank": ("Banking", 1450), "ICICI Bank": ("Banking", 900),
    "Axis Bank": ("Banking", 950), "Kotak Mahindra Bank": ("Banking", 1700),
    "State Bank of India": ("Banking", 620),
    "TCS": ("Information Technology", 3600), "Infosys": ("Information Technology", 1500),
    "Wipro": ("Information Technology", 450), "HCL Technologies": ("Information Technology", 1100),
    "Tech Mahindra": ("Information Technology", 1250),
    "Reliance Industries": ("Energy", 2800), "NTPC": ("Energy", 350),
    "Power Grid Corporation": ("Energy", 320), "Adani Enterprises": ("Energy", 2900),
    "Tata Power": ("Energy", 380),
    "Maruti Suzuki": ("Automobiles", 12500), "Tata Motors": ("Automobiles", 720),
    "Mahindra & Mahindra": ("Automobiles", 2100), "Bajaj Auto": ("Automobiles", 8500),
    "Eicher Motors": ("Automobiles", 3400),
    "ITC": ("FMCG", 430), "Hindustan Unilever": ("FMCG", 2500),
    "Nestle India": ("FMCG", 23000), "Asian Paints": ("FMCG", 2800),
    "Dabur India": ("FMCG", 520),
    "Sun Pharma": ("Pharmaceuticals", 1400), "Dr Reddy's Laboratories": ("Pharmaceuticals", 6200),
    "Cipla": ("Pharmaceuticals", 1350),
    "Bajaj Finance": ("Financial Services", 6800), "SBI Life Insurance": ("Financial Services", 1350),
}

TOPICS = [
    "Earnings", "Costs", "Competition", "Partnership", "Regulatory",
    "Corporate Update", "Investor Relations", "Outlook", "Expansion",
    "New Order", "Management", "Sales Growth", "Sales Decline", "General",
]

SOURCES = [
    "Reuters", "Economic Times", "Moneycontrol", "Bloomberg",
    "Livemint", "Business Standard", "CNBC-TV18",
]

HEADLINES = {
    "Positive": [
        "announces improved profit as operating performance strengthens.",
        "posts better-than-expected earnings for the quarter.",
        "reduces operating costs and improves profit margins.",
        "reports growth in sales compared with the previous quarter.",
        "receives a major new order, improving its business outlook.",
        "announces a strategic partnership to grow its business.",
        "expands capacity after strong demand in its main markets.",
        "raises its annual guidance following a strong quarter.",
    ],
    "Negative": [
        "faces increased competition in an important market.",
        "warns that weak demand may affect near-term growth.",
        "faces higher operating costs that put pressure on margins.",
        "reports lower revenue in one of its major business segments.",
        "reduces its growth outlook for the coming quarter.",
        "faces regulatory scrutiny over recent business practices.",
        "announces restructuring after a weak performance.",
        "loses market share to rivals in its core business.",
    ],
    "Neutral": [
        "publishes its quarterly financial results.",
        "schedules its annual general meeting for next month.",
        "publishes an update regarding its existing business plans.",
        "announces a routine management reshuffle.",
        "confirms that previously announced plans are on track.",
        "files its annual report with the market regulator.",
        "hosts a conference call with analysts this week.",
        "announces the date of its next board meeting.",
    ],
}

# Some topics carry a natural sentiment lean; used to keep the data coherent.
TOPIC_LEAN = {
    "Earnings": 0.45, "Costs": -0.25, "Competition": -0.45, "Partnership": 0.35,
    "Regulatory": -0.20, "Corporate Update": 0.0, "Investor Relations": 0.0,
    "Outlook": -0.15, "Expansion": 0.45, "New Order": 0.55, "Management": 0.0,
    "Sales Growth": 0.60, "Sales Decline": -0.60, "General": 0.0,
}

BODY_SENTENCES = {
    "Positive": "Market participants viewed the update as a sign of improving fundamentals in the {sector} sector.",
    "Negative": "Analysts said the development could weigh on expectations for the coming quarters in the {sector} sector.",
    "Neutral": "The update did not change consensus estimates for the {sector} sector, according to brokerage notes.",
}


def pick_sentiment(rng: random.Random, topic: str) -> str:
    lean = TOPIC_LEAN[topic]
    r = rng.random()
    # lean > 0 biases Positive, lean < 0 biases Negative; 0 stays balanced.
    if lean >= 0:
        p_pos, p_neg = 0.34 + 0.45 * lean, 0.33 - 0.30 * lean
    else:
        p_pos, p_neg = 0.34 + 0.30 * lean, 0.33 - 0.45 * lean
    if r < p_pos:
        return "Positive"
    if r < p_pos + p_neg:
        return "Negative"
    return "Neutral"


def pick_direction(rng: random.Random, sentiment: str) -> str:
    if sentiment == "Positive":
        return "Up" if rng.random() < 0.90 else "Down"
    if sentiment == "Negative":
        return "Down" if rng.random() < 0.90 else "Up"
    return "Up" if rng.random() < 0.5 else "Down"


def pick_impact(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.45:
        return "Low"
    if r < 0.77:
        return "Medium"
    return "High"


def change_for(rng: random.Random, impact: str, direction: str) -> float:
    if impact == "High":
        mag = rng.uniform(5.0, 12.0)
    elif impact == "Medium":
        mag = rng.uniform(1.5, 5.0)
    else:
        mag = rng.uniform(0.05, 1.5)
    value = mag if direction == "Up" else -mag
    return round(value, 2)


def main() -> None:
    rng = random.Random(SEED)
    start = dt.date(2019, 1, 1)
    end = dt.date(2025, 12, 31)
    span_days = (end - start).days

    names = list(COMPANIES)
    rows = []
    for i in range(ROWS):
        company = rng.choice(names)
        sector, base = COMPANIES[company]
        topic = rng.choice(TOPICS)
        sentiment = pick_sentiment(rng, topic)
        headline = f"{company} {rng.choice(HEADLINES[sentiment])}"
        headline = headline.rstrip(".") + "."
        date = start + dt.timedelta(days=rng.randrange(span_days))
        year = date.year
        # gentle per-company drift so prices differ by year
        drift = 1.0 + 0.07 * (year - 2019) + rng.uniform(-0.15, 0.15)
        price_before = round(base * drift, 2)
        impact = pick_impact(rng)
        direction = pick_direction(rng, sentiment)
        change = change_for(rng, impact, direction)
        price_after = round(price_before * (1 + change / 100.0), 2)
        volume = int(rng.lognormvariate(15.2, 1.1))
        volume = max(50_000, min(volume, 30_000_000))
        source = rng.choice(SOURCES)
        body = (
            f"{headline} {BODY_SENTENCES[sentiment].format(sector=sector)} "
            f"The share price moved {abs(change):.2f}% in the session after the report, "
            f"with {volume:,} shares traded. Reported by {source}."
        )
        rows.append({
            "news": headline,
            "date": date.isoformat(),
            "year": year,
            "company": company,
            "sector": sector,
            "source": source,
            "topic": topic,
            "sentiment": sentiment,
            "price_before": price_before,
            "price_after": price_after,
            "price_change_percent": change,
            "price_direction": direction,
            "impact": impact,
            "volume": volume,
            "body": body,
        })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows to {OUT.name}")
    print("Companies:", df['company'].nunique(), "| Sectors:", df['sector'].nunique(),
          "| Years:", sorted(df['year'].unique()))
    print("Sentiment:", df['sentiment'].value_counts().to_dict())
    print("Direction:", df['price_direction'].value_counts().to_dict())
    print("Accuracy of direction sign vs change:",
          ((df['price_change_percent'] > 0) == (df['price_direction'] == 'Up')).mean())


if __name__ == "__main__":
    main()
