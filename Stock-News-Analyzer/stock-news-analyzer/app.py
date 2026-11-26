from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.config import (  # importing config also loads the project-root .env
    AI_UNAVAILABLE, COLUMN_MAP, DISCLAIMER, SIMILARITY_THRESHOLD, TOP_K_DEFAULT,
    ai_ready, get_model,
)

st.set_page_config(page_title="Stock News Analyzer", page_icon="📈", layout="wide")

from src import charts
from src.analyzer import AIUnavailableError, stream_answer
from src.data_loader import DatasetError, clean_csv_bytes, load_and_clean
from src.guardrail import has_sufficient_evidence, insufficient_evidence_message
from src.market import impact_by_group, market_impact_available, market_impact_stats, news_profile_text
from src.retriever import RetrievalError, Retriever, load_embedding_model
from src.ui import CSS, co_tag, dir_badge, fmt_pct, kpi_card, panel, rule_card, status_pill

st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- cached resources
@st.cache_data(show_spinner=False)
def get_dataset():
    df, info, path = load_and_clean()
    return df, info, path.name


@st.cache_data(show_spinner="Reading your uploaded CSV...")
def get_uploaded_dataset(raw_bytes: bytes, name: str):
    df, info = clean_csv_bytes(raw_bytes)
    return df, info, name


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource(show_spinner=False)
def get_retriever(file_hash: str, _df, _info):
    # Keyed by dataset hash: rebuilt only when the CSV changes.
    return Retriever(_df, _info, model=get_embedding_model())


@st.cache_data(show_spinner="Training decision-tree model...")
def get_direction_bundle(file_hash: str, _df, _info):
    from src import models
    return models.train_direction_model(_df)


@st.cache_data(show_spinner="Clustering news (K-means)...")
def get_cluster_bundle(file_hash: str, _df, _info, k: int):
    from src import models
    return models.train_clusters(_df, k)


@st.cache_data(show_spinner=False)
def get_cluster_quality(file_hash: str, _df, _info):
    from src import models
    return models.cluster_quality(_df)


# ----------------------------------------------------------------- header
st.markdown(
    '<div class="sna-hero"><h1>Stock News Analyzer</h1>'
    "<p>AI-powered financial news &amp; market impact analysis</p>"
    '<div class="sna-chips">'
    '<span class="sna-chip">RAG Chat</span>'
    '<span class="sna-chip">Market Impact</span>'
    '<span class="sna-chip">Decision Tree (CO4)</span>'
    '<span class="sna-chip">K-means (CO4)</span>'
    '<span class="sna-chip">Rule Extraction (CO3)</span>'
    '</div></div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- dataset selection (bundled or uploaded)
upload_error: str | None = None
with st.sidebar:
    st.markdown('<div class="sna-side-title">Data source</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Add your own news CSV",
        type=["csv"],
        help="Replaces the bundled dataset for this session. Needs at least: news, year or date, company, topic, sentiment.",
    )

if uploaded is not None:
    try:
        df, info, data_name = get_uploaded_dataset(uploaded.getvalue(), uploaded.name)
        data_source = f"Uploaded · {data_name}"
    except DatasetError as exc:
        upload_error = str(exc)
        df, info, data_name = get_dataset()
        data_source = "Bundled dataset (upload rejected)"
    except Exception:
        upload_error = "The uploaded file could not be read."
        df, info, data_name = get_dataset()
        data_source = "Bundled dataset (upload rejected)"
else:
    try:
        df, info, data_name = get_dataset()
        data_source = f"Bundled · {data_name}"
    except DatasetError as exc:
        st.error(f"The dataset could not be loaded. {exc}")
        st.info("Check the CSV in the data/ folder, add your own CSV in the sidebar, and restart the app.")
        st.stop()
        raise SystemExit
    except Exception:
        st.error("The dataset could not be loaded due to an unexpected problem. Please check the CSV file.")
        st.stop()
        raise SystemExit

if upload_error:
    st.error(f"Your uploaded CSV was not usable: {upload_error} The bundled dataset is being used instead.")
elif uploaded is not None:
    st.success(f"Using your uploaded dataset: **{data_name}** ({info.rows:,} rows).")

retriever: Retriever | None = None
retrieval_problem: str | None = None
try:
    with st.spinner("Preparing semantic search..."):
        retriever = get_retriever(info.file_hash, df, info)
except RetrievalError as exc:
    retrieval_problem = str(exc)
except Exception:
    retrieval_problem = "Semantic search could not be started."

F = info.features
COMPANY, TOPIC, SENT = COLUMN_MAP["ticker"], COLUMN_MAP["event_type"], COLUMN_MAP["sentiment"]
CHANGE, IMPACT = COLUMN_MAP["price_change_percent"], COLUMN_MAP["impact"]
HEADLINE = COLUMN_MAP["headline"]
SECTOR, SOURCE = COLUMN_MAP.get("sector"), COLUMN_MAP.get("source")

# AI models (scikit-learn) — optional; the rest of the app works without them.
try:
    from src import models as _models
    direction_available = _models.direction_model_available(df)
    cluster_available = _models.clustering_available(df)
except Exception:
    _models = None
    direction_available = cluster_available = False

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(status_pill(ai_ready()), unsafe_allow_html=True)
    if ai_ready():
        st.caption(f"Model: {get_model()}")
    else:
        st.caption("Model: not configured")
    st.caption(f"Data: {data_source}")
    st.write("")

    company = "All"
    if F.get("company"):
        company = st.selectbox("Company", ["All"] + info.tickers)
    topic = "All"
    if F.get("topic"):
        topic = st.selectbox("Topic", ["All"] + info.topics)
    sector = "All"
    if F.get("sector") and info.sectors:
        sector = st.selectbox("Sector", ["All"] + info.sectors)
    source = "All"
    if F.get("source") and info.sources:
        source = st.selectbox("News source", ["All"] + info.sources)
    start = end = None
    year_range = None
    if F.get("year"):
        years = sorted(df["__date"].dt.year.dropna().astype(int).unique().tolist())
        if len(years) > 1:
            year_range = st.select_slider("Year", options=years, value=(years[0], years[-1]))
        else:
            year_range = (years[0], years[0])
        start = pd.Timestamp(year=year_range[0], month=1, day=1)
        end = pd.Timestamp(year=year_range[1], month=12, day=31, hour=23, minute=59, second=59)
    k = st.slider("Top-k", 1, 20, TOP_K_DEFAULT)

    st.divider()
    models_enabled = direction_available or cluster_available
    views = ["Chat", "Dashboard"] + (["Market Impact"] if market_impact_available(df) else []) + (["AI Models"] if models_enabled else [])
    view = st.radio("View", views, label_visibility="collapsed")

    st.divider()
    st.caption("Export")
    export_slot = st.container()
    if st.button("Clear conversation", use_container_width=True, key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Data Info", expanded=False):
        st.write(f"Articles: {info.rows:,}")
        st.write(f"Years: {info.date_min or 'N/A'} - {info.date_max or 'N/A'}")
        st.write(f"Companies: {len(info.tickers)}  |  Topics: {len(info.topics)}")
        if info.sectors:
            st.write(f"Sectors: {len(info.sectors)}")
        if info.sources:
            st.write(f"News sources: {len(info.sources)}")
        st.write(f"File: `{data_name}`")
        st.write(f"Retrieval threshold: {SIMILARITY_THRESHOLD:.2f}")
        if info.duplicates_removed:
            st.caption(f"{info.duplicates_removed} duplicate rows removed at load time.")
        st.caption("Missing values: none" if not info.missing else f"Missing values in {len(info.missing)} field(s).")

# ----------------------------------------------------------------- filtered view of the data
fdf = df
if company != "All":
    fdf = fdf[fdf[COMPANY] == company]
if topic != "All":
    fdf = fdf[fdf[TOPIC] == topic]
if sector != "All" and F.get("sector"):
    fdf = fdf[fdf[COLUMN_MAP["sector"]] == sector]
if source != "All" and F.get("source"):
    fdf = fdf[fdf[COLUMN_MAP["source"]] == source]
if start is not None:
    fdf = fdf[(fdf["__date"] >= start) & (fdf["__date"] <= end)]

with export_slot:
    st.download_button(
        "Filtered data (CSV)",
        fdf[[c for c in df.columns if not c.startswith("__")]].to_csv(index=False),
        file_name="stock_news_filtered.csv", mime="text/csv", use_container_width=True,
    )

st.session_state.setdefault("messages", [])

USER_AVATAR = "🧑"
AI_AVATAR = "🤖"


# ----------------------------------------------------------------- chat helpers
def source_lines(rows: list[dict]) -> list[str]:
    lines = []
    for r in rows:
        parts = [f"**{r.get(COLUMN_MAP['headline'], '')}**"]
        meta = [str(r.get(c)) for c in (COMPANY, SECTOR, TOPIC, SENT) if c and c in r]
        if "year" in r:
            meta.append(str(r["year"]))
        if CHANGE in r and pd.notna(r[CHANGE]):
            meta.append(f"{float(r[CHANGE]):+.2f}%")
        if IMPACT in r:
            meta.append(f"{r[IMPACT]} impact")
        meta.append(f"similarity {r['__score']:.2f}")
        lines.append(f"- {parts[0]}  \n  <sub>{' - '.join(meta)}</sub>")
    return lines


def render_sources(sources: list[str]):
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            st.markdown("\n".join(sources), unsafe_allow_html=True)


def retrieve(prompt: str) -> list[dict]:
    """Retrieve evidence; if several companies are named (comparison), balance results across them."""
    common = dict(topic=None if topic == "All" else topic, start=start, end=end,
                  sector=None if sector == "All" else sector)
    named = [c for c in info.tickers if c.lower() in prompt.lower()] if company == "All" else []
    if len(named) >= 2:
        per = max(2, k // len(named))
        rows: list[dict] = []
        for name in named:
            rows += retriever.search(prompt, k=per, ticker=name, **common)
        return sorted(rows, key=lambda r: r["__score"], reverse=True)
    target = company if company != "All" else (named[0] if named else None)
    return retriever.search(prompt, k=k, ticker=target, **common)


def _stats_for(prompt: str):
    """Aggregate the filtered dataset so the model analyses numbers instead of restating text."""
    try:
        named = [c for c in info.tickers if c.lower() in prompt.lower()]
        return news_profile_text(fdf, named or None)
    except Exception:
        return None


def handle_prompt(prompt: str):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("_Searching the news dataset…_")
        stats_text = _stats_for(prompt)
        try:
            results = retrieve(prompt)
        except RetrievalError as exc:
            msg = f"Search is currently unavailable. {exc}"
            placeholder.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "sources": [], "prompt": prompt})
            return

        # Guardrail: below-threshold evidence => the LLM is never called.
        if not has_sufficient_evidence(results):
            if stats_text:
                msg = (
                    "No single news item matches that wording closely enough to quote, but here is what the "
                    f"filtered data as a whole shows:\n\n{stats_text}\n\n{DISCLAIMER}"
                )
                placeholder.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg, "sources": [], "prompt": prompt})
                return
            msg = insufficient_evidence_message()
            placeholder.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "sources": [], "prompt": prompt})
            return

        sources = source_lines(results)
        chunks = []
        placeholder.markdown("")
        try:
            for chunk in stream_answer(prompt, results, history=history, stats_text=stats_text):
                chunks.append(chunk)
                placeholder.markdown("".join(chunks) + " ▌")
            response = "".join(chunks).strip() + f"\n\n{DISCLAIMER}"
        except AIUnavailableError as exc:
            response = f"{exc}\n\nThe most relevant records were still retrieved and are listed under Sources.\n\n{DISCLAIMER}"
        placeholder.markdown(response)
        render_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources, "prompt": prompt})


def scoped(text: str) -> str:
    return text + (f" Focus on {company}." if company != "All" else "")


def contextualize(text: str) -> str:
    """Carry the previous question into a follow-up chip so retrieval always finds evidence."""
    for m in reversed(st.session_state.messages):
        if m["role"] == "user":
            base = m.get("prompt") or m["content"]
            return scoped(f"{text} about: {base}")
    return scoped(text)


EXAMPLES = [
    "Analyze TCS's recent news",
    "Compare banking and IT sector news",
    "Which topics caused the largest historical price movements?",
    "Show negative news with high market impact",
    "How did Infosys stock react to earnings news?",
    "Find similar historical events",
    "Summarize sentiment trends by year",
    "Which companies had the biggest market moves?",
]
QUICK_ACTIONS = {
    "Company Overview": "Give an overview of the news, sentiment and historical price impact for this company.",
    "Compare Companies": "Compare the companies in the retrieved news by sentiment, topics and historical price impact.",
    "Sentiment Analysis": "Summarize the sentiment of the news and how it relates to historical price changes.",
    "Historical Impact": "Summarize the historical price impact observed after the news.",
    "Similar News": "Find and summarize similar historical news events.",
    "Market Movers": "Which news items were associated with the largest historical price movements?",
}

# ----------------------------------------------------------------- views
if view == "Chat":
    if retrieval_problem:
        st.warning(f"{retrieval_problem} Chat is unavailable, but the Dashboard and Market Impact views still work.")

    cols = st.columns(len(QUICK_ACTIONS))
    for col, (label, text) in zip(cols, QUICK_ACTIONS.items()):
        if col.button(label, use_container_width=True, key=f"qa_{label}", disabled=retriever is None):
            st.session_state.pending = scoped(text)
            st.rerun()

    typed = st.chat_input(
        "Ask about a company, market event, sentiment, or historical price impact...",
        disabled=retriever is None,
    )
    prompt = typed or st.session_state.pop("pending", None)

    if not st.session_state.messages and not prompt:
        st.markdown('<div class="sna-empty"><h3>Ask anything about the news dataset</h3>'
                    '<p style="opacity:.7;margin-top:.1rem">Follow-up questions keep context, just like a chat. '
                    "Answers are grounded strictly in the bundled news data.</p></div>",
                    unsafe_allow_html=True)
        ex_cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if ex_cols[i % 2].button(ex, use_container_width=True, key=f"ex_{i}", disabled=retriever is None):
                st.session_state.pending = ex
                st.rerun()
        if not ai_ready():
            st.info(AI_UNAVAILABLE)

    for m in st.session_state.messages:
        avatar = "🤖" if m["role"] == "assistant" else "🧑"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_sources(m.get("sources", []))

    if prompt and retriever is not None:
        handle_prompt(prompt)

    msgs = st.session_state.messages
    if msgs:
        last = msgs[-1]
        followups = ["Explain the main drivers", "Show similar historical events",
                     "What was the market impact?", "Summarize in three bullets"]
        cols = st.columns([1.05, 1.35, 1.7, 1.45, 1.55, 2.5])
        if cols[0].button("Regenerate", key="regen", disabled=not (
                last["role"] == "assistant" and last.get("prompt") and retriever is not None)):
            st.session_state.messages = msgs[:-1]
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()
            st.session_state.pending = last["prompt"]
            st.rerun()
        for i, text in enumerate(followups):
            if cols[i + 1].button(text, key=f"fu_{i}", disabled=retriever is None):
                st.session_state.pending = contextualize(text)
                st.rerun()

elif view == "Dashboard":
    if fdf.empty:
        st.info("No articles match the current filters.")
    else:
        stats = market_impact_stats(fdf) if F.get("price") else None
        c = st.columns(4)
        c[0].markdown(kpi_card("Articles", f"{len(fdf):,}"), unsafe_allow_html=True)
        c[1].markdown(kpi_card("Companies", f"{fdf[COMPANY].nunique()}" if F.get("company") else "n/a"), unsafe_allow_html=True)
        c[2].markdown(kpi_card("Years", f"{fdf['__date'].dt.year.nunique()}" if F.get("year") else "n/a"), unsafe_allow_html=True)
        if stats:
            c[3].markdown(kpi_card("Average Price Change", fmt_pct(stats["average"]), "historical", "pos" if stats["average"] >= 0 else "neg"), unsafe_allow_html=True)
        else:
            c[3].markdown(kpi_card("Average Price Change", "n/a"), unsafe_allow_html=True)
        st.write("")

        def show(fig):
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        a, b = st.columns(2)
        with a:
            show(charts.news_count_by_ticker(fdf))
        with b:
            show(charts.sentiment_split(fdf))
        show(charts.sentiment_over_time(fdf))
        if F.get("sector"):
            a, b = st.columns(2)
            with a:
                show(charts.sector_news_volume(fdf))
            with b:
                show(charts.sector_impact(fdf))
        a, b = st.columns(2)
        with a:
            show(charts.price_impact(fdf))
        with b:
            show(charts.impact_by_topic(fdf))
        st.caption("Historical observations from the bundled dataset. " + DISCLAIMER)

elif view == "Market Impact":
    st.subheader("Market Impact")
    st.markdown(
        '<div class="sna-banner"><b>Historical analysis.</b> These figures describe price changes recorded in the '
        "dataset after past news. They are not predictions and not investment advice.</div>",
        unsafe_allow_html=True,
    )
    stats = market_impact_stats(fdf)
    if not stats:
        st.info("No price observations match the current filters.")
    else:
        c = st.columns(4)
        c[0].markdown(kpi_card("Average price change", fmt_pct(stats["average"]), f"{stats['observations']:,} observations", "pos" if stats["average"] >= 0 else "neg"), unsafe_allow_html=True)
        c[1].markdown(kpi_card("Median price change", fmt_pct(stats["median"]), "", "pos" if stats["median"] >= 0 else "neg"), unsafe_allow_html=True)
        c[2].markdown(kpi_card("Positive movement", fmt_pct(stats["positive_pct"], signed=False), "of observations", "pos"), unsafe_allow_html=True)
        c[3].markdown(kpi_card("Negative movement", fmt_pct(stats["negative_pct"], signed=False), "of observations", "neg"), unsafe_allow_html=True)
        st.write("")
        a, b = st.columns(2)
        with a:
            fig = charts.impact_distribution(fdf) if F.get("impact") else None
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            fig = charts.price_impact(fdf)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = charts.impact_by_topic(fdf)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        fig = charts.price_by_company(fdf)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        if F.get("impact"):
            summary = impact_by_group(fdf, "impact")
            if summary is not None:
                st.caption("Recorded impact category vs. historical price change")
                st.dataframe(
                    summary.rename(columns={"observations": "Observations", "mean_change": "Mean %", "median_change": "Median %", "mean_abs_change": "Mean |%|"}).round(2),
                    use_container_width=True, hide_index=True,
                )
        st.write("")
        st.markdown("#### Top movers")
        movers = fdf.dropna(subset=[CHANGE])
        if not movers.empty:
            disp_cols = [c for c in (HEADLINE, COMPANY, SECTOR, CHANGE, IMPACT, "year") if c in movers.columns]
            ren = {HEADLINE: "News", COMPANY: "Company", SECTOR: "Sector",
                   CHANGE: "Change %", IMPACT: "Impact", "year": "Year"}

            def movers_table(sub):
                return sub.sort_values(CHANGE, ascending=False)[disp_cols].rename(columns=ren)

            ta, tb = st.columns(2)
            with ta:
                st.caption("Biggest gains")
                st.dataframe(movers_table(movers.nlargest(8, CHANGE)), use_container_width=True, hide_index=True)
            with tb:
                st.caption("Biggest drops")
                st.dataframe(movers_table(movers.nsmallest(8, CHANGE)), use_container_width=True, hide_index=True)
        st.caption("Historical observations only. " + DISCLAIMER)

elif view == "AI Models" and _models is not None:
    st.subheader("AI Models")
    st.markdown(
        '<div class="sna-banner"><b>Course policy mapping.</b> These models implement Unit 5 '
        '<i>Learning</i> (CO4: supervised decision tree + unsupervised K-means) and Unit 3/6 '
        '<i>Knowledge Representation &amp; Expert Systems</i> (CO3: decision rules extracted from '
        "the trained tree). All figures are learned from the bundled dataset; they are modelling of "
        "past observations, not forecasts.</div>",
        unsafe_allow_html=True,
    )
    tab_sup, tab_unsup, tab_map = st.tabs(
        ["Supervised · Decision Tree", "Unsupervised · K-means", "Course Mapping"]
    )

    # ---------------------------------------------------------------- supervised
    with tab_sup:
        if not direction_available:
            st.info("No price-direction column is available, so the supervised model is disabled.")
        else:
            try:
                bundle = get_direction_bundle(info.file_hash, df, info)
            except _models.ModelUnavailableError as exc:
                bundle = None
                st.info(str(exc))
            except Exception:
                bundle = None
                st.error("The decision-tree model could not be trained with the current dataset.")
            if bundle:
                m = bundle["metrics"]
                c = st.columns(4)
                c[0].markdown(kpi_card("Test Accuracy", f"{m['accuracy']:.1%}", f"{bundle['test_rows']:,} held-out rows", "pos"), unsafe_allow_html=True)
                c[1].markdown(kpi_card("Balanced Accuracy", f"{m['balanced_accuracy']:.1%}", "avg of per-class recall", "pos"), unsafe_allow_html=True)
                c[2].markdown(kpi_card("Majority Baseline", f"{m['baseline_majority']:.1%}", "always-predict mode", ""), unsafe_allow_html=True)
                c[3].markdown(kpi_card("Features Used", str(bundle["n_features"]), f"{bundle['rows_used']:,} training rows", ""), unsafe_allow_html=True)
                if bundle["dropped_classes"]:
                    st.caption(f"Rare class(es) dropped for balance: {', '.join(bundle['dropped_classes'])}.")
                st.write("")

                a, b = st.columns(2)
                with a:
                    show = charts.feature_importance(bundle["field_importance"])
                    if show is not None:
                        st.plotly_chart(show, use_container_width=True)
                with b:
                    fig = charts.confusion_heatmap(bundle["confusion"], bundle["labels"])
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    '<h4 style="margin:1rem 0 .2rem;">Extracted decision rules '
                    + co_tag("CO3 · Knowledge Representation") + "</h4>",
                    unsafe_allow_html=True,
                )
                st.caption("The trained tree unwrapped into IF-THEN production rules (the expert-system view).")
                for r in bundle["rules"][:6]:
                    st.markdown(rule_card(r["conditions"], r["predicted"], r["confidence"], r["support"]), unsafe_allow_html=True)

                st.markdown('<h4 style="margin:1.2rem 0 .3rem;">Interactive predictor ' + co_tag("CO4 · Supervised") + "</h4>", unsafe_allow_html=True)
                st.caption("Enter a hypothetical news record to see the model's predicted direction.")
                with st.form("whatif"):
                    values: dict[str, object] = {}
                    cols = st.columns(min(4, max(1, len(bundle["feature_frame_columns"]))))
                    for i, col in enumerate(bundle["feature_frame_columns"]):
                        field = cols[i % len(cols)]
                        if col in bundle["categories"]:
                            values[col] = field.selectbox(col.replace("_", " ").title(), bundle["categories"][col], key=f"wi_{col}")
                        elif col == "year":
                            yrs = sorted(df["__date"].dt.year.dropna().astype(int).tolist()) or [2023]
                            values[col] = field.number_input("Year", min_value=int(min(yrs)), max_value=int(max(yrs)), value=int(max(yrs)), key="wi_year")
                        elif col == "log_volume":
                            vol = field.number_input("Trading volume", min_value=0, value=5_000_000, step=100_000, key="wi_vol")
                            values["log_volume"] = float(math.log1p(max(int(vol), 0)))
                        else:
                            values[col] = field.number_input(col.replace("_", " ").title(), value=0.0, key=f"wi_{col}")
                    submitted = st.form_submit_button("Predict direction", use_container_width=True)

                if submitted:
                    pred = _models.predict_direction(bundle, values)
                    pc = st.columns(2)
                    pc[0].markdown(
                        '<div class="sna-card"><div class="sna-kpi-label">Predicted direction</div>'
                        f'<div class="sna-kpi-value">{dir_badge(pred["predicted"])}</div></div>',
                        unsafe_allow_html=True,
                    )
                    with pc[1]:
                        for cls, p in pred["probs"].items():
                            st.progress(float(p), text=f"{cls}: {p:.0%}")

    # ---------------------------------------------------------------- unsupervised
    with tab_unsup:
        if not cluster_available:
            st.info("No news text column is available, so clustering is disabled.")
        else:
            try:
                quality = get_cluster_quality(info.file_hash, df, info)
                default_k = int(quality.loc[quality["silhouette"].idxmax(), "k"]) if not quality.empty else 5
            except Exception:
                quality, default_k = None, 5
            k = st.slider("Number of clusters (k)", 2, 8, default_k, key="cluster_k")
            try:
                cb = get_cluster_bundle(info.file_hash, df, info, k)
            except _models.ModelUnavailableError as exc:
                cb = None
                st.info(str(exc))
            except Exception:
                cb = None
                st.error("Clustering could not be run with the current dataset.")
            if cb:
                c = st.columns(3)
                c[0].markdown(kpi_card("Clusters (k)", str(cb["k"])), unsafe_allow_html=True)
                c[1].markdown(kpi_card("Silhouette Score", f"{cb['silhouette']:.2f}", "closer to 1 = better separated", "pos"), unsafe_allow_html=True)
                c[2].markdown(kpi_card("News Clustered", f"{len(cb['points']):,}", "TF-IDF text features", ""), unsafe_allow_html=True)
                st.write("")
                fig = charts.cluster_scatter(cb["points"])
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown('<h4 style="margin:.6rem 0 .3rem;">Cluster profiles ' + co_tag("CO4 · Unsupervised") + "</h4>", unsafe_allow_html=True)
                st.caption("Each cluster summarised by size, dominant sentiment, and average historical price change.")
                view_df = cb["stats"].rename(columns={
                    "cluster": "Cluster", "size": "Size", "dominant_sentiment": "Dominant sentiment",
                    "sentiment_mix": "Sentiment mix", "avg_change_pct": "Avg price change %", "example": "Example headline",
                })
                st.dataframe(view_df, use_container_width=True, hide_index=True)
                if quality is not None and not quality.empty:
                    a, b = st.columns(2)
                    with a:
                        qfig = charts.silhouette_curve(quality)
                        if qfig is not None:
                            st.plotly_chart(qfig, use_container_width=True)
                    with b:
                        st.markdown(panel(
                            "<b>How to read this.</b> K-means groups the news purely by wording "
                            "(no labels used). The silhouette chart helps pick a sensible k. "
                            "Clusters tend to line up with sentiment, which is a useful sanity check."
                        ), unsafe_allow_html=True)

    # ---------------------------------------------------------------- mapping
    with tab_map:
        st.markdown(panel(
            "<h4 style='margin-top:0'>Where each model maps in the course policy</h4>"
            "<table style='width:100%; border-collapse:collapse; font-size:.92rem;'>"
            "<tr style='border-bottom:1px solid rgba(148,163,184,.3);'>"
            "<th align='left'>Course outcome / unit</th><th align='left'>Implemented here</th></tr>"
            "<tr style='border-bottom:1px solid rgba(148,163,184,.2);'>"
            "<td><b>CO4</b> · Unit 5 Learning, Exp. 9 Decision Tree</td>"
            "<td>Supervised classifier predicting post-news price <i>direction</i> from sentiment, topic, company, year and volume.</td></tr>"
            "<tr style='border-bottom:1px solid rgba(148,163,184,.2);'>"
            "<td><b>CO4</b> · Unit 5 Learning, Exp. 10 K-means</td>"
            "<td>Unsupervised clustering of news wording via TF-IDF + K-means, with silhouette model selection.</td></tr>"
            "<tr>"
            "<td><b>CO3</b> · Unit 3/6 Knowledge Representation &amp; Expert Systems</td>"
            "<td>The trained tree is unwrapped into IF-THEN production rules, the classic forward-chaining expert-system form.</td></tr>"
            "</table>",
        ), unsafe_allow_html=True)
        st.caption("Russell & Norvig, Artificial Intelligence: A Modern Approach — Units 5 (Learning) and 3/6 (Knowledge representation, Expert systems). " + DISCLAIMER)

# Export of the conversation lives in the sidebar slot (rendered after chat handled).
if st.session_state.messages:
    md = ["# Stock News Analyzer - Conversation", ""]
    for m in st.session_state.messages:
        md += [f"**{m['role'].title()}:**", "", m["content"], ""]
    with export_slot:
        st.download_button("Conversation (Markdown)", "\n".join(md), file_name="stock_news_conversation.md",
                           mime="text/markdown", use_container_width=True)
