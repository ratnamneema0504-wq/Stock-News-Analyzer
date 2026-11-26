"""Shared styling and small HTML helpers for the Streamlit UI."""
from __future__ import annotations

import html

CSS = """
<style>
:root { --sna-accent:#6366f1; --sna-accent2:#8b5cf6; --sna-pos:#22c55e; --sna-neg:#ef4444;
        --sna-line:rgba(148,163,184,0.22); }
.block-container {padding-top: 2.2rem; max-width: 1240px; padding-bottom: 3rem;}
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}
html, body, [class*="css"] {font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif;}

/* Hero */
.sna-hero {
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #9333ea 100%);
  border-radius: 22px; padding: 1.7rem 1.9rem; margin-bottom: 1.6rem;
  color: #fff; box-shadow: 0 12px 34px rgba(79,70,229,0.28); position: relative; overflow: hidden;
}
.sna-hero::after {content:""; position:absolute; right:-40px; top:-40px; width:180px; height:180px;
  background: radial-gradient(circle, rgba(255,255,255,0.28), transparent 70%); border-radius:50%;}
.sna-hero h1 {font-size: 2.05rem; font-weight: 800; margin: 0; letter-spacing: -0.03em; color:#fff;}
.sna-hero p {margin: .35rem 0 0 0; opacity: .92; font-size: 1.03rem; font-weight: 400;}
.sna-hero .sna-chips {margin-top: .9rem; display:flex; gap:.5rem; flex-wrap:wrap;}

/* Cards / KPI */
.sna-card {
  background: rgba(148,163,184,0.07); border: 1px solid var(--sna-line);
  border-radius: 18px; padding: 1.05rem 1.2rem; height: 100%;
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.sna-card:hover {transform: translateY(-2px); border-color: rgba(99,102,241,0.5);
  box-shadow: 0 10px 26px rgba(2,6,23,0.14);}
.sna-kpi-label {font-size: .76rem; text-transform: uppercase; letter-spacing: .07em; opacity: .62; font-weight:600;}
.sna-kpi-value {font-size: 1.95rem; font-weight: 800; line-height: 1.22; letter-spacing:-0.02em;}
.sna-kpi-sub {font-size: .8rem; opacity: .62; margin-top:.15rem;}
.sna-pos {color: var(--sna-pos);} .sna-neg {color: var(--sna-neg);}

/* Panels + banners */
.sna-panel {background: rgba(148,163,184,0.06); border: 1px solid var(--sna-line);
  border-radius: 16px; padding: 1.1rem 1.25rem; margin: .4rem 0 1rem 0;}
.sna-banner {
  border-left: 4px solid var(--sna-accent); background: rgba(99,102,241,0.10);
  border-radius: 12px; padding: .8rem 1.05rem; margin: .35rem 0 1.1rem 0; font-size: .92rem;
}
.sna-empty h3 {margin-bottom: .2rem;}

/* Status pill + badges */
.sna-status {font-size: .85rem; padding: .4rem .8rem; border-radius: 999px;
  display: inline-flex; align-items:center; border: 1px solid var(--sna-line); font-weight:600;}
.sna-dot {display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px;}
.sna-chip {display:inline-block; font-size:.74rem; font-weight:600; padding:.22rem .6rem;
  border-radius:999px; background:rgba(255,255,255,0.18); border:1px solid rgba(255,255,255,0.35); color:#fff;}
.sna-badge {display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.03em;
  padding:.18rem .55rem; border-radius:8px; text-transform:uppercase;}
.sna-badge-up {background:rgba(34,197,94,0.18); color:#4ade80;}
.sna-badge-down {background:rgba(239,68,68,0.18); color:#f87171;}
.sna-badge-neutral {background:rgba(148,163,184,0.22); color:#a8b6c8;}

/* Rule cards (knowledge representation) */
.sna-rule {background: rgba(139,92,246,0.08); border:1px solid rgba(139,92,246,0.28);
  border-radius:12px; padding:.7rem .9rem; margin:.4rem 0; font-size:.9rem; line-height:1.5;}
.sna-rule code {background:rgba(139,92,246,0.14); padding:.05rem .35rem; border-radius:6px; font-size:.85em;}
.sna-rule .sna-rule-meta {opacity:.7; font-size:.8rem; margin-top:.25rem;}

/* CO tag */
.sna-co {display:inline-block; font-size:.72rem; font-weight:700; color:#6366f1;
  border:1px dashed rgba(99,102,241,0.5); border-radius:8px; padding:.1rem .5rem; margin-left:.35rem;}

/* Controls */
div[data-testid="stButton"] > button {border-radius: 12px; text-align: left; font-weight:500;
  transition: border-color .15s ease, background .15s ease;}
div[data-testid="stButton"] > button:hover {border-color: var(--sna-accent);}
div[data-testid="stDownloadButton"] > button {border-radius: 12px;}
div[data-testid="stExpander"] {border-radius: 12px; border:1px solid var(--sna-line);}
.stTabs [data-baseweb="tab-list"] {gap: .35rem; border-bottom: 1px solid var(--sna-line);}
.stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 0 1rem; height: 2.6rem; font-weight:600;}
.stTabs [aria-selected="true"] {color: var(--sna-accent);}

/* Chat bubbles */
[data-testid="stChatMessage"] {background: rgba(148,163,184,0.06); border: 1px solid rgba(148,163,184,0.10);
  border-radius: 16px; padding: .9rem 1.1rem; margin-bottom: .6rem;}
[data-testid="stChatMessage"]:hover {border-color: rgba(99,102,241,0.35);}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {width:100%;}
.stChatInput {padding-top:.4rem;}
div[data-testid="stSidebarUserContent"] {padding-top: 1rem;}
[data-testid="stSidebar"] {background:#0a0f1d; border-right:1px solid rgba(148,163,184,0.12);}

.sna-side-title {font-size:.78rem; text-transform:uppercase; letter-spacing:.08em;
  opacity:.55; font-weight:700; margin-bottom:.15rem;}
@media (max-width: 640px) {.sna-kpi-value {font-size: 1.5rem;} .block-container {padding-left: 1rem; padding-right: 1rem;}
  .sna-hero {padding: 1.25rem 1.3rem;} .sna-hero h1 {font-size: 1.6rem;}}
</style>
"""


def kpi_card(label: str, value: str, sub: str = "", tone: str = "") -> str:
    tone_cls = {"pos": " sna-pos", "neg": " sna-neg"}.get(tone, "")
    sub_html = f'<div class="sna-kpi-sub">{html.escape(sub)}</div>' if sub else ""
    return (
        '<div class="sna-card">'
        f'<div class="sna-kpi-label">{html.escape(label)}</div>'
        f'<div class="sna-kpi-value{tone_cls}">{html.escape(value)}</div>{sub_html}</div>'
    )


def status_pill(ready: bool) -> str:
    color = "#22c55e" if ready else "#94a3b8"
    text = "AI Engine: Ready" if ready else "AI Engine: Not configured"
    return f'<span class="sna-status"><span class="sna-dot" style="background:{color}"></span>{text}</span>'


def dir_badge(direction: str) -> str:
    d = (direction or "").lower()
    cls = "sna-badge-up" if d == "up" else "sna-badge-down" if d == "down" else "sna-badge-neutral"
    return f'<span class="sna-badge {cls}">{html.escape(direction)}</span>'


def co_tag(co: str) -> str:
    return f'<span class="sna-co">{html.escape(co)}</span>'


def rule_card(conditions: str, predicted: str, confidence: float, support: int) -> str:
    conds = html.escape(conditions)
    for token in ("AND", "!=", "=="):
        conds = conds.replace(f" {token} ", f" <code>{token}</code> ")
    return (
        '<div class="sna-rule">'
        f'IF {conds} <code>THEN</code> {dir_badge(predicted)}'
        f'<div class="sna-rule-meta">confidence {confidence:.0%} &nbsp;|&nbsp; {support:,} training rows</div>'
        "</div>"
    )


def panel(text: str) -> str:
    return f'<div class="sna-panel">{text}</div>'


def fmt_pct(value: float, signed: bool = True) -> str:
    return f"{value:+.2f}%" if signed else f"{value:.1f}%"
