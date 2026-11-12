from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="UFC Edge", page_icon="🥊", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at 12% 0%, rgba(214,72,72,.12), transparent 34%), #0c1016; }
.block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem; }
[data-testid="stMetric"] { background: linear-gradient(145deg, rgba(255,255,255,.065), rgba(255,255,255,.025)); border: 1px solid rgba(255,255,255,.09); border-radius: 16px; padding: 16px 18px; }
.ufc-hero { border: 1px solid rgba(225,91,91,.25); border-radius: 22px; padding: 25px 28px; margin-bottom: 24px; background: linear-gradient(112deg, rgba(225,91,91,.16), rgba(255,255,255,.035) 58%); }
.ufc-kicker { color: #ff9c9c; text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 800; }
.ufc-title { color: #fff; font-size: clamp(2rem,5vw,3.1rem); font-weight: 850; letter-spacing: -.055em; line-height: 1.05; margin: 8px 0; }
.ufc-subtitle { color: #aeb7c4; }
@media (max-width: 640px) { .block-container { padding-left: .8rem; padding-right: .8rem; } .ufc-hero { padding: 20px 18px; } }
</style>
<div class="ufc-hero">
  <div class="ufc-kicker">Fight analytics · Research and prediction</div>
  <div class="ufc-title">🥊 UFC EDGE</div>
  <div class="ufc-subtitle">A new model, built and tested from the ground up.</div>
</div>
""", unsafe_allow_html=True)

st.caption("Separate from NBA Edge and Horse Racing Edge. No picks will be published until the model passes chronological testing.")

DATA_BASE = "https://raw.githubusercontent.com/Sam-Armstrong/UFC-Almanac/main/data"
DATA_FILES = {
    "Fight results": f"{DATA_BASE}/FightResults.csv",
    "Fight statistics": f"{DATA_BASE}/FightStats.csv",
    "Fighter profiles": f"{DATA_BASE}/FighterData.csv",
}

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_dataset(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))

st.subheader("1. Historical data")
st.write("The first build uses public historical UFC data. The initial winner model will be based on pre-fight records and form calculated chronologically from previous results. Post-fight statistics will not be fed into historical predictions.")

try:
    with st.spinner("Checking the historical UFC dataset…"):
        results = load_dataset(DATA_FILES["Fight results"])
        stats = load_dataset(DATA_FILES["Fight statistics"])
        fighters = load_dataset(DATA_FILES["Fighter profiles"])

    a, b, c = st.columns(3)
    a.metric("Historical fight results", f"{len(results):,}")
    b.metric("Fight-stat rows", f"{len(stats):,}")
    c.metric("Fighter profiles", f"{len(fighters):,}")

    if {"Date", "Fighter 1", "Fighter 2", "Result"}.issubset(results.columns):
        parsed_dates = pd.to_datetime(results["Date"], dayfirst=True, errors="coerce")
        st.caption(
            f"Results date coverage: {parsed_dates.min().date() if parsed_dates.notna().any() else 'unknown'} "
            f"to {parsed_dates.max().date() if parsed_dates.notna().any() else 'unknown'}"
        )
        st.success("Historical results feed loaded. Data structure is ready for chronological feature engineering.")
        with st.expander("Preview historical results"):
            st.dataframe(results.head(12), use_container_width=True, hide_index=True)
    else:
        st.error("The results file is missing required columns. Model training is blocked until the data is corrected.")
        st.write("Columns found:", list(results.columns))
except Exception as exc:
    st.error("The public dataset could not be loaded. No predictions have been generated.")
    st.caption(str(exc))
    st.button("Retry data connection", on_click=load_dataset.clear)

st.divider()
st.subheader("2. Model build plan")
steps = [
    ("Historical data audit", "Check dates, duplicates, outcomes, missing values and fighter-name consistency.", True),
    ("Pre-fight feature engineering", "Build each fighter's record, recent form, streaks and opponent strength using past fights only.", False),
    ("Winner model", "Train a simple baseline and stronger challenger; compare them on later, unseen events.", False),
    ("Probability calibration", "Check Brier score, log loss and reliability before showing win probabilities.", False),
    ("Method model", "Add KO/TKO, submission and decision probabilities only after the winner model is validated.", False),
    ("Fight-card predictions and tracking", "Add upcoming cards, save predictions before fights, and settle them against official results.", False),
]
for title, detail, complete in steps:
    icon = "✅" if complete else "◯"
    st.markdown(f"**{icon} {title}**  \n{detail}")

st.divider()
st.subheader("3. Model standards")
left, right = st.columns(2)
with left:
    st.markdown("**What we will measure**")
    st.write("- Chronological holdout accuracy")
    st.write("- Brier score and log loss")
    st.write("- Calibration by probability band")
    st.write("- Performance by division and favourite level")
with right:
    st.markdown("**Leakage controls**")
    st.write("- No post-fight stats in pre-fight features")
    st.write("- No future results in fighter form")
    st.write("- Test on later events not used for training")
    st.write("- Keep a permanent prediction history")

st.warning("Research model only. A good historical score does not prove a profitable betting edge. No betting recommendations are shown at this stage.")

st.caption("Historical dataset: UFC-Almanac, sourced from UFCStats. Dataset and source attribution are retained for this project.")
