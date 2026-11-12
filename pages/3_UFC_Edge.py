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
st.subheader("2. Train and validate the winner model")
st.write("This first model uses only each fighter's historical results before the bout: Elo rating, win/loss record, experience, recent form, streak and time since last fight. Draws are excluded from the binary winner model. Fight statistics recorded after the bout are deliberately not used.")
if st.button("Train UFC winner model", type="primary", use_container_width=True):
    try:
        import importlib
        import ufc_model
        ufc_model = importlib.reload(ufc_model)
        with st.spinner("Building pre-fight histories and running the chronological holdout…"):
            model, metrics, snapshot, frame = ufc_model.train_and_evaluate(results)
        st.session_state["ufc_model"] = model
        st.session_state["ufc_snapshot"] = snapshot
        st.session_state["ufc_metrics"] = metrics
        st.session_state["ufc_trained"] = True
    except Exception as exc:
        st.session_state["ufc_trained"] = False
        st.session_state.pop("ufc_metrics", None)
        st.session_state.pop("ufc_model", None)
        st.session_state.pop("ufc_snapshot", None)
        st.error(f"Model training failed: {exc}")

if st.session_state.get("ufc_trained"):
    metrics = st.session_state["ufc_metrics"]
    st.success("Model trained. The metrics below are from a later-in-time holdout that was not used to fit the model.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Holdout accuracy", f"{metrics['accuracy']*100:.1f}%", f"{(metrics['accuracy']-metrics['baseline_accuracy'])*100:+.1f} pp vs baseline")
    m2.metric("Log loss", f"{metrics['log_loss']:.3f}")
    m3.metric("Brier score", f"{metrics['brier']:.3f}", f"{metrics['brier']-metrics['baseline_brier']:+.3f} vs baseline")
    m4.metric("Test fights", f"{metrics['test_fights']:,}")
    st.caption(f"Training fights: {metrics['train_fights']:,} · Calibration fights: {metrics['calibration_fights']:,} ({metrics['calibration_start']} to {metrics['calibration_end']}) · Untouched test: {metrics['test_start']} to {metrics['test_end']}. Lower log loss and Brier score are better.")
    with st.expander("Raw vs calibrated probability comparison"):
        raw_a, raw_b, raw_c = st.columns(3)
        raw_a.metric("Raw Brier", f"{metrics['raw_brier']:.3f}")
        raw_b.metric("Calibrated Brier", f"{metrics['brier']:.3f}", f"{metrics['brier']-metrics['raw_brier']:+.3f}")
        raw_c.metric("Raw log loss", f"{metrics['raw_log_loss']:.3f}")
    st.subheader("Calibration")
    st.write("Compare predicted win probabilities with observed holdout outcomes.")
    if metrics.get("calibration"):
        cal = pd.DataFrame(metrics["calibration"])
        cal["predicted"] = (cal["predicted"] * 100).round(1)
        cal["observed"] = (cal["observed"] * 100).round(1)
        st.dataframe(cal.rename(columns={"band": "Probability band", "fights": "Fights", "predicted": "Predicted %", "observed": "Observed %"}), use_container_width=True, hide_index=True)

    st.subheader("Confidence-threshold backtest")
    st.caption("Accuracy and share of unseen test fights selected at each minimum confidence. This is not a profit/ROI test because historical bookmaker odds are not included.")
    if metrics.get("thresholds"):
        threshold_table = pd.DataFrame(metrics["thresholds"])
        threshold_table["threshold"] = (threshold_table["threshold"] * 100).round(0).astype(int).astype(str) + "%"
        threshold_table["coverage"] = (threshold_table["coverage"] * 100).round(1)
        threshold_table["accuracy"] = (threshold_table["accuracy"] * 100).round(1)
        st.dataframe(threshold_table.rename(columns={"threshold": "Minimum confidence", "fights": "Selected fights", "coverage": "Coverage %", "accuracy": "Accuracy %"}), use_container_width=True, hide_index=True)
    else:
        st.info("No test fights met the selected confidence thresholds.")

    st.subheader("Historical holdout simulation")
    if metrics.get("temporal"):
        temporal = pd.DataFrame(metrics["temporal"])
        temporal["accuracy"] = (temporal["accuracy"] * 100).round(1)
        st.dataframe(temporal.rename(columns={"period": "Time block", "fights": "Fights", "accuracy": "Accuracy %", "start": "From", "end": "To"}), use_container_width=True, hide_index=True)

    st.subheader("Performance by weight class")
    if metrics.get("division_available"):
        div = pd.DataFrame(metrics["division"])
        div["accuracy"] = (div["accuracy"] * 100).round(1)
        st.dataframe(div.rename(columns={"division": "Division", "fights": "Fights", "accuracy": "Accuracy %"}), use_container_width=True, hide_index=True)
    else:
        st.info("The current results feed does not include weight class. A reliable bout-to-division mapping is needed before calculating division accuracy.")

    st.subheader("Method-of-victory model")
    st.info("Method predictions are switched off. The previous classifier achieved 27.9% accuracy against a 48.7% majority baseline and 17.7% balanced accuracy. We will rebuild it separately after adding verified pre-fight fighter attributes and testing it on a fresh chronological holdout.")

    st.subheader("Try a matchup")
    names = sorted(st.session_state["ufc_snapshot"].keys())
    f1 = st.selectbox("Fighter 1", names, key="ufc_f1")
    f2 = st.selectbox("Fighter 2", names, index=min(1, max(0, len(names)-1)), key="ufc_f2")
    if st.button("Predict winner", disabled=(f1 == f2 or not names)):
        from ufc_model import FEATURES
        snap = st.session_state["ufc_snapshot"]
        a, b = snap[f1], snap[f2]
        feature_row = pd.DataFrame([{key: a[key] - b[key] for key in [
            "elo", "wins", "losses", "experience", "win_rate",
            "recent5", "recent10", "streak", "days_since_fight"
        ]}])
        feature_row.columns = FEATURES
        prob = float(st.session_state["ufc_model"].predict_proba(feature_row)[0, 1])
        chosen = f1 if prob >= 0.5 else f2
        st.metric("Model lean", chosen, f"{max(prob, 1-prob)*100:.1f}% estimated win probability")
        st.caption("This is a historical-profile matchup estimate, not a verified upcoming-card prediction. Check current injuries, layoffs, weight changes and bout context separately.")
else:
    st.info("The model has not been trained in this app yet. Tap the button above once the historical data has loaded.")

st.divider()
st.subheader("3. Model build plan")
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
    st.write("- Performance by division and confidence threshold")
    st.write("- Separate calibration period before final holdout")
with right:
    st.markdown("**Leakage controls**")
    st.write("- No post-fight stats in pre-fight features")
    st.write("- No future results in fighter form")
    st.write("- Test on later events not used for training")
    st.write("- Keep a permanent prediction history")

st.warning("Research model only. A good historical score does not prove a profitable betting edge. No betting recommendations are shown at this stage.")

st.caption("Historical dataset: UFC-Almanac, sourced from UFCStats. Dataset and source attribution are retained for this project.")
