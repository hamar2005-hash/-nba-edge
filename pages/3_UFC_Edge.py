from datetime import datetime, timezone
from io import BytesIO
import os

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

ESPN_UFC_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"


@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_upcoming_ufc_cards():
    """Fetch the live UFC schedule and any bout details ESPN exposes on the scoreboard."""
    response = requests.get(
        ESPN_UFC_SCOREBOARD,
        headers={"User-Agent": "Mozilla/5.0 UFC-Edge/1.0"},
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    now = pd.Timestamp.now(tz="UTC")
    upcoming = []
    for event in payload.get("events", []):
        event_date = pd.to_datetime(event.get("date"), utc=True, errors="coerce")
        if pd.isna(event_date) or event_date < now:
            continue
        bouts = []
        competitions = event.get("competitions") or []
        for competition in competitions:
            competitors = competition.get("competitors") or []
            names = []
            for competitor in competitors:
                athlete = competitor.get("athlete") or {}
                name = (athlete.get("displayName") or athlete.get("fullName")
                        or competitor.get("displayName") or competitor.get("name"))
                if name:
                    names.append(str(name))
            if len(names) >= 2:
                bouts.append({
                    "Fighter 1": names[0],
                    "Fighter 2": names[1],
                    "Bout": f"{names[0]} vs {names[1]}",
                    "Weight class": (competition.get("type") or {}).get("text")
                        or ((competition.get("notes") or [{}])[0].get("text", "")),
                    "Status": (competition.get("status") or {}).get("type", {}).get("detail", "Scheduled"),
                })
        venue = event.get("competitions", [{}])[0].get("venue", {}) if event.get("competitions") else {}
        venue_name = venue.get("fullName") or venue.get("address", {}).get("city") or ""
        upcoming.append({
            "id": str(event.get("id", "")),
            "name": event.get("name") or event.get("shortName") or "Upcoming UFC event",
            "date": event_date,
            "venue": venue_name,
            "bouts": bouts,
            "link": next((item.get("href") for item in event.get("links", [])
                          if item.get("rel") and "summary" in item.get("rel")), ""),
        })
    return sorted(upcoming, key=lambda item: item["date"])


st.subheader("1. Next UFC card")
st.caption("The schedule refreshes automatically every 30 minutes. The next announced event is shown first, with bout listings whenever ESPN includes them in its feed.")
if st.button("Refresh upcoming UFC card", key="refresh_ufc_schedule"):
    load_upcoming_ufc_cards.clear()
try:
    upcoming_cards = load_upcoming_ufc_cards()
    if not upcoming_cards:
        st.info("No future UFC event is currently listed in the schedule feed. The section will populate automatically when an event is announced.")
    else:
        next_card = upcoming_cards[0]
        local_date = next_card["date"].tz_convert("Europe/London")
        days_to_go = max(0, (local_date.date() - pd.Timestamp.now(tz="Europe/London").date()).days)
        c1, c2, c3 = st.columns(3)
        c1.metric("Next event", next_card["name"])
        c2.metric("Date", local_date.strftime("%a %d %b %Y"))
        c3.metric("Countdown", f"{days_to_go} days")
        if next_card["venue"]:
            st.caption(f"Venue: {next_card['venue']}")
        if next_card["bouts"]:
            st.markdown("**Confirmed bouts currently listed**")
            st.dataframe(pd.DataFrame(next_card["bouts"]), width="stretch", hide_index=True)
        else:
            st.info("The event is listed, but the feed has not exposed its individual matchups yet. The card will refresh automatically as bouts are added.")
        if len(upcoming_cards) > 1:
            with st.expander("Other announced UFC events"):
                other_events = pd.DataFrame([{
                    "Event": item["name"],
                    "Date (UK)": item["date"].tz_convert("Europe/London").strftime("%d %b %Y"),
                    "Bouts listed": len(item["bouts"]),
                } for item in upcoming_cards[1:]])
                st.dataframe(other_events, width="stretch", hide_index=True)
except Exception as schedule_exc:
    st.warning("The live UFC schedule could not be reached just now. Historical model tools remain available.")
    st.caption(f"Schedule feed error: {schedule_exc}")

st.divider()

st.subheader("2. Historical data")
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
        try:
            from ufc_model import audit_results
            audit = audit_results(results)
            with st.expander("Historical data quality audit", expanded=False):
                audit_cols = st.columns(4)
                audit_cols[0].metric("Usable rows", f"{audit['usable_rows']:,}")
                audit_cols[1].metric("Invalid dates", f"{audit['invalid_dates']:,}")
                audit_cols[2].metric("Invalid results", f"{audit['invalid_results']:,}")
                audit_cols[3].metric("Duplicate bouts", f"{audit['duplicate_bout_rows']:,}")
                st.caption(
                    f"Blank fighter names: {audit['blank_names']:,} · Self-matchups: {audit['self_matchups']:,} · "
                    f"Fighters appearing more than once on the same date: {audit['same_day_repeat_fighters']:,}. "
                    "These are audit flags, not automatic corrections."
                )
        except Exception as audit_exc:
            st.warning(f"Data audit unavailable: {audit_exc}")
        with st.expander("Preview historical results"):
            st.dataframe(results.head(12), width="stretch", hide_index=True)
    else:
        st.error("The results file is missing required columns. Model training is blocked until the data is corrected.")
        st.write("Columns found:", list(results.columns))
except Exception as exc:
    st.error("The public dataset could not be loaded. No predictions have been generated.")
    st.caption(str(exc))
    st.button("Retry data connection", on_click=load_dataset.clear)

# Automatically prepare the model after the historical feed loads.
if "results" in locals() and not st.session_state.get("ufc_trained"):
    try:
        import ufc_model
        with st.spinner("Preparing the UFC model for the upcoming card…"):
            _model, _metrics, _snapshot, _frame = ufc_model.train_and_evaluate(results)
        st.session_state["ufc_model"] = _model
        st.session_state["ufc_snapshot"] = _snapshot
        st.session_state["ufc_metrics"] = _metrics
        st.session_state["ufc_trained"] = True
    except Exception as _train_exc:
        st.session_state["ufc_trained"] = False
        st.warning(f"Automatic card predictions are waiting for a trainable historical dataset: {_train_exc}")

st.divider()
st.subheader("3. Train and validate the winner model")
st.write("This first model uses only each fighter's historical results before the bout: Elo rating, win/loss record, experience, recent form, streak and time since last fight. Draws are excluded from the binary winner model. Fight statistics recorded after the bout are deliberately not used.")
if st.button("Train UFC winner model", type="primary", width="stretch"):
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
    st.caption(f"Training: {metrics['train_fights']:,} fights ({metrics['train_start']} to {metrics['train_end']}) · Calibration: {metrics['calibration_fights']:,} ({metrics['calibration_start']} to {metrics['calibration_end']}) · Untouched test: {metrics['test_fights']:,} ({metrics['test_start']} to {metrics['test_end']}). Splits use whole event dates. Lower log loss and Brier score are better.")
    st.subheader("Benchmark comparison")
    st.caption("All rows are evaluated on the same untouched test events. Elo-only converts the pre-fight Elo difference directly into a probability; the prior baseline predicts the training-set class rate.")
    benchmark = pd.DataFrame([
        {"Model": "Full calibrated model", "Accuracy %": metrics["accuracy"] * 100,
         "Balanced accuracy %": metrics["balanced_accuracy"] * 100,
         "Log loss": metrics["log_loss"], "Brier": metrics["brier"]},
        {"Model": "Elo-only benchmark", "Accuracy %": metrics["elo_accuracy"] * 100,
         "Balanced accuracy %": metrics["elo_balanced_accuracy"] * 100,
         "Log loss": metrics["elo_log_loss"], "Brier": metrics["elo_brier"]},
        {"Model": "Training prior baseline", "Accuracy %": metrics["baseline_accuracy"] * 100,
         "Balanced accuracy %": metrics["baseline_balanced_accuracy"] * 100,
         "Log loss": metrics["baseline_log_loss"], "Brier": metrics["baseline_brier"]},
    ])
    benchmark["Accuracy %"] = benchmark["Accuracy %"].round(1)
    benchmark["Balanced accuracy %"] = benchmark["Balanced accuracy %"].round(1)
    benchmark["Log loss"] = benchmark["Log loss"].round(3)
    benchmark["Brier"] = benchmark["Brier"].round(3)
    st.dataframe(benchmark, width="stretch", hide_index=True)
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
        st.dataframe(cal.rename(columns={"band": "Probability band", "fights": "Fights", "predicted": "Predicted %", "observed": "Observed %"}), width="stretch", hide_index=True)

    st.subheader("Confidence-threshold backtest")
    st.caption("Accuracy and share of unseen test fights selected at each minimum confidence. This is not a profit/ROI test because historical bookmaker odds are not included.")
    if metrics.get("thresholds"):
        threshold_table = pd.DataFrame(metrics["thresholds"])
        threshold_table["threshold"] = (threshold_table["threshold"] * 100).round(0).astype(int).astype(str) + "%"
        threshold_table["coverage"] = (threshold_table["coverage"] * 100).round(1)
        threshold_table["accuracy"] = (threshold_table["accuracy"] * 100).round(1)
        st.dataframe(threshold_table.rename(columns={"threshold": "Minimum confidence", "fights": "Selected fights", "coverage": "Coverage %", "accuracy": "Accuracy %"}), width="stretch", hide_index=True)
    else:
        st.info("No test fights met the selected confidence thresholds.")

    st.subheader("Historical holdout simulation")
    if metrics.get("temporal"):
        temporal = pd.DataFrame(metrics["temporal"])
        temporal["accuracy"] = (temporal["accuracy"] * 100).round(1)
        st.dataframe(temporal.rename(columns={"period": "Time block", "fights": "Fights", "accuracy": "Accuracy %", "start": "From", "end": "To"}), width="stretch", hide_index=True)

    st.subheader("Performance by weight class")
    if metrics.get("division_available"):
        div = pd.DataFrame(metrics["division"])
        div["accuracy"] = (div["accuracy"] * 100).round(1)
        st.dataframe(div.rename(columns={"division": "Division", "fights": "Fights", "accuracy": "Accuracy %"}), width="stretch", hide_index=True)
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
st.subheader("4. Full-card winner predictions")
st.caption("Every bout exposed by the live schedule feed is included, including close calls. Win probabilities are separate from confidence. These are model estimates, not betting recommendations.")
if upcoming_cards and st.session_state.get("ufc_trained"):
    _card = upcoming_cards[0]
    if _card.get("bouts"):
        from ufc_model import predict_matchups
        _predictions = predict_matchups(st.session_state["ufc_model"], st.session_state["ufc_snapshot"], _card["bouts"])
        if not _predictions.empty:
            st.markdown("**" + _card["name"] + " · " + _card["date"].tz_convert("Europe/London").strftime("%d %B %Y") + "**")
            st.dataframe(_predictions, width="stretch", hide_index=True)
            st.caption("Confidence is the probability assigned to the selected winner. A 51/49 pick is shown as 51% confidence. Missing-history warnings identify fighters not found in the historical database.")
        else:
            st.info("The schedule feed has not supplied usable fighter names for this card yet.")
    else:
        st.info("Predictions appear when the schedule feed publishes matchups. No fights are invented.")
elif upcoming_cards:
    st.warning("The card is available, but predictions are paused because the historical model could not be trained.")
else:
    st.info("The full-card board will populate when an upcoming event and its bout list are available.")


# Best-bets layer: compare validated model probabilities with UK bookmaker prices.
@st.cache_data(ttl=15 * 60, show_spinner=False)
def load_uk_ufc_odds(api_key):
    endpoint = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds/"
    response = requests.get(endpoint, params={
        "apiKey": api_key, "regions": "uk", "markets": "h2h",
        "oddsFormat": "decimal",
    }, timeout=25)
    response.raise_for_status()
    return response.json()


st.divider()
st.subheader("5. Best bets")
st.caption("Ranks upcoming-card moneyline opportunities by model edge, expected value and the best available UK price. A predicted winner is not automatically a bet.")
try:
    _odds_key = st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", ""))
except Exception:
    _odds_key = os.getenv("ODDS_API_KEY", "")

if not _odds_key:
    st.info("Live bookmaker odds are not connected. Add ODDS_API_KEY to Streamlit secrets to compare UK prices.")
elif not upcoming_cards or not upcoming_cards[0].get("bouts") or not st.session_state.get("ufc_trained"):
    st.info("Best bets will appear when the next card has published bouts and the winner model is trained.")
else:
    _metrics = st.session_state.get("ufc_metrics", {})
    _beats_benchmarks = (
        _metrics
        and _metrics.get("brier") is not None
        and _metrics.get("log_loss") is not None
        and _metrics.get("elo_brier") is not None
        and _metrics.get("elo_log_loss") is not None
        and _metrics.get("baseline_brier") is not None
        and _metrics.get("baseline_log_loss") is not None
        and _metrics["brier"] < min(_metrics["elo_brier"], _metrics["baseline_brier"])
        and _metrics["log_loss"] < min(_metrics["elo_log_loss"], _metrics["baseline_log_loss"])
    )
    if not _beats_benchmarks:
        st.warning("Best bets are paused: the calibrated model must beat both the Elo-only and training-prior benchmarks on the untouched test set for Brier score and log loss before its prices are treated as betting signals.")
    else:
        try:
            with st.spinner("Comparing the upcoming card with current UK bookmaker prices…"):
                _odds_events = load_uk_ufc_odds(_odds_key)
            from ufc_model import _fighter_key, predict_matchups

            _card = upcoming_cards[0]
            _card_predictions = predict_matchups(
                st.session_state["ufc_model"],
                st.session_state["ufc_snapshot"],
                _card["bouts"],
            )
            _prediction_lookup = {}
            for _, _row in _card_predictions.iterrows():
                _parts = str(_row.get("Bout", "")).split(" vs ", 1)
                if len(_parts) == 2:
                    _pair = tuple(sorted((_fighter_key(_parts[0]), _fighter_key(_parts[1]))))
                    _prediction_lookup[_pair] = _row

            _value_rows = []
            for _event in _odds_events:
                _home = str(_event.get("home_team", "")).strip()
                _away = str(_event.get("away_team", "")).strip()
                if not _home or not _away:
                    continue
                _pair_key = tuple(sorted((_fighter_key(_home), _fighter_key(_away))))
                _model_row = _prediction_lookup.get(_pair_key)
                if _model_row is None:
                    continue

                # Avoid betting recommendations where either fighter has no historical profile.
                if "No historical profile" in str(_model_row.get("Data warning", "")):
                    continue

                _home_col = next(
                    (col for col in _model_row.index
                     if str(col).endswith(" win %")
                     and _fighter_key(str(col)[:-6]) == _fighter_key(_home)),
                    None,
                )
                _away_col = next(
                    (col for col in _model_row.index
                     if str(col).endswith(" win %")
                     and _fighter_key(str(col)[:-6]) == _fighter_key(_away)),
                    None,
                )
                if not _home_col or not _away_col:
                    continue
                _model_probs = {
                    _fighter_key(_home): float(_model_row[_home_col]) / 100.0,
                    _fighter_key(_away): float(_model_row[_away_col]) / 100.0,
                }

                _best_prices, _book_names, _book_probs = {}, {}, []
                for _book in _event.get("bookmakers", []):
                    _outcomes = {}
                    for _market in _book.get("markets", []):
                        if _market.get("key") != "h2h":
                            continue
                        for _outcome in _market.get("outcomes", []):
                            try:
                                _price = float(_outcome.get("price", 0))
                            except (TypeError, ValueError):
                                continue
                            if _price > 1:
                                _outcomes[_fighter_key(_outcome.get("name", ""))] = _price

                    _p_home = _outcomes.get(_fighter_key(_home))
                    _p_away = _outcomes.get(_fighter_key(_away))
                    if not _p_home or not _p_away:
                        continue
                    _implied_home, _implied_away = 1 / _p_home, 1 / _p_away
                    _overround = _implied_home + _implied_away
                    if _overround <= 0:
                        continue
                    _book_probs.append((_implied_home / _overround, _implied_away / _overround))

                    for _fighter, _price in ((_home, _p_home), (_away, _p_away)):
                        _key = _fighter_key(_fighter)
                        if _key not in _best_prices or _price > _best_prices[_key]:
                            _best_prices[_key] = _price
                            _book_names[_key] = _book.get("title", _book.get("key", "UK bookmaker"))

                if not _book_probs:
                    continue
                _market_probs = {
                    _fighter_key(_home): sum(pair[0] for pair in _book_probs) / len(_book_probs),
                    _fighter_key(_away): sum(pair[1] for pair in _book_probs) / len(_book_probs),
                }

                for _fighter in (_home, _away):
                    _key = _fighter_key(_fighter)
                    _price = _best_prices.get(_key)
                    if not _price:
                        continue
                    _model_probability = _model_probs[_key]
                    _market_probability = _market_probs[_key]
                    _edge = _model_probability - _market_probability
                    _ev = (_model_probability * (_price - 1)) - (1 - _model_probability)

                    # Require a meaningful edge as well as positive expected value at the best price.
                    if _edge >= 0.05 and _ev > 0:
                        _value_rows.append({
                            "Fight": f"{_home} vs {_away}",
                            "Pick": _fighter,
                            "Model win %": round(_model_probability * 100, 1),
                            "Market win %": round(_market_probability * 100, 1),
                            "Edge": round(_edge * 100, 1),
                            "Best odds": round(_price, 2),
                            "Bookmaker": _book_names[_key],
                            "Expected value / £1": round(_ev, 3),
                        })

            if _value_rows:
                _best_bets = (
                    pd.DataFrame(_value_rows)
                    .sort_values(["Expected value / £1", "Edge"], ascending=False)
                    .drop_duplicates(subset=["Fight", "Pick"])
                    .head(10)
                )
                st.dataframe(_best_bets, width="stretch", hide_index=True)
                st.caption("Top 10 qualifying moneyline bets. Market probability is the average no-vig UK bookmaker consensus; best odds are selected separately. Each pick needs at least a 5 percentage-point edge and positive expected value. No stake sizing is implied.")
            else:
                st.info("No UFC moneyline bets currently meet the rules. The full-card winner predictions remain available above; no bet is better than forcing one.")
            if st.button("Refresh bookmaker odds", key="refresh_ufc_odds"):
                load_uk_ufc_odds.clear()
                st.rerun()
        except Exception as _odds_exc:
            st.warning(f"Live UK odds could not be loaded: {_odds_exc}")
            st.caption("Check the Odds API key and confirm that the account includes MMA odds.")

st.divider()
st.subheader("6. Model build plan")
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
st.subheader("7. Model standards")
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
