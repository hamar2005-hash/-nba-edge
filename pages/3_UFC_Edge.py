from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="UFC Edge", page_icon="🥊", layout="wide")

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)

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

st.caption("Separate from NBA Edge and Horse Racing Edge. Full-card winner picks and value bets are shown together near the top.")

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
def load_upcoming_ufc_cards(week_key):
    """Fetch the live UFC schedule and keep one selected card for each Monday-Sunday week."""
    # week_key is deliberately part of the cache key so the card rolls over on Monday,
    # even if the previous week's 30-minute schedule cache has not expired yet.
    now = pd.Timestamp.now(tz="UTC")
    now_local = now.tz_convert("Europe/London")
    week_start = now_local.normalize() - pd.Timedelta(days=now_local.weekday())
    week_end = week_start + pd.Timedelta(days=6)

    # Fetch both the current Monday-Sunday window and the normal future schedule.
    # The dated request keeps a card visible after its start time until Sunday;
    # the weekly selection then rolls forward on Monday.
    headers = {"User-Agent": "Mozilla/5.0 UFC-Edge/1.0"}
    week_response = requests.get(
        ESPN_UFC_SCOREBOARD,
        params={"dates": f"{week_start:%Y%m%d}-{week_end:%Y%m%d}"},
        headers=headers,
        timeout=25,
    )
    week_response.raise_for_status()
    future_response = requests.get(
        ESPN_UFC_SCOREBOARD,
        headers=headers,
        timeout=25,
    )
    future_response.raise_for_status()

    events_by_id = {}
    for payload in (week_response.json(), future_response.json()):
        for event in payload.get("events", []):
            event_id = str(event.get("id", ""))
            if event_id:
                events_by_id[event_id] = event

    upcoming = []
    for event in events_by_id.values():
        event_date = pd.to_datetime(event.get("date"), utc=True, errors="coerce")
        if pd.isna(event_date):
            continue
        event_local_date = event_date.tz_convert("Europe/London").date()
        is_this_week = week_start.date() <= event_local_date <= week_end.date()
        # Keep this week's event visible even once its scheduled start has passed.
        # Outside the current week, only show genuinely future events.
        if not is_this_week and event_date < now:
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
    def weekly_card_order(item):
        event_date = item["date"]
        local_date = event_date.tz_convert("Europe/London").date()
        in_this_week = week_start.date() <= local_date <= week_end.date()
        if in_this_week and event_date >= now:
            return (0, 0, event_date.value)       # upcoming card in this week
        if in_this_week:
            return (0, 1, -event_date.value)      # retain this week's card through Sunday
        return (1, 0, event_date.value)           # otherwise, next future card

    return sorted(upcoming, key=weekly_card_order)



# Prepare the model before rendering the top-of-page card and betting sections.
try:
    results = load_dataset(DATA_FILES["Fight results"])
    if not st.session_state.get("ufc_trained"):
        import ufc_model
        with st.spinner("Preparing predictions for the upcoming UFC card…"):
            _early_model, _early_metrics, _early_snapshot, _early_frame = ufc_model.train_and_evaluate(results)
        st.session_state["ufc_model"] = _early_model
        st.session_state["ufc_snapshot"] = _early_snapshot
        st.session_state["ufc_metrics"] = _early_metrics
        st.session_state["ufc_trained"] = True
    if not st.session_state.get("ufc_aux_trained"):
        import ufc_model
        with st.spinner("Training method-of-victory and round-duration models…"):
            _early_stats = load_dataset(DATA_FILES["Fight statistics"])
            _aux_models, _aux_metrics, _aux_snapshot, _aux_frame = ufc_model.train_method_and_round_models(results, _early_stats)
        st.session_state["ufc_aux_models"] = _aux_models
        st.session_state["ufc_aux_metrics"] = _aux_metrics
        st.session_state["ufc_aux_snapshot"] = _aux_snapshot
        st.session_state["ufc_aux_trained"] = True
except Exception as _early_train_exc:
    st.caption(f"Some automatic UFC predictions are waiting for historical model data: {_early_train_exc}")

st.subheader("1. Next UFC card")
st.caption("The selected UFC card stays in place from Monday through Sunday, including after fight time begins. It rolls forward to the next announced card every Monday. The schedule feed refreshes every 30 minutes.")
if st.button("Refresh upcoming UFC card", key="refresh_ufc_schedule"):
    load_upcoming_ufc_cards.clear()
try:
    _uk_now = pd.Timestamp.now(tz="Europe/London")
    _week_key = (_uk_now.normalize() - pd.Timedelta(days=_uk_now.weekday())).strftime("%Y-%m-%d")
    upcoming_cards = load_upcoming_ufc_cards(_week_key)
    if not upcoming_cards:
        st.info("No UFC event is currently listed for this week or a future date. The section will populate automatically when ESPN announces the next card.")
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
            st.markdown("**Confirmed bouts and predicted winners**")
            _bout_table = pd.DataFrame(next_card["bouts"])
            if st.session_state.get("ufc_trained"):
                from ufc_model import predict_matchups
                _bout_predictions = predict_matchups(
                    st.session_state["ufc_model"],
                    st.session_state["ufc_snapshot"],
                    next_card["bouts"],
                )
                if not _bout_predictions.empty:
                    _bout_table = _bout_table.merge(
                        _bout_predictions[["Bout", "Predicted winner", "Confidence", "Data warning"]],
                        on="Bout", how="left",
                    )
            if st.session_state.get("ufc_aux_trained"):
                try:
                    from ufc_model import predict_auxiliary_matchups
                    _aux_predictions = predict_auxiliary_matchups(
                        st.session_state["ufc_aux_models"],
                        st.session_state["ufc_aux_snapshot"],
                        next_card["bouts"],
                    )
                    if not _aux_predictions.empty:
                        _aux_columns = [c for c in _aux_predictions.columns if c != "Bout"]
                        _bout_table = _bout_table.merge(
                            _aux_predictions[["Bout"] + _aux_columns],
                            on="Bout", how="left",
                        )
                except Exception as _aux_predict_exc:
                    st.caption(f"Method and round predictions are temporarily unavailable: {_aux_predict_exc}")
            st.dataframe(_bout_table, width="stretch", hide_index=True)
            st.caption("Method probabilities and Over/Under 1.5 and 2.5 rounds are produced by separate models. Higher round lines are withheld until the historical feed provides reliable scheduled-round information. Check the validation section below before relying on experimental outputs.")
            if st.session_state.get("ufc_trained") and "_bout_predictions" in locals():
                try:
                    from ufc_tracking import record_predictions
                    _saved_count = record_predictions(
                        DATA / "ufc_prediction_history.csv",
                        next_card,
                        _bout_predictions,
                        now_utc=pd.Timestamp.now(tz="UTC"),
                    )
                    if _saved_count:
                        st.caption(f"Saved {_saved_count} pre-fight prediction(s) to the live accuracy tracker.")
                except Exception as _tracking_exc:
                    st.caption(f"Live prediction tracking is temporarily unavailable: {_tracking_exc}")
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
            try:
                from ufc_tracking import record_best_prices
                _priced = record_best_prices(DATA / "ufc_prediction_history.csv", _odds_events)
                if _priced:
                    st.caption(f"Saved pre-fight UK odds for {_priced} tracked picks.")
            except Exception as _price_track_exc:
                st.caption(f"Could not save historical odds snapshots: {_price_track_exc}")
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

                st.markdown("### Suggested parlay")
                # Use at most one selection per fight to avoid contradictory legs.
                _parlay_legs = (
                    _best_bets.sort_values(["Expected value / £1", "Edge"], ascending=False)
                    .drop_duplicates(subset=["Fight"])
                    .head(3)
                    .copy()
                )
                if len(_parlay_legs) >= 2:
                    _combined_odds = float(_parlay_legs["Best odds"].prod())
                    _combined_probability = float((_parlay_legs["Model win %"] / 100).prod())
                    _parlay_ev = (_combined_probability * _combined_odds) - 1
                    st.dataframe(
                        _parlay_legs[["Fight", "Pick", "Best odds", "Bookmaker", "Model win %"]].rename(
                            columns={"Model win %": "Model win probability %"}
                        ),
                        width="stretch", hide_index=True,
                    )
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Combined decimal odds", f"{_combined_odds:.2f}")
                    p2.metric("Estimated chance", f"{_combined_probability * 100:.1f}%")
                    p3.metric("Model EV per £1", f"£{_parlay_ev:+.2f}")
                    st.caption("Automatically combines the top qualifying picks from different fights, with no more than one leg per bout. The combined chance multiplies the model probabilities and assumes the fight outcomes are independent. Parlays are high variance; this is an estimate, not a guarantee or a stake recommendation.")
                else:
                    st.info("A parlay will appear once at least two different fights have qualifying value picks.")
            else:
                st.info("No UFC moneyline bets currently meet the rules. The full-card winner predictions remain available above; no bet is better than forcing one.")
                st.markdown("### Suggested parlay")
                st.caption("A parlay needs at least two qualifying picks from different fights, so none is shown until the model finds them.")
            if st.button("Refresh bookmaker odds", key="refresh_ufc_odds"):
                load_uk_ufc_odds.clear()
                st.rerun()
        except Exception as _odds_exc:
            st.warning(f"Live UK odds could not be loaded: {_odds_exc}")
            st.caption("Check the Odds API key and confirm that the account includes MMA odds.")


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

st.divider()
st.subheader("Refresh and retrain")
st.caption("Use this after a UFC event has finished and the results feed has updated. It clears the cached historical dataset, reloads the latest results and trains a fresh challenger against the existing chronological benchmarks.")
if st.button("Refresh latest results and retrain UFC model", key="refresh_retrain_ufc", width="stretch"):
    load_dataset.clear()
    for _key in (
        "ufc_trained", "ufc_model", "ufc_snapshot", "ufc_metrics",
        "ufc_aux_trained", "ufc_aux_models", "ufc_aux_snapshot", "ufc_aux_metrics",
    ):
        st.session_state.pop(_key, None)
    st.rerun()

# Live record: predictions are frozen before event start and settled only from
# an unambiguous matching result in the historical feed.
st.divider()
st.subheader("Live prediction accuracy")
st.caption("Tracks the actual full-card winner picks saved before fight night. Results are matched by both fighter names and event date; ambiguous matches, draws and no-contests are never counted as losses.")
_UFC_HISTORY = DATA / "ufc_prediction_history.csv"
if "results" in locals() and _UFC_HISTORY.exists():
    try:
        from ufc_tracking import settle_predictions
        _newly_settled = settle_predictions(_UFC_HISTORY, results)
        if _newly_settled:
            st.caption(f"Updated {_newly_settled} completed fight result(s).")
    except Exception as _settle_exc:
        st.warning(f"Could not settle live UFC results: {_settle_exc}")

if _UFC_HISTORY.exists():
    try:
        _live_history = pd.read_csv(_UFC_HISTORY)
        _graded = _live_history[_live_history["status"].isin(["Correct", "Incorrect"])].copy()
        _correct = int((_graded["status"] == "Correct").sum())
        _total = len(_graded)
        _accuracy = (_correct / _total * 100) if _total else None
        _pending = int((_live_history["status"] == "Pending").sum())
        _void = int((_live_history["status"] == "Void").sum())
        _priced_graded = _graded[pd.to_numeric(_graded.get("predicted_odds"), errors="coerce").notna()].copy() if "predicted_odds" in _graded.columns else _graded.iloc[0:0].copy()
        _priced_graded["predicted_odds"] = pd.to_numeric(_priced_graded.get("predicted_odds"), errors="coerce")
        _priced_graded["Units"] = _priced_graded.apply(
            lambda row: (row["predicted_odds"] - 1.0) if row["status"] == "Correct" else -1.0,
            axis=1,
        ) if len(_priced_graded) else pd.Series(dtype=float)
        _units_profit = float(_priced_graded["Units"].sum()) if len(_priced_graded) else 0.0
        _l1, _l2, _l3, _l4, _l5 = st.columns(5)
        _l1.metric("Live accuracy", f"{_accuracy:.1f}%" if _accuracy is not None else "Not rated yet")
        _l2.metric("Correct picks", f"{_correct}/{_total}")
        _l3.metric("1u flat-stake P/L", f"{_units_profit:+.2f}u" if len(_priced_graded) else "No odds yet")
        _l4.metric("Pending fights", f"{_pending}")
        _l5.metric("Voids", f"{_void}")
        if len(_priced_graded):
            st.caption(f"Profit/loss uses 1 unit staked on every prediction with a saved best-available UK decimal price: winner = odds minus 1 unit; loser = minus 1 unit. Draws/no-contests are void and excluded. {len(_priced_graded)} graded picks have usable odds; picks without a saved price are excluded, not assigned invented odds.")
        if _total:
            _live_history["_sort_date"] = pd.to_datetime(_live_history["event_date"], utc=True, errors="coerce")
            _graded = _live_history[_live_history["status"].isin(["Correct", "Incorrect"])].sort_values("_sort_date")
            _recent_rows = []
            for _window in (5, 10, 20):
                _sample = _graded.tail(_window)
                if len(_sample):
                    _recent_rows.append({
                        "Window": f"Last {len(_sample)} graded fights",
                        "Record": f"{int((_sample['status'] == 'Correct').sum())}-{int((_sample['status'] == 'Incorrect').sum())}",
                        "Accuracy": f"{(_sample['status'] == 'Correct').mean() * 100:.1f}%",
                    })
            if _recent_rows:
                st.markdown("**Recent form**")
                st.dataframe(pd.DataFrame(_recent_rows), width="stretch", hide_index=True)
        _display_history = _live_history.drop(columns=["_sort_date"], errors="ignore").copy()
        _display_history = _display_history.rename(columns={
            "event_name": "Event", "event_date": "Event date", "bout": "Fight",
            "predicted_winner": "Predicted winner", "confidence": "Confidence %",
            "predicted_odds": "Odds snapshot", "odds_bookmaker": "Bookmaker",
            "actual_winner": "Actual winner", "actual_method": "Actual method",
            "status": "Result",
        })
        _display_cols = ["Event", "Event date", "Fight", "Predicted winner", "Confidence %", "Odds snapshot", "Bookmaker", "Actual winner", "Actual method", "Result"]
        st.dataframe(_display_history[[c for c in _display_cols if c in _display_history.columns]].sort_values("Event date", ascending=False), width="stretch", hide_index=True)
        st.download_button("Download UFC prediction record", _live_history.to_csv(index=False), "ufc_edge_prediction_accuracy.csv", "text/csv")
        st.caption("The live record starts when this tracker is deployed. Historical holdout results remain separate and are shown in the model validation section below.")
    except Exception as _history_exc:
        st.warning(f"Live accuracy history could not be displayed: {_history_exc}")
else:
    st.info("No live prediction record yet. The tracker will start saving full-card predictions automatically when the next card is available and predictions are generated before the event starts.")

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
    st.caption(f"Selected model: logistic regression C={metrics.get('selected_c', 0.5):g}. {metrics.get('selection_reason', 'Candidate settings were compared chronologically.') } The final test period remains untouched.")
    if metrics.get("candidate_comparison"):
        _candidate_table = pd.DataFrame(metrics["candidate_comparison"])
        _candidate_table = _candidate_table.rename(columns={
            "C": "Regularisation C",
            "calibration_brier": "Calibration Brier",
            "calibration_log_loss": "Calibration log loss",
        })
        for _metric_col in ("Calibration Brier", "Calibration log loss"):
            _candidate_table[_metric_col] = _candidate_table[_metric_col].round(4)
        st.dataframe(_candidate_table, width="stretch", hide_index=True)
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
    _aux_metrics = st.session_state.get("ufc_aux_metrics", {})
    _method_metrics = _aux_metrics.get("method", {})
    if _method_metrics.get("available"):
        _method_status = "Passed benchmark" if _method_metrics.get("passed") else "Experimental · benchmark not passed"
        st.metric("Method model status", _method_status)
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.metric("Holdout accuracy", f"{_method_metrics['accuracy']*100:.1f}%")
        _mc2.metric("Balanced accuracy", f"{_method_metrics['balanced_accuracy']*100:.1f}%")
        _mc3.metric("Log loss", f"{_method_metrics['log_loss']:.3f}", f"{_method_metrics['log_loss']-_method_metrics['baseline_log_loss']:+.3f} vs prior")
        _mc4.metric("Test fights", f"{_method_metrics['test_fights']:,}")
        st.caption(f"Chronological test period: {_method_metrics['test_start']} to {_method_metrics['test_end']}. Classes: KO/TKO, submission and decision. The model must beat the prior baseline on log loss and balanced accuracy before being treated as validated.")
        if not _method_metrics.get("passed"):
            st.warning("Method probabilities are displayed for research, but this model has not passed both baseline checks. Do not treat its most likely method as a betting edge.")
    else:
        st.info(_method_metrics.get("reason", "Method model has not trained yet."))

    st.subheader("Over/under rounds model")
    _round_metrics = _aux_metrics.get("rounds", {})
    for _line in ("1.5", "2.5"):
        _rm = _round_metrics.get(_line, {})
        if _rm.get("available"):
            _round_status = "Passed benchmark" if _rm.get("passed") else "Experimental · benchmark not passed"
            st.markdown(f"**Over/Under {_line} rounds · {_round_status}**")
            _rc1, _rc2, _rc3, _rc4 = st.columns(4)
            _rc1.metric("Holdout accuracy", f"{_rm['accuracy']*100:.1f}%")
            _rc2.metric("Balanced accuracy", f"{_rm['balanced_accuracy']*100:.1f}%")
            _rc3.metric("Brier score", f"{_rm['brier']:.3f}", f"{_rm['brier']-_rm['baseline_brier']:+.3f} vs prior")
            _rc4.metric("Test fights", f"{_rm['test_fights']:,}")
            st.caption(f"Log loss {_rm['log_loss']:.3f} vs prior {_rm['baseline_log_loss']:.3f} · test period {_rm['test_start']} to {_rm['test_end']}. Round duration is trained from the recorded elapsed fight time.")
            if not _rm.get("passed"):
                st.warning(f"Over/under {_line} probabilities are experimental because they did not beat the prior baseline on both log loss and balanced accuracy.")
        else:
            st.info(f"Over/under {_line}: {_rm.get('reason', 'Model has not trained yet.')}")

    if not st.session_state.get("ufc_aux_trained"):
        if st.button("Train method and round models", key="train_ufc_aux_models"):
            try:
                _aux_stats = load_dataset(DATA_FILES["Fight statistics"])
                from ufc_model import train_method_and_round_models
                with st.spinner("Training independent method and round-duration models…"):
                    _models, _metrics, _snapshot, _frame = train_method_and_round_models(results, _aux_stats)
                st.session_state["ufc_aux_models"] = _models
                st.session_state["ufc_aux_metrics"] = _metrics
                st.session_state["ufc_aux_snapshot"] = _snapshot
                st.session_state["ufc_aux_trained"] = True
                st.rerun()
            except Exception as _aux_train_exc:
                st.error(f"Method and round model training failed: {_aux_train_exc}")

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
