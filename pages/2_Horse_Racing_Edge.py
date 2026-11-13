from datetime import date
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Horse Racing Edge", page_icon="🐎", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at 12% 0%, rgba(64,190,145,.10), transparent 34%), #0c1016; }
.block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem; }
[data-testid="stMetric"] { background: linear-gradient(145deg, rgba(255,255,255,.065), rgba(255,255,255,.025)); border: 1px solid rgba(255,255,255,.09); border-radius: 16px; padding: 16px 18px; }
.edge-hero { border: 1px solid rgba(64,190,145,.24); border-radius: 22px; padding: 25px 28px; margin-bottom: 24px; background: linear-gradient(112deg, rgba(64,190,145,.15), rgba(255,255,255,.035) 58%); }
.edge-kicker { color: #7fe0bd; text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 800; }
.edge-title { color: #fff; font-size: clamp(2rem,5vw,3.1rem); font-weight: 850; letter-spacing: -.055em; line-height: 1.05; margin: 8px 0; }
.edge-subtitle { color: #aeb7c4; }
@media (max-width: 640px) { .block-container { padding-left: .8rem; padding-right: .8rem; } .edge-hero { padding: 20px 18px; } }
</style>
<div class="edge-hero">
  <div class="edge-kicker">Racing intelligence · Great Britain & Ireland</div>
  <div class="edge-title">🐎 HORSE RACING EDGE</div>
  <div class="edge-subtitle">Racecards first. Evidence-led runner ratings and tested predictions next.</div>
</div>
""", unsafe_allow_html=True)

st.caption("This is a separate racing area. It does not change the NBA model or its predictions.")

with st.sidebar:
    st.subheader("Racecard settings")
    day_choice = st.selectbox("Racing day", ["today", "tomorrow"])
    st.divider()
    st.markdown("**Data connection**")
    st.caption("Connect The Racing API to load live UK and Irish racecards. Your credentials stay in Streamlit secrets and are not shown on the page.")
    st.markdown("[Create a free API account](https://www.theracingapi.com/)")

def get_credentials():
    try:
        username = st.secrets.get("RACING_API_USERNAME", "")
        password = st.secrets.get("RACING_API_PASSWORD", "")
        return username, password
    except Exception:
        return "", ""

def unpack_racecards(payload):
    data = (payload.get("racecards") or payload.get("data") or []) if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        data = [data]
    rows = []
    def add_race(race, meeting=None):
        if not isinstance(race, dict):
            return
        meeting = meeting or {}
        runners = race.get("runners") or race.get("horses") or []
        if isinstance(runners, dict):
            runners = list(runners.values())
        if not runners:
            rows.append({
                "Course": race.get("course") or meeting.get("course") or meeting.get("name", ""),
                "Time": race.get("time") or race.get("off_time", ""),
                "Race": race.get("race_name") or race.get("name") or race.get("title", ""),
                "Runner": "",
                "Jockey": "",
                "Trainer": "",
                "Age": "",
                "Weight": "",
                "Draw": "",
                "Race ID": race.get("race_id") or race.get("id", ""),
                "Horse ID": runner.get("horse_id") or runner.get("horseId") or runner.get("id", ""),
                "Form": runner.get("form") or runner.get("form_string") or runner.get("recent_form", ""),
                "Going": race.get("going") or meeting.get("going", ""),
                "Distance": race.get("distance") or race.get("dist", ""),
                "Class": race.get("class") or race.get("race_class", ""),
                "Odds": runner.get("odds_decimal") or runner.get("decimal_odds") or runner.get("odds") or runner.get("price", ""),
                "Price Movement": runner.get("odds_movement") or runner.get("price_movement") or runner.get("movement", ""),
                "Trainer win %": runner.get("trainer_win_pct") or runner.get("trainer_win_percentage") or "",
                "Jockey win %": runner.get("jockey_win_pct") or runner.get("jockey_win_percentage") or "",
            })
        for runner in runners:
            if not isinstance(runner, dict):
                continue
            rows.append({
                "Course": race.get("course") or meeting.get("course") or meeting.get("name", ""),
                "Time": race.get("time") or race.get("off_time", ""),
                "Race": race.get("race_name") or race.get("name") or race.get("title", ""),
                "Runner": runner.get("horse") or runner.get("horse_name") or runner.get("name", ""),
                "Jockey": runner.get("jockey") or runner.get("jockey_name", ""),
                "Trainer": runner.get("trainer") or runner.get("trainer_name", ""),
                "Age": runner.get("age", ""),
                "Weight": runner.get("weight") or runner.get("lbs", ""),
                "Draw": runner.get("draw", ""),
                "Race ID": race.get("race_id") or race.get("id", ""),
            })
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        races = item.get("races")
        if isinstance(races, list):
            for race in races:
                add_race(race, item)
        elif item.get("runners") or item.get("horses"):
            add_race(item, item)
        else:
            for nested in item.values():
                if isinstance(nested, list):
                    for race in nested:
                        if isinstance(race, dict) and (race.get("runners") or race.get("horses")):
                            add_race(race, item)
    return pd.DataFrame(rows)

username, password = get_credentials()

if not username or not password:
    st.info("The racing page has been added, but live racecards are not connected yet.")
    st.markdown("### Connect the free racecard feed")
    st.markdown("""
1. Create an account with [The Racing API](https://www.theracingapi.com/).
2. Find your API username and password in your account dashboard.
3. In your Streamlit app, open **Settings → Secrets** and add:
""")
    st.code('RACING_API_USERNAME = "your_username"\nRACING_API_PASSWORD = "your_password"', language="toml")
    st.caption("The free racecard endpoint provides basic cards for today and tomorrow. We will use historical results to build and validate the prediction model before showing any picks.")
else:
    @st.cache_data(ttl=180, show_spinner=False)
    def load_racecards(day, api_user, api_password):
        response = requests.get(
            "https://api.theracingapi.com/v1/racecards/free",
            params={"day": day, "region_codes": ["gb", "ire"], "limit": 500},
            auth=(api_user, api_password),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    if st.button("Refresh racecards", type="primary"):
        load_racecards.clear()
    try:
        with st.spinner("Loading UK and Irish racecards…"):
            payload = load_racecards(day_choice, username, password)
        cards = unpack_racecards(payload)
        if cards.empty:
            st.warning("The API connected, but its response did not contain race entries in the expected format. No predictions have been generated.")
            st.json(payload)
        else:
            courses = sorted(cards["Course"].dropna().astype(str).unique())
            selected_course = st.selectbox("Filter by course", ["All courses"] + courses)
            view = cards if selected_course == "All courses" else cards[cards["Course"] == selected_course]
            a, b, c = st.columns(3)
            a.metric("Courses", view["Course"].nunique())
            b.metric("Races", view[["Course", "Time", "Race ID"]].drop_duplicates().shape[0])
            c.metric("Declared runners", view["Runner"].replace("", pd.NA).notna().sum())
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button("Download racecards", view.to_csv(index=False), "horse_racing_edge_racecards.csv", "text/csv")
            st.success("Racecards loaded.")

            st.divider()
            st.subheader("Live form-based predictions")
            st.caption("The first live version uses each declared horse's previous UK/Irish runs, recent finishing positions, course and going record. The estimates are race-relative and provisional, not yet calibrated. Odds and price movement are shown only when supplied by your data plan.")
            from racing_predictions import fetch_form_predictions

            @st.cache_data(ttl=1800, show_spinner=False)
            def cached_form_predictions(card_frame, api_user, api_password):
                return fetch_form_predictions(card_frame, api_user, api_password)

            if st.button("Generate horse predictions", type="primary", key="generate_horse_predictions"):
                try:
                    with st.spinner("Checking recent form for each declared runner…"):
                        predicted = cached_form_predictions(view, username, password)
                    st.session_state["horse_edge_predictions"] = predicted
                    from pathlib import Path
                    output_dir = Path(__file__).resolve().parents[1] / "data"
                    output_dir.mkdir(exist_ok=True)
                    predicted.to_csv(output_dir / "horse_predictions.csv", index=False)
                except PermissionError as exc:
                    st.error(str(exc))
                except requests.RequestException as exc:
                    st.error(f"Could not retrieve horse form: {exc}")
                except Exception as exc:
                    st.error(f"Could not generate predictions: {exc}")

            predicted = st.session_state.get("horse_edge_predictions")
            if isinstance(predicted, pd.DataFrame) and not predicted.empty:
                race_keys = ["Course", "Time", "Race ID"]
                if all(key in predicted.columns for key in race_keys):
                    usable = predicted.groupby(race_keys, dropna=False)["Form score"].transform("nunique") > 1
                    predicted = predicted[usable].copy()
                if predicted.empty:
                    st.warning("The feed did not provide enough distinct recent-form information to rank these runners. No picks have been forced.")
                else:
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Runners scored", len(predicted))
                    p2.metric("Races scored", predicted[["Course", "Time", "Race ID"]].drop_duplicates().shape[0])
                    p3.metric("Runners with form data", int((predicted["Form score"] > 0).sum()))
                    top_picks = predicted.sort_values("Win estimate", ascending=False).groupby(
                        ["Course", "Time", "Race ID"], dropna=False
                    ).head(1)
                    st.markdown("**Top-ranked runner in each race**")
                    pick_columns = [c for c in ["Course", "Time", "Runner", "Recent form", "Runs", "Win rate", "Model estimate %", "Odds", "Price Movement", "Model vs market (pp)"] if c in top_picks.columns]
                    picks_view = top_picks[pick_columns].copy()
                    if "Win rate" in picks_view:
                        picks_view["Win rate"] = (picks_view["Win rate"] * 100).round(1).astype(str) + "%"
                    if "Model estimate %" in picks_view:
                        picks_view["Model estimate %"] = picks_view["Model estimate %"].round(1).astype(str) + "%"
                    if "Model vs market (pp)" in picks_view:
                        picks_view["Model vs market (pp)"] = picks_view["Model vs market (pp)"].round(1)
                    st.dataframe(picks_view, use_container_width=True, hide_index=True)
                    st.markdown("**Full runner rankings**")
                    st.dataframe(predicted, use_container_width=True, hide_index=True)
                    st.download_button("Download horse predictions", predicted.to_csv(index=False), "horse_racing_edge_predictions.csv", "text/csv")
                st.warning("These are experimental form scores, not proven betting edges. We need to record predictions before each race and compare them with results before promoting this as a validated model.")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        st.error(f"Racing API request failed (HTTP {status}). Check your account credentials and plan access.")
    except requests.RequestException as exc:
        st.error(f"Could not reach the racing data provider: {exc}")
    except Exception as exc:
        st.error(f"Could not process the racecards: {exc}")

st.divider()
st.subheader("Historical data and model training")
st.write("Pick Lab can fetch the historical database directly from Kaggle. You do not need to download the 1.2 GB ZIP to your iPhone or upload it to ChatGPT.")

st.markdown("**One-time Kaggle connection**")
st.caption("Kaggle requires an API token for dataset downloads. In Streamlit → Settings → Secrets, add your Kaggle username and API key:")
st.code('KAGGLE_USERNAME = "your_kaggle_username"\nKAGGLE_KEY = "your_kaggle_api_key"', language="toml")
st.caption("Create a token in your Kaggle account under Settings → API → Create New Token. Keep the key private.")

import os
import sqlite3
from pathlib import Path

@st.cache_resource(show_spinner=False)
def download_racing_dataset(kaggle_user, kaggle_key):
    os.environ["KAGGLE_USERNAME"] = kaggle_user
    os.environ["KAGGLE_KEY"] = kaggle_key
    import kagglehub
    folder = Path(kagglehub.dataset_download("deltaromeo/horse-racing-results-ukireland-2015-2025"))
    dbs = list(folder.rglob("*.db")) + list(folder.rglob("*.sqlite")) + list(folder.rglob("*.sqlite3"))
    if not dbs:
        raise FileNotFoundError("Kaggle download completed, but no SQLite database was found in the dataset.")
    return str(max(dbs, key=lambda f: f.stat().st_size))

kaggle_user = st.secrets.get("KAGGLE_USERNAME", "")
kaggle_key = st.secrets.get("KAGGLE_KEY", "")
if st.button("Connect to Kaggle and fetch historical results", type="primary", key="fetch_kaggle_racing"):
    if not kaggle_user or not kaggle_key:
        st.error("Add KAGGLE_USERNAME and KAGGLE_KEY to Streamlit Secrets first.")
    else:
        try:
            with st.spinner("Downloading the historical database directly to the Streamlit app. This is a large download and may take some time…"):
                db_path = download_racing_dataset(kaggle_user, kaggle_key)
            st.session_state["racing_db_path"] = db_path
            st.success(f"Historical database downloaded: {Path(db_path).name} ({Path(db_path).stat().st_size / (1024**3):.2f} GB)")
        except Exception as exc:
            st.error(f"Kaggle download failed: {exc}")

db_path = st.session_state.get("racing_db_path")
if db_path and Path(db_path).exists():
    try:
        with sqlite3.connect(db_path) as conn:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name", conn
            )["name"].tolist()
            st.write(f"Database ready. Found {len(tables)} tables.")
            table_info = []
            for table in tables:
                safe_table = table.replace('"', '""')
                cols = pd.read_sql_query(f'PRAGMA table_info("{safe_table}")', conn)
                count = pd.read_sql_query(f'SELECT COUNT(*) AS n FROM "{safe_table}"', conn).iloc[0]["n"]
                table_info.append({"Table": table, "Rows": int(count), "Columns": ", ".join(cols["name"].astype(str))})
        st.dataframe(pd.DataFrame(table_info), use_container_width=True, hide_index=True)
        selected_table = st.selectbox("Choose the table containing one row per horse per race", tables)
        if st.button("Load selected table and train model", type="primary", key="train_from_sqlite"):
            try:
                with sqlite3.connect(db_path) as conn:
                    safe_table = selected_table.replace('"', '""')
                    history_df = pd.read_sql_query(f'SELECT * FROM "{safe_table}"', conn)
                st.caption(f"Loaded {len(history_df):,} rows from {selected_table}.")
                from racing_model import train_and_backtest
                with st.spinner("Building historical form features and running the chronological holdout…"):
                    racing_model, racing_columns, racing_metrics, racing_test = train_and_backtest(history_df)
                st.session_state["racing_model"] = racing_model
                st.session_state["racing_model_columns"] = racing_columns
                st.session_state["racing_model_metrics"] = racing_metrics
                st.session_state["racing_model_test"] = racing_test
                st.success("Historical model trained and backtested.")
            except Exception as exc:
                st.error(f"Could not train from this table. We may need to map its database columns first: {exc}")
    except Exception as exc:
        st.error(f"Could not inspect the SQLite database: {exc}")
else:
    st.info("Connect to Kaggle above to fetch the historical database directly. The database stays on the app host; it is not uploaded through this chat.")

if "racing_model_metrics" in st.session_state:
    metrics = st.session_state["racing_model_metrics"]
    st.markdown("**Chronological holdout results**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test races", f'{metrics["test_races"]:,}')
    m2.metric("Top-ranked win rate", f'{metrics["top_pick_win_rate"]:.1%}')
    m3.metric("Brier score", f'{metrics["brier"]:.4f}')
    m4.metric("Log loss", f'{metrics["log_loss"]:.4f}')
    st.caption(f'Trained on {metrics["train_races"]:,} earlier races and tested on {metrics["test_races"]:,} later races from {metrics["cutoff"]}.')
    st.dataframe(st.session_state["racing_model_test"].sort_values(["race_date", "race_key", "rank"]), use_container_width=True, hide_index=True)
    st.warning("Backtest results are research metrics, not proof of profitability. Live selections remain disabled until current runners can be safely matched to historical form.")


st.divider()
st.subheader("Model status")
left, right = st.columns(2)
with left:
    st.markdown("**Planned inputs**")
    st.write("- Recent finishing positions and speed/performance ratings")
    st.write("- Course, distance and going suitability")
    st.write("- Race type, class, field size and weight")
with right:
    st.markdown("**Validation before publishing picks**")
    st.write("- Historical, time-ordered backtesting")
    st.write("- Win and place probability calibration")
    st.write("- Blind results tracking, including non-runners")
st.caption("The historical-data trainer and chronological holdout report are available above. Live runner predictions remain disabled until the current racecards can be joined to historical runner form.")
