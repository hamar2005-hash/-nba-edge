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
            st.success("Racecards loaded. Runner predictions will appear here after the historical-data model has been built and backtested.")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        st.error(f"Racing API request failed (HTTP {status}). Check your account credentials and plan access.")
    except requests.RequestException as exc:
        st.error(f"Could not reach the racing data provider: {exc}")
    except Exception as exc:
        st.error(f"Could not process the racecards: {exc}")

st.divider()
st.subheader("Prediction model — build stage")
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
st.warning("No selections or confidence percentages are displayed until the model has been trained and tested. Racecards alone are not enough to make reliable predictions.")
