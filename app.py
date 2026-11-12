from pathlib import Path
import subprocess, sys, zipfile
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
BUNDLE = ROOT / "NBA_Trained_Models.zip"
PREDICTIONS = DATA / "daily_predictions.csv"

st.set_page_config(page_title="NBA Edge", page_icon="🏀", layout="wide")

# Restore the committed trained model bundle if Streamlit's instance starts clean.
if BUNDLE.exists() and not (MODELS / "points_model.joblib").exists():
    try:
        with zipfile.ZipFile(BUNDLE) as archive:
            archive.extractall(ROOT)
    except Exception as exc:
        st.error(f"Could not extract trained model bundle: {exc}")

st.title("🏀 NBA Edge")
st.caption("Automated NBA player projections • Points • Rebounds • Assists")

model_files = [MODELS / f"{market}_model.joblib" for market in ("points", "rebounds", "assists")]
features_file = MODELS / "latest_player_features.csv"
models_ready = all(path.exists() for path in model_files) and features_file.exists()

with st.sidebar:
    st.subheader("System status")
    if models_ready:
        st.success("Trained models available")
    else:
        st.error("Model bundle incomplete")
        st.caption("The repository needs the three trained model files and latest_player_features.csv inside NBA_Trained_Models.zip.")
    st.caption("Projections refresh automatically through the scheduled GitHub workflow. You can also request a refresh below.")
    if st.button("Refresh today's projections", type="primary", disabled=not models_ready, use_container_width=True):
        with st.spinner("Fetching today's NBA schedule and rosters, then generating projections…"):
            run = subprocess.run(
                [sys.executable, str(ROOT / "auto_daily.py"), "--models", str(MODELS), "--out", str(PREDICTIONS)],
                cwd=ROOT, capture_output=True, text=True, timeout=300
            )
        if run.returncode == 0:
            st.success(run.stdout.strip() or "Projections refreshed.")
            st.rerun()
        else:
            st.error("Refresh failed. The schedule/roster feed may be temporarily unavailable.")
            st.code((run.stderr or run.stdout)[-8000:])

if not models_ready:
    st.warning("The model files or historical feature snapshot are missing. Automated projections cannot run until the trained bundle is complete.")
elif not PREDICTIONS.exists():
    st.info("No daily projections have been generated yet. The scheduled workflow will create them, or use **Refresh today's projections** in the sidebar.")
else:
    try:
        predictions = pd.read_csv(PREDICTIONS)
    except Exception as exc:
        predictions = pd.DataFrame()
        st.error(f"Could not read the latest projections: {exc}")

    if not predictions.empty:
        date = str(predictions["game_date"].iloc[0])
        st.subheader(f"Projections for {date}")
        a, b, c, d = st.columns(4)
        a.metric("Games", predictions["game"].nunique())
        b.metric("Players", predictions["player"].nunique())
        c.metric("Player projections", len(predictions))
        d.metric("Markets", "3")

        games = ["All games"] + sorted(predictions["game"].dropna().unique().tolist())
        selected_game = st.selectbox("Filter by game", games)
        view = predictions if selected_game == "All games" else predictions[predictions["game"] == selected_game]
        search = st.text_input("Find a player")
        if search:
            view = view[view["player"].str.contains(search, case=False, na=False)]

        display = view.rename(columns={
            "game": "Matchup",
            "player": "Player",
            "team": "Team",
            "opponent": "Opponent",
            "minutes_baseline": "Recent avg minutes",
            "points_projection": "Projected points",
            "rebounds_projection": "Projected rebounds",
            "assists_projection": "Projected assists",
        })
        cols = ["Matchup", "Player", "Team", "Opponent", "Recent avg minutes", "Projected points", "Projected rebounds", "Projected assists"]
        st.dataframe(display[[c for c in cols if c in display]], use_container_width=True, hide_index=True)
        st.download_button("Download today's projections", predictions.to_csv(index=False), "nba_edge_daily_projections.csv", "text/csv")
        st.caption("Projections use the saved historical player feature snapshot. Minutes are a recent historical average, not a confirmed playing-time forecast. ESPN roster availability can be incomplete or delayed.")

        st.divider()
        st.subheader("Highest projected performances")
        market = st.segmented_control("Choose a market", ["Points", "Rebounds", "Assists"], default="Points")
        market_col = {"Points": "points_projection", "Rebounds": "rebounds_projection", "Assists": "assists_projection"}[market]
        leaders = predictions.sort_values(market_col, ascending=False).head(10)
        st.dataframe(leaders[["player", "team", "opponent", market_col]].rename(columns={market_col: f"Projected {market.lower()}"}), use_container_width=True, hide_index=True)
    else:
        st.warning("The daily predictions file is empty.")

with st.expander("About the model and its limitations"):
    st.markdown("""
- The current trained models predict **points, rebounds and assists** only.
- No bookmaker odds, betting lines, prices or expected-value calculations are used or displayed.
- Player projections are generated for players found on ESPN's team rosters and matched to the saved historical feature snapshot.
- The model uses historical rolling statistics. It does **not** yet make reliable adjustments for confirmed starting lineups, late injury news, matchup-specific defensive factors or expected minutes changes.
- A projection is an estimate, not a guarantee. Track results over time before relying on it.
""")
