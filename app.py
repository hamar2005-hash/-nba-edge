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
    if st.button("Update yesterday's accuracy", use_container_width=True):
        with st.spinner("Checking completed NBA box scores…"):
            settle = subprocess.run(
                [sys.executable, str(ROOT / "settle_results.py")],
                cwd=ROOT, capture_output=True, text=True, timeout=300
            )
        if settle.returncode == 0:
            st.success(settle.stdout.strip() or "Accuracy tracker updated.")
            st.rerun()
        else:
            st.error("Could not update the accuracy tracker.")
            st.code((settle.stderr or settle.stdout)[-5000:])

# If today's file is missing or stale, generate it when the user opens the app.
if models_ready:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    today_et = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    needs_refresh = not PREDICTIONS.exists()
    if not needs_refresh:
        try:
            existing_date = str(pd.read_csv(PREDICTIONS, usecols=["game_date"])["game_date"].iloc[0])
            needs_refresh = existing_date != today_et
        except Exception:
            needs_refresh = True
    if needs_refresh:
        with st.spinner("Preparing today's NBA projections automatically…"):
            first_run = subprocess.run(
                [sys.executable, str(ROOT / "auto_daily.py"), "--models", str(MODELS), "--out", str(PREDICTIONS)],
                cwd=ROOT, capture_output=True, text=True, timeout=300
            )
        if first_run.returncode != 0:
            st.warning("Today's automatic refresh did not complete. Use the refresh button in the sidebar to try again.")
            st.caption((first_run.stderr or first_run.stdout)[-2000:])

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
        st.subheader("Top 10 projected performances")
        for market_label, market_col in [
            ("Points", "points_projection"),
            ("Rebounds", "rebounds_projection"),
            ("Assists", "assists_projection"),
        ]:
            with st.expander(f"Top 10 — {market_label}", expanded=(market_label == "Points")):
                leaders = predictions.sort_values(market_col, ascending=False).head(10).copy()
                leaders = leaders.rename(columns={
                    "player": "Player", "team": "Team", "opponent": "Opponent",
                    "minutes_baseline": "Recent avg minutes", market_col: f"Projected {market_label.lower()}",
                })
                st.dataframe(
                    leaders[["Player", "Team", "Opponent", "Recent avg minutes", f"Projected {market_label.lower()}"]],
                    use_container_width=True, hide_index=True
                )

st.divider()
st.subheader("Daily accuracy tracker")
SETTLED = DATA / "settled_results.csv"
if SETTLED.exists():
    try:
        settled = pd.read_csv(SETTLED)
    except Exception:
        settled = pd.DataFrame()
else:
    settled = pd.DataFrame()

if settled.empty:
    st.info("No completed games have been matched yet. The tracker will fill in after ESPN publishes final box scores and the next settlement run completes.")
else:
    metric_rows = []
    for market_label, projection_col, actual_col, error_col in [
        ("Points", "points_projection", "actual_points", "points_absolute_error"),
        ("Rebounds", "rebounds_projection", "actual_rebounds", "rebounds_absolute_error"),
        ("Assists", "assists_projection", "actual_assists", "assists_absolute_error"),
    ]:
        if {projection_col, actual_col, error_col}.issubset(settled.columns):
            valid = settled.dropna(subset=[projection_col, actual_col])
            if not valid.empty:
                metric_rows.append({
                    "Market": market_label,
                    "Settled players": len(valid),
                    "Mean absolute error": round(valid[error_col].mean(), 2),
                    "Within 3 stats": f"{(valid[error_col] <= 3).mean() * 100:.1f}%",
                })
    if metric_rows:
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    if "game_date" in settled:
        daily_rows = []
        for game_date, day in settled.groupby("game_date"):
            for market_label, error_col in [
                ("Points", "points_absolute_error"),
                ("Rebounds", "rebounds_absolute_error"),
                ("Assists", "assists_absolute_error"),
            ]:
                if error_col in day:
                    values = pd.to_numeric(day[error_col], errors="coerce").dropna()
                    if not values.empty:
                        daily_rows.append({"Date": game_date, "Market": market_label, "Players": len(values), "MAE": round(values.mean(), 2)})
        if daily_rows:
            st.caption("Daily mean absolute error (lower is better)")
            st.dataframe(pd.DataFrame(daily_rows).sort_values(["Date", "Market"], ascending=[False, True]), use_container_width=True, hide_index=True)
    st.download_button("Download accuracy history", settled.to_csv(index=False), "nba_edge_accuracy_history.csv", "text/csv")
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
