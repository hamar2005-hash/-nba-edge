from pathlib import Path
import math, subprocess, sys, zipfile
import joblib
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

# NBA Edge visual identity: dark court, warm orange accent, compact mobile-friendly cards.
st.markdown("""
<style>
:root { --edge-orange: #ff8a32; --edge-amber: #ffb15c; --edge-line: rgba(255,255,255,.09); }
.stApp { background: radial-gradient(ellipse at 12% 0%, rgba(255,138,50,.11), transparent 34%), #0c1016; }
.block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem; }
[data-testid="stHeader"] { background: rgba(12,16,22,.72); }
h1, h2, h3 { letter-spacing: -.035em; }
h2 { padding-top: .3rem; }
[data-testid="stMetric"] { background: linear-gradient(145deg, rgba(255,255,255,.065), rgba(255,255,255,.025)); border: 1px solid var(--edge-line); border-radius: 16px; padding: 16px 18px; }
[data-testid="stMetricLabel"] { color: #aeb7c4; font-size: .82rem; }
[data-testid="stMetricValue"] { color: #fff; font-weight: 750; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--edge-line); border-radius: 14px; overflow: hidden; }
.stButton > button, .stDownloadButton > button { border-radius: 11px; border: 1px solid rgba(255,138,50,.45); transition: all .15s ease; font-weight: 650; }
.stButton > button[kind="primary"] { background: linear-gradient(100deg,#f47721,#ff9a45); color: #111; border: 0; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--edge-orange); transform: translateY(-1px); }
[data-testid="stTextInput"] input, [data-testid="stSelectbox"] > div, [data-testid="stMultiSelect"] > div { border-radius: 11px; }
[data-testid="stExpander"] { border: 1px solid var(--edge-line); border-radius: 14px; background: rgba(255,255,255,.025); }
[data-testid="stSidebar"] { background: #111720; border-right: 1px solid var(--edge-line); }
hr { border-color: var(--edge-line); }
.edge-hero { border: 1px solid rgba(255,138,50,.22); border-radius: 22px; padding: 25px 28px; margin: 0 0 24px; background: linear-gradient(112deg, rgba(255,138,50,.17), rgba(255,255,255,.035) 58%, rgba(255,138,50,.06)); }
.edge-kicker { color: #ffad68; text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 800; }
.edge-title { color: #fff; font-size: clamp(2rem,5vw,3.15rem); font-weight: 850; letter-spacing: -.055em; line-height: 1.05; margin: 8px 0; }
.edge-subtitle { color: #aeb7c4; font-size: .98rem; }
.edge-pill { display:inline-block; padding:5px 10px; margin:4px 5px 0 0; border-radius:999px; background:rgba(255,138,50,.13); border:1px solid rgba(255,138,50,.25); color:#ffc18f; font-size:.76rem; font-weight:650; }
@media (max-width: 640px) { .block-container { padding-left: .8rem; padding-right: .8rem; } .edge-hero { padding: 20px 18px; border-radius: 17px; } [data-testid="stMetric"] { padding: 12px; } }
</style>
<div class="edge-hero">
  <div class="edge-kicker">Basketball intelligence</div>
  <div class="edge-title">🏀 NBA EDGE</div>
  <div class="edge-subtitle">Daily player projections, prop-line comparisons and model performance.</div>
  <span class="edge-pill">POINTS</span><span class="edge-pill">REBOUNDS</span><span class="edge-pill">ASSISTS</span><span class="edge-pill">NO ODDS OR PRICES</span>
</div>
""", unsafe_allow_html=True)

# Restore the committed trained model bundle if Streamlit's instance starts clean.
if BUNDLE.exists() and not (MODELS / "points_model.joblib").exists():
    try:
        with zipfile.ZipFile(BUNDLE) as archive:
            archive.extractall(ROOT)
    except Exception as exc:
        st.error(f"Could not extract trained model bundle: {exc}")

model_files = [MODELS / f"{market}_model.joblib" for market in ("points", "rebounds", "assists")]
features_file = MODELS / "latest_player_features.csv"
models_ready = all(path.exists() for path in model_files) and features_file.exists()

with st.sidebar:
    st.subheader("⚙️ System status")
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
        st.subheader(f"📊 Today's projections · {date}")
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
        st.subheader("🎯 Player prop line comparison")
        st.caption("Current lines are fetched from Stat Pick. Bookmaker names, odds and prices are not shown or stored.")
        if st.button("Refresh prop lines", key="refresh_prop_lines"):
            st.cache_data.clear()
            st.rerun()

        @st.cache_data(ttl=900, show_spinner=False)
        def load_prop_lines(player_names):
            # Streamlit may retain an older imported module across a hot reload.
            import importlib
            import prop_lines
            importlib.reload(prop_lines)
            try:
                return prop_lines.fetch_current_lines(player_names)
            except TypeError as exc:
                # Compatibility fallback if the deployment still has the previous no-argument module.
                if "positional argument" not in str(exc):
                    raise
                all_lines = prop_lines.fetch_current_lines()
                wanted = {str(name).lower().replace(" ", "") for name in player_names}
                all_lines["_player_match"] = all_lines["player"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
                return all_lines[all_lines["_player_match"].isin(wanted)].drop(columns="_player_match")

        try:
            with st.spinner("Fetching current NBA player prop lines…"):
                prop_lines = load_prop_lines(tuple(predictions["player"].dropna().astype(str).unique()))
            prediction_rows = predictions[["player", "team", "opponent", "minutes_baseline",
                                               "points_projection", "rebounds_projection", "assists_projection"]].copy()
            prediction_rows["player_key"] = prediction_rows["player"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
            compare = prop_lines.merge(
                prediction_rows,
                on="player_key",
                how="inner",
                suffixes=("_line", "_projection"),
            )
            # Use the model's residual spread to estimate over/under probabilities.
            market_to_col = {
                "points": "points_projection",
                "rebounds": "rebounds_projection",
                "assists": "assists_projection",
            }
            compare_rows = []
            for _, row in compare.iterrows():
                market = row["market"]
                projection_col = market_to_col.get(market)
                if not projection_col:
                    continue
                projection = pd.to_numeric(row.get(projection_col), errors="coerce")
                line = pd.to_numeric(row.get("line"), errors="coerce")
                if pd.isna(projection) or pd.isna(line):
                    continue
                pack = joblib.load(MODELS / f"{market}_model.joblib")
                sd = max(float(pack.get("residual_std", 1.0)), 0.5)
                z = (float(line) + 0.5 - float(projection)) / sd
                p_over = 1 - (0.5 * (1 + math.erf(z / math.sqrt(2))))
                p_over = min(max(p_over, 0.0001), 0.9999)
                compare_rows.append({
                    "Player": row.get("player_line", row.get("player")),
                    "Team": row.get("team_projection", row.get("team")),
                    "Opponent": row.get("opponent_projection", row.get("opponent")),
                    "Market": market.title(),
                    "Projection": round(float(projection), 1),
                    "Prop line": float(line),
                    "Projection gap": round(float(projection) - float(line), 1),
                    "Over probability": f"{p_over * 100:.1f}%",
                    "Under probability": f"{(1-p_over) * 100:.1f}%",
                    "Recent avg minutes": row.get("minutes_baseline"),
                    "Source": "Stat Pick",
                })
            if compare_rows:
                comparisons = pd.DataFrame(compare_rows)
                sort_choice = st.selectbox("Rank lines by", ["Largest projection gap", "Highest over probability", "Highest under probability"], key="line_rank")
                if sort_choice == "Largest projection gap":
                    comparisons["_sort"] = pd.to_numeric(comparisons["Projection gap"])
                elif sort_choice == "Highest over probability":
                    comparisons["_sort"] = comparisons["Over probability"].str.rstrip("%").astype(float)
                else:
                    comparisons["_sort"] = comparisons["Under probability"].str.rstrip("%").astype(float)
                comparisons = comparisons.sort_values("_sort", ascending=False).drop(columns="_sort")
                market_filter = st.multiselect("Markets", ["Points", "Rebounds", "Assists"], default=["Points", "Rebounds", "Assists"], key="line_markets")
                comparisons = comparisons[comparisons["Market"].isin(market_filter)]
                st.dataframe(comparisons, use_container_width=True, hide_index=True)
                st.download_button("Download prop line comparisons", comparisons.to_csv(index=False), "nba_edge_prop_line_comparisons.csv", "text/csv")
            else:
                st.info("No current points, rebounds or assists lines matched players in today's model projections.")
        except Exception as exc:
            st.warning("Current prop lines could not be loaded or matched. The projection dashboard is still available.")
            st.caption(str(exc))

        st.divider()
        st.subheader("🏆 Top projected performances")
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

    else:
        st.warning("The daily predictions file is empty.")

st.divider()
st.subheader("📈 Daily accuracy tracker")
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

with st.expander("About the model and its limitations"):
    st.markdown("""
- The current trained models predict **points, rebounds and assists** only.
- Player prop line numbers may be shown for comparison, but bookmaker names, odds, prices and expected-value calculations are not displayed.
- Player projections are generated for players found on ESPN's team rosters and matched to the saved historical feature snapshot.
- The model uses historical rolling statistics. It does **not** yet make reliable adjustments for confirmed starting lineups, late injury news, matchup-specific defensive factors or expected minutes changes.
- A projection is an estimate, not a guarantee. Track results over time before relying on it.
""")
