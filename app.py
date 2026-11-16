from pathlib import Path
import math, os, subprocess, sys, zipfile
import requests
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
  <span class="edge-pill">POINTS</span><span class="edge-pill">REBOUNDS</span><span class="edge-pill">ASSISTS</span><span class="edge-pill">OPTIONAL SKY BET ODDS TRACKING</span>
</div>
""", unsafe_allow_html=True)

# Restore the committed trained model bundle if Streamlit's instance starts clean
# or if a previous extraction left only part of the required files behind.
model_files = [MODELS / f"{market}_model.joblib" for market in ("points", "rebounds", "assists")]
required_bundle_files = model_files + [MODELS / "latest_player_features.csv"]
if BUNDLE.exists() and not all(path.exists() for path in required_bundle_files):
    try:
        with zipfile.ZipFile(BUNDLE) as archive:
            archive.extractall(ROOT)
    except Exception as exc:
        st.error(f"Could not extract trained model bundle: {exc}")

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
    st.divider()
    st.markdown("**Model training**")
    st.caption("Retrain the points model using the latest historical game logs. The workflow validates it against the current model and only promotes it if holdout MAE improves.")
    if st.button("Retrain points model", width="stretch"):
        github_token = st.secrets.get("NBA_EDGE_GITHUB_TOKEN", os.getenv("NBA_EDGE_GITHUB_TOKEN", ""))
        if not github_token:
            st.warning("Training from this button needs the NBA_EDGE_GITHUB_TOKEN secret configured in the app deployment. See the setup note in the repository README.")
        else:
            dispatch_url = "https://api.github.com/repos/hamar2005-hash/-nba-edge/actions/workflows/weekly-retrain.yml/dispatches"
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {github_token}",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            try:
                response = requests.post(
                    dispatch_url,
                    headers=headers,
                    json={"ref": "main", "inputs": {"market": "points"}},
                    timeout=20,
                )
                if response.status_code == 204:
                    st.success("Points-model retraining has been queued. It will collect fresh game logs, validate the candidate and keep the current model if the candidate is worse.")
                    st.link_button("View training progress", "https://github.com/hamar2005-hash/-nba-edge/actions/workflows/weekly-retrain.yml")
                else:
                    st.error(f"GitHub could not start training (HTTP {response.status_code}). Check that the app token has Actions: write permission.")
                    st.caption(response.text[:1000])
            except requests.RequestException as exc:
                st.error(f"Could not contact GitHub to start training: {exc}")
    if st.button("Refresh today's projections", type="primary", disabled=not models_ready, width="stretch"):
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
    if st.button("Update yesterday's accuracy", width="stretch"):
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

# Do not run the full schedule/roster scraper synchronously during Streamlit startup.
# ESPN roster requests can exceed the app startup budget and crash the entire page.
# Daily projections are generated by the scheduled GitHub workflow or on explicit
# user request through the sidebar refresh button.
if models_ready and not PREDICTIONS.exists():
    st.info("Today's projections have not been generated yet. The scheduled workflow will create them, or use **Refresh today's projections** in the sidebar.");

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
            "expected_minutes": "Forecast minutes",
            "points_confidence": "Points confidence",
            "points_projection": "Projected points",
            "rebounds_projection": "Projected rebounds",
            "assists_projection": "Projected assists",
            "threes_projection": "Projected threes",
            "steals_projection": "Projected steals",
            "blocks_projection": "Projected blocks",
            "turnovers_projection": "Projected turnovers",
            "player_status": "Availability",
        })
        cols = ["Matchup", "Player", "Team", "Opponent", "Availability", "Recent avg minutes", "Forecast minutes", "Points confidence", "Projected points", "Projected rebounds", "Projected assists", "Projected threes", "Projected steals", "Projected blocks", "Projected turnovers"]
        st.dataframe(display[[c for c in cols if c in display]], width="stretch", hide_index=True)
        st.download_button("Download today's projections", predictions.to_csv(index=False), "nba_edge_daily_projections.csv", "text/csv")
        trend_cols = [c for c in ["pts_last5", "pts_last10", "pts_trend", "reb_last5", "reb_last10", "reb_trend", "ast_last5", "ast_last10", "ast_trend"] if c in predictions.columns]
        if trend_cols:
            with st.expander("📈 Recent player form (last 5 vs last 10 games)"):
                trend_view = predictions[["player", "team", "game"] + trend_cols].copy()
                trend_view = trend_view.rename(columns={
                    "player": "Player", "team": "Team", "game": "Matchup",
                    "pts_last5": "PTS L5", "pts_last10": "PTS L10", "pts_trend": "PTS trend",
                    "reb_last5": "REB L5", "reb_last10": "REB L10", "reb_trend": "REB trend",
                    "ast_last5": "AST L5", "ast_last10": "AST L10", "ast_trend": "AST trend",
                })
                st.dataframe(trend_view, width="stretch", hide_index=True)
        st.caption("Points V2 forecasts playing time separately from scoring rate. Confidence tiers are heuristic indicators based on historical holdout error and the gap between forecast and recent minutes, not calibrated probabilities. Availability is based on the roster feed and may not reflect late injury news.")

        st.divider()
        st.subheader("🏀 First basket scorer")
        st.caption("Ranks likely first made field-goal scorers using historical starter and opening-basket data. Percentages are model estimates, not bookmaker odds or calibrated guarantees.")
        FIRST_BASKET = DATA / "first_basket_predictions.csv"
        FIRST_BASKET_HISTORY = DATA / "first_basket_player_history.csv"
        if FIRST_BASKET_HISTORY.exists():
            try:
                run_fb = subprocess.run(
                    [sys.executable, str(ROOT / "first_basket.py"), "--mode", "predict",
                     "--predictions", str(PREDICTIONS), "--history", str(FIRST_BASKET_HISTORY),
                     "--out", str(FIRST_BASKET)],
                    cwd=ROOT, capture_output=True, text=True, timeout=120
                )
                if run_fb.returncode != 0:
                    st.info("First-basket rankings are not ready yet: " + (run_fb.stderr or run_fb.stdout)[-800:])
            except Exception as exc:
                st.info(f"First-basket ranking could not be refreshed: {exc}")
        if FIRST_BASKET.exists():
            try:
                first_basket = pd.read_csv(FIRST_BASKET)
                if not first_basket.empty:
                    fb_view = first_basket.copy()
                    if selected_game != "All games":
                        fb_view = fb_view[fb_view["game"] == selected_game]
                    fb_view = fb_view.rename(columns={
                        "game": "Matchup", "player": "Player", "team": "Team",
                        "opponent": "Opponent", "rank": "Rank",
                        "estimated_chance_pct": "Estimated chance (%)",
                        "first_baskets": "Career sample first baskets",
                        "starts": "Historical starts in sample",
                    })
                    st.dataframe(
                        fb_view[["Matchup", "Rank", "Player", "Team", "Opponent",
                                 "Estimated chance (%)", "Career sample first baskets",
                                 "Historical starts in sample"]],
                        width="stretch", hide_index=True
                    )
                    st.download_button(
                        "Download first-basket picks",
                        fb_view.to_csv(index=False),
                        "nba_edge_first_basket_picks.csv",
                        "text/csv",
                    )
                else:
                    st.info("No first-basket candidates have enough historical data yet.")
            except Exception as exc:
                st.info(f"First-basket predictions are being prepared: {exc}")
        else:
            st.info("First-basket history is being collected. Rankings will appear after the first successful historical-data run.")

        st.divider()
        st.subheader("🎯 Player prop line comparison")
        st.caption("Current prop lines are fetched from Stat Pick. Bookmaker prices are not supplied by this feed; enter actual Sky Bet decimal odds in the performance tracker when available.")
        @st.cache_data(ttl=300, show_spinner=False)
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

        refresh_props = st.button("Refresh prop lines", key="refresh_prop_lines")
        if refresh_props:
            # The button already triggers a rerun; do not call st.rerun() again.
            load_prop_lines.clear()

        try:
            with st.spinner("Fetching today's posted prop lines…"):
                prop_lines = load_prop_lines(
                    tuple(predictions["player"].dropna().astype(str).unique())
                )
            prediction_columns = ["player", "team", "opponent", "game_date", "game_time_utc", "minutes_baseline",
                                   "points_projection", "rebounds_projection", "assists_projection",
                                   "pts_last5", "pts_last10", "pts_trend",
                                   "reb_last5", "reb_last10", "reb_trend",
                                   "ast_last5", "ast_last10", "ast_trend"]
            prediction_rows = predictions[[c for c in prediction_columns if c in predictions.columns]].copy()
            prediction_rows["player_key"] = prediction_rows["player"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
            # Team is a second identity check. A player-name-only join can attach
            # a stale or misidentified feed record to the wrong roster entry.
            prop_lines["team_key"] = prop_lines["team"].fillna("").astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
            prediction_rows["team_key"] = prediction_rows["team"].fillna("").astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
            compare = prop_lines.merge(
                prediction_rows,
                on="player_key",
                how="inner",
                suffixes=("_line", "_projection"),
            )
            compare = compare[
                (compare["team_key_line"] == "") |
                (compare["team_key_projection"] == "") |
                (compare["team_key_line"] == compare["team_key_projection"])
            ].copy()
            compare = compare.drop_duplicates(["player_key", "market", "line"])
            if compare.empty:
                st.warning("No prop lines passed the player-and-team identity check. The app will not force a match.")
            # Use the model's residual spread to estimate over/under probabilities.
            market_to_col = {
                "points": ("points_projection",),
                "rebounds": ("rebounds_projection",),
                "assists": ("assists_projection",),
                "points + rebounds": ("points_projection", "rebounds_projection"),
                "points + assists": ("points_projection", "assists_projection"),
                "rebounds + assists": ("rebounds_projection", "assists_projection"),
                "points + rebounds + assists": ("points_projection", "rebounds_projection", "assists_projection"),
            }
            compare_rows = []
            # Load each trained market model once per render, not once per prop row.
            # A slate can contain hundreds of lines, so disk-loading inside the loop
            # needlessly stalls Streamlit and leaves the page in its dark loading state.
            residual_spreads = {}
            for market in compare["market"].dropna().unique():
                model_path = MODELS / f"{market}_model.joblib"
                if model_path.exists():
                    pack = joblib.load(model_path)
                    residual_spreads[market] = max(float(pack.get("residual_std", 1.0)), 0.5)

            for _, row in compare.iterrows():
                market = row["market"]
                projection_cols = market_to_col.get(market)
                if not projection_cols:
                    continue
                component_values = [pd.to_numeric(row.get(col), errors="coerce") for col in projection_cols]
                if any(pd.isna(value) for value in component_values):
                    continue
                projection = sum(float(value) for value in component_values)
                line = pd.to_numeric(row.get("line"), errors="coerce")
                if pd.isna(projection) or pd.isna(line):
                    continue

                # Sanity-check source lines against both broad NBA market bounds
                # and the player's own model projection. This prevents a bad
                # feed mapping (e.g. 11.5 assists for a 1.3-assist projection)
                # from being presented as a high-confidence pick.
                hard_max = {"points": 60.0, "rebounds": 30.0, "assists": 25.0,
                            "points + rebounds": 85.0, "points + assists": 85.0,
                            "rebounds + assists": 55.0, "points + rebounds + assists": 110.0}
                # Reject lines that are too far from the projection to be
                # credible same-market comparisons. The previous relative
                # limits admitted extreme mismatches (e.g. 20.5 REB vs 11.0).
                max_gap = {
                    "points": 10.0, "rebounds": 4.0, "assists": 3.5,
                    "points + rebounds": 12.0, "points + assists": 12.0,
                    "rebounds + assists": 7.5, "points + rebounds + assists": 15.0,
                }
                if float(line) < 0.5 or float(line) > hard_max[market]:
                    continue
                if abs(float(line) - float(projection)) > max_gap[market]:
                    continue

                # Guard against degenerate residual estimates. A tiny or
                # missing spread must never turn a projection into 99-100%
                # certainty. These are conservative minimum error scales.
                minimum_sd = {"points": 5.0, "rebounds": 2.5, "assists": 2.0}
                if market in minimum_sd:
                    sd = max(residual_spreads.get(market, minimum_sd[market]), minimum_sd[market])
                else:
                    component_markets = [key for key in ("points", "rebounds", "assists")
                                         if key in market.replace(" + ", "|").split("|")]
                    sd = math.sqrt(sum(max(residual_spreads.get(key, minimum_sd[key]), minimum_sd[key]) ** 2
                                         for key in component_markets))
                z = (float(line) + 0.5 - float(projection)) / sd
                p_over = 1 - (0.5 * (1 + math.erf(z / math.sqrt(2))))
                # Keep probabilities away from false certainty; this is an
                # estimate from a simple residual model, not a calibrated market.
                p_over = min(max(p_over, 0.05), 0.95)
                p_under = 1 - p_over
                best_side = "Over" if p_over >= p_under else "Under"
                best_probability = max(p_over, p_under)
                compare_rows.append({
                    "Player": row.get("player_line", row.get("player")),
                    "Game date": row.get("game_date"),
                    "Game time UTC": row.get("game_time_utc"),
                    "Team": row.get("team_projection", row.get("team")),
                    "Opponent": row.get("opponent_projection", row.get("opponent")),
                    "Market": market.title(),
                    "Projection": round(float(projection), 1),
                    "Prop line": float(line),
                    "Projection gap": round(float(projection) - float(line), 1),
                    "Over probability": f"{p_over * 100:.1f}%",
                    "Under probability": f"{p_under * 100:.1f}%",
                    "Best side": best_side,
                    "Estimated confidence": round(best_probability * 100, 1),
                    "Recent avg minutes": row.get("minutes_baseline"),
                    "Recent L5": None if len(projection_cols) > 1 else row.get({"points": "pts_last5", "rebounds": "reb_last5", "assists": "ast_last5"}.get(market)),
                    "Recent L10": None if len(projection_cols) > 1 else row.get({"points": "pts_last10", "rebounds": "reb_last10", "assists": "ast_last10"}.get(market)),
                    "Recent trend": None if len(projection_cols) > 1 else row.get({"points": "pts_trend", "rebounds": "reb_trend", "assists": "ast_trend"}.get(market)),
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
                st.dataframe(comparisons, width="stretch", hide_index=True)
                st.download_button("Download prop line comparisons", comparisons.to_csv(index=False), "nba_edge_prop_line_comparisons.csv", "text/csv")

                st.divider()
                st.subheader("🔥 Top model edges")
                st.caption("Ranked by estimated chance of the recommended side, not by bookmaker price or guaranteed profitability. Expand a pick to see why the model favours it.")
                confidence_pool = pd.DataFrame(compare_rows)
                confidence_pool = confidence_pool[confidence_pool["Estimated confidence"] >= 55].copy()
                confidence_pool = confidence_pool.sort_values("Estimated confidence", ascending=False).head(10)

                # Archive only single-stat, pre-game recommendations. Composite markets
                # are excluded until their exact settlement rules are implemented.
                _pick_history_path = DATA / "prop_pick_history.csv"
                _pick_markets = {"Points", "Rebounds", "Assists"}
                _new_picks = confidence_pool[confidence_pool["Market"].isin(_pick_markets)].copy()
                if not _new_picks.empty:
                    _now_utc = pd.Timestamp.now(tz="UTC")
                    _new_picks["_start"] = pd.to_datetime(_new_picks["Game time UTC"], utc=True, errors="coerce")
                    _new_picks = _new_picks[_new_picks["_start"].isna() | (_new_picks["_start"] > _now_utc)].copy()
                    _new_picks["Pick date"] = pd.to_datetime(_new_picks["Game date"], errors="coerce").dt.strftime("%Y-%m-%d")
                    _new_picks["Sky Bet odds"] = pd.NA
                    _new_picks["Actual result"] = "Pending"
                    _new_picks["Benchmark result"] = "Pending"
                    _new_picks = _new_picks.rename(columns={"Player":"player","Team":"team","Opponent":"opponent","Market":"market","Best side":"side","Prop line":"line","Projection":"projection","Estimated confidence":"confidence_pct","Game date":"game_date","Game time UTC":"game_time_utc"})
                    _keep = ["game_date","game_time_utc","player","team","opponent","market","side","line","projection","confidence_pct","Sky Bet odds","Actual result","Benchmark result"]
                    _new_picks = _new_picks[[x for x in _keep if x in _new_picks.columns]]
                    _old_picks = pd.read_csv(_pick_history_path) if _pick_history_path.exists() else pd.DataFrame()
                    _all_picks = pd.concat([_old_picks, _new_picks], ignore_index=True)
                    _dedupe = [x for x in ["game_date","player","team","opponent","market","side","line"] if x in _all_picks.columns]
                    _all_picks = _all_picks.drop_duplicates(_dedupe, keep="last")
                    _all_picks.to_csv(_pick_history_path, index=False)

                if confidence_pool.empty:
                    st.info("No matched prop currently reaches the 55% confidence threshold. No picks are being promoted.")
                else:
                    confidence_view = confidence_pool[[
                        "Player", "Team", "Opponent", "Market", "Best side",
                        "Prop line", "Projection", "Projection gap",
                        "Estimated confidence", "Recent avg minutes"
                    ]].copy()
                    confidence_view = confidence_view.rename(columns={
                        "Best side": "Pick",
                        "Estimated confidence": "Model confidence (%)",
                    })
                    # Mobile-friendly pick cards keep the full market name visible
                    # instead of hiding it in a horizontally clipped dataframe.
                    for _, pick in confidence_pool.iterrows():
                        market_key = str(pick["Market"]).lower()
                        stat_name = pick["Market"].lower()
                        projection_gap = abs(float(pick["Projection gap"]))
                        direction = "above" if pick["Best side"] == "Over" else "below"
                        title = (f"{pick['Player']} · {pick['Market']} · "
                                 f"{pick['Best side']} {pick['Prop line']:.1f}")
                        with st.expander(title):
                            st.markdown(f"**Market:** {pick['Market']}")
                            st.markdown(f"**Pick:** {pick['Best side']} {pick['Prop line']:.1f}")
                            st.markdown(f"**Projection:** {pick['Projection']:.1f} {stat_name}")
                            st.markdown(f"**Model confidence:** {pick['Estimated confidence']:.1f}%")
                            st.markdown(f"**Team / opponent:** {pick['Team']} vs {pick['Opponent']}")
                            reasons = [
                                f"The model projects **{pick['Projection']:.1f} {stat_name}**, which is **{projection_gap:.1f} {stat_name} {direction}** the listed line of {pick['Prop line']:.1f}.",
                            ]
                            l5, l10 = pick.get("Recent L5"), pick.get("Recent L10")
                            if pd.notna(l5) and pd.notna(l10):
                                form_word = "up" if float(l5) > float(l10) else "down" if float(l5) < float(l10) else "level"
                                reasons.append(f"Recent form: **{float(l5):.1f} over the last 5 games** versus **{float(l10):.1f} over the last 10** — the short-term average is {form_word}.")
                            minutes = pd.to_numeric(pd.Series([pick.get("Recent avg minutes")]), errors="coerce").iloc[0]
                            if pd.notna(minutes):
                                reasons.append(f"Recent playing-time baseline: **{float(minutes):.1f} minutes per game**. This is historical context, not a confirmed minutes forecast.")
                            for reason in reasons:
                                st.markdown(f"- {reason}")
                            st.caption("The probability is derived from the model projection and historical residual spread. It is an estimate, not a calibrated guarantee. No bookmaker odds are used or displayed.")
                    st.download_button(
                        "Download most confident picks",
                        confidence_view.to_csv(index=False),
                        "nba_edge_most_confident_picks.csv",
                        "text/csv",
                    )
            else:
                st.info("The prop feed returned data, but none of its player names and markets matched today's projections. Try refreshing the prop lines shortly.")
        except Exception as exc:
            st.warning("Current prop lines could not be loaded or matched. The projection dashboard is still available.")
            st.caption(str(exc))

        st.divider()
        st.subheader("🏆 Top projected performances")
        for market_label, market_col in [
            ("Points", "points_projection"),
            ("Rebounds", "rebounds_projection"),
            ("Assists", "assists_projection"),
            ("Three-pointers made", "threes_projection"),
            ("Steals", "steals_projection"),
            ("Blocks", "blocks_projection"),
            ("Turnovers", "turnovers_projection"),
        ]:
            if market_col not in predictions.columns:
                continue
            with st.expander(f"Top 10 — {market_label}", expanded=(market_label == "Points")):
                leaders = predictions.sort_values(market_col, ascending=False).head(10).copy()
                leaders = leaders.rename(columns={
                    "player": "Player", "team": "Team", "opponent": "Opponent",
                    "minutes_baseline": "Recent avg minutes", market_col: f"Projected {market_label.lower()}",
                })
                st.dataframe(
                    leaders[["Player", "Team", "Opponent", "Recent avg minutes", f"Projected {market_label.lower()}"]],
                    width="stretch", hide_index=True
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
        st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)
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
            st.dataframe(pd.DataFrame(daily_rows).sort_values(["Date", "Market"], ascending=[False, True]), width="stretch", hide_index=True)
    st.download_button("Download accuracy history", settled.to_csv(index=False), "nba_edge_accuracy_history.csv", "text/csv")

# Betting performance is tracked separately from projection MAE. Only archived
# pre-game recommendations are graded; actual Sky Bet prices are optional/manual.
PICK_HISTORY = DATA / "prop_pick_history.csv"
st.divider()
st.subheader("💷 NBA prop betting performance")
st.caption("One unit staked on every archived single-stat recommendation. Enter the actual Sky Bet decimal odds when available. A separate 1.91 fixed-odds benchmark is shown for consistent model comparison.")
if PICK_HISTORY.exists():
    try:
        picks = pd.read_csv(PICK_HISTORY)
    except Exception:
        picks = pd.DataFrame()
else:
    picks = pd.DataFrame()

if picks.empty:
    st.info("No pre-game prop picks have been archived yet. New eligible recommendations will be saved automatically before tip-off.")
else:
    if not settled.empty:
        _actual_map = {"Points":"actual_points","Rebounds":"actual_rebounds","Assists":"actual_assists"}
        for _market_name, _actual_col in _actual_map.items():
            _mask = picks["market"].astype(str).str.lower().eq(_market_name.lower())
            _day = settled[["game_date","player","team",_actual_col]].copy() if _actual_col in settled.columns else pd.DataFrame()
            if not _day.empty:
                _day["player_key"] = _day["player"].astype(str).str.lower().str.replace(r"[^a-z0-9]","",regex=True)
                picks["player_key"] = picks["player"].astype(str).str.lower().str.replace(r"[^a-z0-9]","",regex=True)
                _join = picks.loc[_mask, ["game_date","player_key","team","line","side"]].merge(_day[["game_date","player_key","team",_actual_col]],on=["game_date","player_key","team"],how="left")
                _actual_values = _join[_actual_col]
                _results = []
                for _idx, _row in _join.iterrows():
                    if pd.isna(_row[_actual_col]):
                        _results.append("Pending")
                    elif float(_row[_actual_col]) == float(_row["line"]):
                        _results.append("Push")
                    elif (_row["side"] == "Over" and float(_row[_actual_col]) > float(_row["line"])) or (_row["side"] == "Under" and float(_row[_actual_col]) < float(_row["line"])):
                        _results.append("Win")
                    else:
                        _results.append("Loss")
                picks.loc[_mask, "Actual result"] = _results
                picks.loc[_mask, "Benchmark result"] = _results
        picks.to_csv(PICK_HISTORY,index=False)

    _editable_cols = ["Sky Bet odds"]
    if "Sky Bet odds" not in picks:
        picks["Sky Bet odds"] = pd.NA
    _edited = st.data_editor(picks, hide_index=True, width="stretch", num_rows="fixed", disabled=[x for x in picks.columns if x not in _editable_cols], column_config={"Sky Bet odds":st.column_config.NumberColumn("Sky Bet decimal odds",min_value=1.01,max_value=100.0,step=0.01,format="%.2f",help="Enter the price actually available on Sky Bet before the event starts.")})
    if st.button("Save Sky Bet odds", key="save_nba_sky_odds"):
        _edited.to_csv(PICK_HISTORY,index=False)
        st.success("Sky Bet prices saved to the local pick history.")
        st.rerun()

    _graded = picks[picks["Actual result"].isin(["Win","Loss","Push"])].copy()
    _settled_count = len(_graded)
    _wins = int((_graded["Actual result"]=="Win").sum())
    _losses = int((_graded["Actual result"]=="Loss").sum())
    _pushes = int((_graded["Actual result"]=="Push").sum())
    _fixed_pl = float((_graded["Actual result"].map({"Win":0.91,"Loss":-1.0,"Push":0.0})).sum())
    _priced = _graded[pd.to_numeric(_graded["Sky Bet odds"],errors="coerce").notna()].copy()
    _priced["Sky Bet odds"] = pd.to_numeric(_priced["Sky Bet odds"],errors="coerce")
    _actual_pl = float(_priced.apply(lambda row: row["Sky Bet odds"]-1.0 if row["Actual result"]=="Win" else -1.0 if row["Actual result"]=="Loss" else 0.0,axis=1).sum()) if len(_priced) else 0.0
    _fixed_yield = (_fixed_pl / _settled_count * 100) if _settled_count else 0.0
    _actual_yield = (_actual_pl / len(_priced) * 100) if len(_priced) else None
    _m1,_m2,_m3,_m4,_m5=st.columns(5)
    _m1.metric("Settled picks",_settled_count)
    _m2.metric("Record",f"{_wins}-{_losses}-{_pushes}")
    _m3.metric("1.91 benchmark P/L",f"{_fixed_pl:+.2f}u")
    _m4.metric("Actual Sky Bet P/L",f"{_actual_pl:+.2f}u" if len(_priced) else "No prices entered")
    _m5.metric("Priced picks",len(_priced))
    _y1,_y2=st.columns(2)
    _y1.metric("1.91 benchmark yield",f"{_fixed_yield:+.1f}%")
    _y2.metric("Sky Bet yield",f"{_actual_yield:+.1f}%" if _actual_yield is not None else "Not available")
    _market_rows=[]
    for _market_name in ["Points","Rebounds","Assists"]:
        _subset=_graded[_graded["market"].astype(str).str.lower().eq(_market_name.lower())]
        _mw=int((_subset["Actual result"]=="Win").sum())
        _ml=int((_subset["Actual result"]=="Loss").sum())
        _mp=int((_subset["Actual result"]=="Push").sum())
        _benchmark=float(_subset["Actual result"].map({"Win":0.91,"Loss":-1.0,"Push":0.0}).sum())
        _market_rows.append({"Market":_market_name,"Record":f"{_mw}-{_ml}-{_mp}","1.91 P/L":f"{_benchmark:+.2f}u"})
    st.markdown("**Performance by market**")
    st.dataframe(pd.DataFrame(_market_rows),hide_index=True,width="stretch")
    st.download_button("Download NBA betting history",picks.to_csv(index=False),"nba_edge_betting_history.csv","text/csv")

if "predictions" in locals() and isinstance(predictions, pd.DataFrame) and not predictions.empty:
    st.divider()
    st.subheader("🧩 Same-player Bet Builders")
    st.caption("Two-leg milestones for the same player, such as 10+ points + 5+ rebounds. The joint chance is estimated together using a bivariate simulation, not by multiplying two separate prop probabilities.")
    try:
        from same_player_builders import generate_same_player_builders
        builders = generate_same_player_builders(predictions, MODELS)
        if builders.empty:
            st.info("No same-player combinations currently fall inside the model's target probability range. No forced picks.")
        else:
            builder_games = ["All games"] + sorted(builders["game"].dropna().unique().tolist())
            selected_builder_game = st.selectbox("Bet Builder game", builder_games, key="same_player_builder_game")
            builder_view = builders if selected_builder_game == "All games" else builders[builders["game"] == selected_builder_game]
            for _, builder in builder_view.sort_values("joint_probability_pct", ascending=False).iterrows():
                with st.container(border=True):
                    st.markdown(f"**{builder['player']} · {builder['team']} vs {builder['opponent']}**")
                    st.markdown(f"- {builder['leg_1'].title()}")
                    st.markdown(f"- {builder['leg_2'].title()}")
                    st.metric("Estimated chance both land", f"{builder['joint_probability_pct']:.1f}%")
                    st.caption(f"Model-only fair odds estimate: {builder['estimated_fair_decimal_odds']:.2f} · Not bookmaker odds")
            st.download_button("Download same-player Bet Builders", builder_view.to_csv(index=False), "nba_edge_same_player_bet_builders.csv", "text/csv")
    except Exception as exc:
        st.warning("Same-player Bet Builders could not be generated.")
        st.caption(str(exc))

with st.expander("About the model and its limitations"):
    st.markdown("""
- The current trained models predict **points, rebounds and assists** only.
- Player prop line numbers may be shown for comparison, but bookmaker names, odds, prices and expected-value calculations are not displayed.
- Player projections are generated for players found on ESPN's team rosters and matched to the saved historical feature snapshot.
- The model uses historical rolling statistics. It does **not** yet make reliable adjustments for confirmed starting lineups, late injury news, matchup-specific defensive factors or expected minutes changes.
- A projection is an estimate, not a guarantee. Track results over time before relying on it.
""")
