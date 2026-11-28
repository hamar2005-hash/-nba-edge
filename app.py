from pathlib import Path
import json, math, os, subprocess, sys, zipfile, html, re
import requests
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from probability_utils import probability_at_least

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
BUNDLE = ROOT / "NBA_Trained_Models.zip"
PREDICTIONS = DATA / "daily_predictions.csv"


def mobile_player_cards(frame, market_columns, title_column="player", line_lookup=None):
    """Render compact, readable player cards for narrow screens."""
    cards = []
    for _, row in frame.iterrows():
        player = html.escape(str(row.get(title_column, "Unknown")))
        matchup = html.escape(str(row.get("game", "")))
        team = html.escape(str(row.get("team", "")))
        status = html.escape(str(row.get("player_status", "")))
        metrics = []
        for label, column in market_columns:
            value = row.get(column)
            if pd.notna(value):
                try:
                    value = f"{float(value):.1f}"
                except (TypeError, ValueError):
                    value = str(value)
            else:
                value = "—"
            line_value = None
            if line_lookup is not None:
                player_key = re.sub(r"[^a-z0-9]", "", str(row.get("player", "")).lower())
                market_key = {"PTS": "points", "REB": "rebounds", "AST": "assists", "3PM": "threes"}.get(label)
                line_value = line_lookup.get((player_key, market_key)) if market_key else None
            line_text = f'<small>Line: {html.escape(str(line_value))}</small>' if line_value is not None else '<small>Line: —</small>'
            metrics.append('<div class="edge-mobile-stat"><span>' + html.escape(label) + '</span><strong>' + html.escape(str(value)) + '</strong>' + line_text + '</div>')
        cards.append(
            '<article class="edge-mobile-card">'
            + '<div class="edge-mobile-card-head"><strong>' + player + '</strong><span>' + team + '</span></div>'
            + '<div class="edge-mobile-matchup">' + matchup + '</div>'
            + ('<div class="edge-mobile-status">' + status + '</div>' if status and status.lower() != "nan" else "")
            + '<div class="edge-mobile-stats">' + ''.join(metrics) + '</div></article>'
        )
    st.markdown('<div class="edge-mobile-card-list">' + ''.join(cards) + '</div>', unsafe_allow_html=True)

st.set_page_config(page_title="NBA Edge", page_icon="🏀", layout="wide")

# Restrained sports-data dashboard styling: flat surfaces, clear hierarchy, minimal decoration.
st.markdown("""
<style>

/* Dedicated Streamlit component hosts the full-screen video layer. */
/* Streamlit renders components as iframe elements, not divs.
   Pin the component iframe to the viewport so its 1px layout placeholder
   does not clip the full-screen video inside it. */
iframe[data-testid="stIFrame"],
[data-testid="stIFrame"] {
 position:fixed !important;
 top:0 !important; left:0 !important; right:0 !important; bottom:0 !important;
 display:block !important;
 width:100vw !important; height:100vh !important;
 min-width:100vw !important; min-height:100vh !important;
 max-width:none !important; max-height:none !important;
 z-index:0 !important; pointer-events:none !important;
 border:0 !important; overflow:hidden !important;
}
.stApp { position:relative; background:#080c11 !important; }
.stApp { position:relative; background:#080c11 !important; }
.stApp > header, .stApp [data-testid="stAppViewContainer"] { position:relative; z-index:1; background:transparent; }
[data-testid="stSidebar"] { background:rgba(15,21,28,.96); }
.block-container { position:relative; z-index:1; }
:root { --edge-orange:#e87524; --edge-line:#303944; --edge-muted:#a5afb9; }
.stApp { background:#080c11 !important; color:#e7ebef; }
.block-container { max-width:1440px; padding:1.2rem 1.8rem 3rem; }
[data-testid="stHeader"] { background:#10151b; }
h1,h2,h3 { color:#f0f2f4; letter-spacing:-.015em; font-weight:650; }
h2 { padding-top:.3rem; font-size:1.4rem; } h3 { font-size:1.08rem; }
p,[data-testid="stCaptionContainer"] { color:var(--edge-muted); }
[data-testid="stMetric"] { background:#171e26; border:1px solid var(--edge-line); border-radius:5px; padding:14px 16px; min-height:96px; box-shadow:none; }
[data-testid="stMetricLabel"] { color:#a5afb9; font-size:.76rem; font-weight:550; }
[data-testid="stMetricValue"] { color:#f2f4f6; font-size:1.75rem; font-weight:650; letter-spacing:-.02em; }
[data-testid="stDataFrame"],[data-testid="stTable"] { border:1px solid var(--edge-line); border-radius:4px; overflow:hidden; background:#171e26; }
[data-testid="stDataFrame"] [role="columnheader"] { background:#202a35; }
[data-testid="stDataFrame"] [role="gridcell"] { font-size:.85rem; }
.stButton > button,.stDownloadButton > button,[data-testid="stLinkButton"] a { border-radius:4px; border:1px solid #46515d; transition:border-color .12s ease; font-weight:600; min-height:2.45rem; }
.stButton > button[kind="primary"] { background:var(--edge-orange); color:#11151a; border:1px solid var(--edge-orange); box-shadow:none; }
.stButton > button:hover,.stDownloadButton > button:hover { border-color:var(--edge-orange); transform:none; }
[data-testid="stTextInput"] input,[data-testid="stSelectbox"] > div,[data-testid="stMultiSelect"] > div,[data-testid="stDateInput"] input { border-radius:4px; background:#171e26; }
[data-testid="stExpander"] { border:1px solid var(--edge-line); border-radius:4px; background:#141b22; }
[data-testid="stSidebar"] { background:#141b22; border-right:1px solid var(--edge-line); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#a5afb9; }
[data-testid="stSidebar"] [data-testid="stButton"] button { width:100%; }
hr { border-color:var(--edge-line); margin:1.35rem 0; }
.edge-hero { border-bottom:2px solid var(--edge-orange); padding:4px 0 15px; margin:0 0 22px; background:transparent; }
.edge-kicker { color:#c0c7ce; text-transform:uppercase; letter-spacing:.09em; font-size:.68rem; font-weight:650; }
.edge-title { color:#f3f5f7; font-size:clamp(1.8rem,4vw,2.55rem); font-weight:750; letter-spacing:-.035em; line-height:1.08; margin:5px 0; }
.edge-subtitle { color:#a5afb9; font-size:.9rem; max-width:720px; }
.edge-mobile-card-list { display:none; }
.edge-pill { display:inline-block; padding:2px 7px; margin:8px 5px 0 0; border-radius:3px; background:#202a35; border:1px solid #394552; color:#cbd2d9; font-size:.65rem; font-weight:600; letter-spacing:.02em; }
[data-testid="stStatusWidget"] { border-radius:4px; }
@media (max-width:800px) {
 .st-key-desktop-projections,.st-key-desktop-threes { display:none !important; }
 .edge-mobile-card-list { display:grid; gap:8px; }
 .edge-mobile-card { background:#171e26; border:1px solid #303944; border-radius:4px; padding:11px 12px; }
 .edge-mobile-card-head { display:flex; justify-content:space-between; gap:8px; align-items:center; color:#edf0f3; font-size:.9rem; }
 .edge-mobile-card-head span { color:#a5afb9; font-size:.75rem; }
 .edge-mobile-matchup { color:#929eaa; font-size:.7rem; margin-top:3px; }
 .edge-mobile-status { color:#e99a5e; font-size:.68rem; margin-top:3px; }
 .edge-mobile-stats { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:5px; margin-top:9px; }
 .edge-mobile-stat { background:#202a35; border-radius:3px; padding:7px 4px; text-align:center; min-width:0; }
 .edge-mobile-stat span { display:block; color:#a5afb9; font-size:.62rem; white-space:nowrap; }
 .edge-mobile-stat strong { display:block; color:#f0f2f4; font-size:.92rem; margin-top:2px; }
 .edge-mobile-stat small { display:block; color:#a5afb9; font-size:.57rem; margin-top:3px; white-space:nowrap; }
 .block-container { padding:1rem .8rem 2.2rem; }
 .edge-hero { padding-bottom:12px; }
 [data-testid="stMetric"] { padding:10px; min-height:82px; }
 [data-testid="stMetricValue"] { font-size:1.4rem; }
}
</style>
<div class="edge-hero">
  <div class="edge-kicker">Player prop analytics</div>
  <div class="edge-title">NBA EDGE</div>
  <div class="edge-subtitle">Daily projections, market comparisons and results tracking.</div>
  <span class="edge-pill">PTS</span><span class="edge-pill">REB</span><span class="edge-pill">AST</span><span class="edge-pill">3PM</span>
</div>
""", unsafe_allow_html=True)


# Render the YouTube highlight reel in its own Streamlit iframe component.
# Keeping the iframe separate from st.markdown avoids HTML sanitisation issues.
components.html("""
<!doctype html>
<html><head><style>
html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:transparent}
iframe{position:fixed;top:50%;left:50%;width:max(100vw,177.78vh);height:max(100vh,56.25vw);
transform:translate(-50%,-50%);border:0;opacity:.62;filter:brightness(1.22) contrast(1.12) saturate(1.10);pointer-events:none}
/* Keep the footage visible; use a gentle gradient only where dashboard content needs contrast. */
.shade{position:fixed;inset:0;background:linear-gradient(180deg,rgba(8,12,17,.22) 0%,rgba(8,12,17,.34) 46%,rgba(8,12,17,.62) 100%);pointer-events:none}
/* Portrait phones: bias the crop toward LeBron and keep the dashboard readable. */
@media (max-width:600px) and (orientation:portrait) {
 iframe{left:44%;width:max(100vw,177.78vh);height:max(100vh,56.25vw);opacity:.54}
 .shade{background:linear-gradient(180deg,rgba(8,12,17,.28) 0%,rgba(8,12,17,.38) 48%,rgba(8,12,17,.70) 100%)}
}
</style></head><body>
<iframe src="https://www.youtube-nocookie.com/embed/pkBfvNiuQuk?autoplay=1&mute=1&controls=0&loop=1&playlist=pkBfvNiuQuk&playsinline=1&rel=0&modestbranding=1"
title="LeBron James highlights" allow="autoplay; encrypted-media; picture-in-picture" tabindex="-1"></iframe>
<div class="shade"></div>
</body></html>
""", height=1, width=1, scrolling=False)

# Restore the committed trained model bundle if Streamlit's instance starts clean
# or if a previous extraction left only part of the required files behind.
model_files = [MODELS / f"{market}_model.joblib" for market in ("points", "rebounds", "assists")]
required_bundle_files = model_files + [MODELS / "latest_player_features.csv"]
if BUNDLE.exists() and (not all(path.exists() for path in required_bundle_files) or not (MODELS / "threes_model.joblib").exists()):
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
        st.success("Core trained models available")
        if (MODELS / "threes_model.joblib").exists():
            st.success("Dedicated 3-pointers model available")
        else:
            st.info("3-pointers model not trained yet. Run the 3PM training workflow to enable it.")
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
        with st.spinner("Reloading the latest trained models, then generating projections…"):
            try:
                # Always restore the committed bundle before a manual refresh. This
                # prevents a long-lived Streamlit instance from using stale model files.
                if BUNDLE.exists():
                    with zipfile.ZipFile(BUNDLE) as archive:
                        archive.extractall(ROOT)
                threes_model = MODELS / "threes_model.joblib"
                if not threes_model.exists():
                    st.error("The latest model bundle does not contain models/threes_model.joblib. The 3PM model cannot be used yet.")
                else:
                    run = subprocess.run(
                        [sys.executable, str(ROOT / "auto_daily.py"), "--models", str(MODELS), "--out", str(PREDICTIONS)],
                        cwd=ROOT, capture_output=True, text=True, timeout=300
                    )
                    if run.returncode == 0:
                        refreshed = pd.read_csv(PREDICTIONS)
                        if "threes_projection" not in refreshed.columns:
                            st.error("Daily refresh completed, but threes_projection is still missing. The model loaded, but the prediction generator did not emit the 3PM column.")
                            st.code(run.stdout[-4000:])
                        else:
                            st.success(run.stdout.strip() or "Projections refreshed, including dedicated 3PM forecasts.")
                            st.rerun()
                    else:
                        st.error("Refresh failed. The schedule/roster feed may be temporarily unavailable.")
                        st.code((run.stderr or run.stdout)[-8000:])
            except Exception as exc:
                st.error(f"Could not reload the latest model bundle or refresh projections: {exc}")
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
        d.metric("Markets", "4" if "threes_projection" in predictions.columns else "3")

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
        with st.container(key="desktop-projections"):
            st.dataframe(display[[c for c in cols if c in display]], width="stretch", hide_index=True)
        @st.cache_data(ttl=300, show_spinner=False)
        def fetch_card_prop_lines(player_names):
            import importlib
            import prop_lines as prop_lines_module
            importlib.reload(prop_lines_module)
            return prop_lines_module.fetch_current_lines(player_names)

        card_line_lookup = {}
        try:
            card_lines = fetch_card_prop_lines(tuple(predictions["player"].dropna().astype(str).unique()))
            if not card_lines.empty:
                for _, line_row in card_lines.iterrows():
                    player_key = re.sub(r"[^a-z0-9]", "", str(line_row.get("player", "")).lower())
                    market_key = str(line_row.get("market", "")).lower()
                    if market_key in {"points", "rebounds", "assists", "threes"}:
                        card_line_lookup[(player_key, market_key)] = line_row.get("line")
        except Exception as line_exc:
            st.caption(f"Live prop lines are temporarily unavailable: {line_exc}")

        st.subheader("Games playing today")
        st.caption("Open a game to see both teams together. Each team shows its five highest-minute projected players plus up to three key rotation options. Starting lineups are not confirmed in this feed, so this is a role-based estimate. Each stat shows the model projection and current prop line.")
        rotation_source = view.copy()
        rotation_minutes_col = "expected_minutes" if "expected_minutes" in rotation_source.columns else "minutes_baseline"
        if rotation_minutes_col in rotation_source.columns:
            rotation_source[rotation_minutes_col] = pd.to_numeric(rotation_source[rotation_minutes_col], errors="coerce")
        for matchup_name in sorted(rotation_source["game"].dropna().astype(str).unique()):
            game_rows = rotation_source[rotation_source["game"].astype(str) == matchup_name].copy()
            team_names = sorted(game_rows["team"].dropna().astype(str).unique())
            with st.expander(f"{matchup_name} · {len(team_names)} teams", expanded=False):
                for team_name in team_names:
                    team_rows = game_rows[game_rows["team"].astype(str) == team_name].copy()
                    team_rows = team_rows.sort_values(rotation_minutes_col, ascending=False, na_position="last").head(8)
                    opponent_name = str(team_rows["opponent"].iloc[0]) if "opponent" in team_rows.columns and not team_rows.empty else ""
                    st.markdown(f"**{team_name} vs {opponent_name}**")
                    st.caption(f"Top five are the likely starting group; next three are key rotation options, ranked by {rotation_minutes_col.replace('_', ' ')}.")
                    mobile_player_cards(
                        team_rows,
                        [("MIN", rotation_minutes_col), ("PTS", "points_projection"),
                         ("REB", "rebounds_projection"), ("AST", "assists_projection"),
                         ("3PM", "threes_projection")],
                        line_lookup=card_line_lookup,
                    )

        # Made threes now live in the standard projections table and team cards.
        # The standalone section is removed to avoid duplicating the same players.

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
        # Last-five and last-ten prop hit rates are calculated from individual completed games.
        st.divider()
        st.subheader("Last 5 & Last 10 game prop hits")
        st.caption("How often each player has cleared common prop milestones in their most recent games. Only appearances with minutes played count; this is historical hit rate, not a prediction.")
        RECENT_LOGS = DATA / "recent_player_game_logs.csv"
        if not RECENT_LOGS.exists():
            st.info("Recent game logs are being prepared by the historical-data workflow. Hit rates will appear after its next successful run.")
        else:
            try:
                recent_logs = pd.read_csv(RECENT_LOGS, parse_dates=["GAME_DATE"])
                recent_logs["PLAYER_KEY"] = recent_logs["PLAYER_NAME"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
                recent_logs["MIN"] = pd.to_numeric(recent_logs["MIN"], errors="coerce")
                recent_logs = recent_logs[recent_logs["MIN"].fillna(0) > 0].copy()
                recent_logs = recent_logs.sort_values(["PLAYER_KEY", "GAME_DATE", "GAME_ID"], ascending=[True, False, False])
                current_players = predictions[["player", "team", "opponent", "game"]].copy()
                current_players["PLAYER_KEY"] = current_players["player"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
                market_specs = {"Points": ("PTS", [10, 15, 20, 25, 30, 35]), "Rebounds": ("REB", [3, 5, 7, 10, 12, 15]), "Assists": ("AST", [3, 5, 7, 10, 12, 15]), "Threes made": ("FG3M", [1, 2, 3, 4, 5, 6])}
                hit_rows = []
                for _, player_row in current_players.drop_duplicates("PLAYER_KEY").iterrows():
                    player_games = recent_logs[recent_logs["PLAYER_KEY"] == player_row["PLAYER_KEY"]].head(10)
                    if player_games.empty:
                        continue
                    for market_name, (stat_col, thresholds) in market_specs.items():
                        if stat_col not in player_games.columns:
                            continue
                        values = pd.to_numeric(player_games[stat_col], errors="coerce").fillna(0).tolist()
                        for threshold in thresholds:
                            row = {"Player": player_row["player"], "Team": player_row["team"], "Opponent": player_row["opponent"], "Market": market_name, "Prop": f"{threshold}+"}
                            for window in (5, 10):
                                sample = values[:window]
                                count = sum(value >= threshold for value in sample)
                                row[f"L{window} Hits"] = f"{count}/{len(sample)}"
                                row[f"L{window} Rate"] = round(count / len(sample) * 100, 1) if sample else None
                            for idx in range(10):
                                row[f"G{idx+1}"] = values[idx] if idx < len(values) else None
                            hit_rows.append(row)
                if hit_rows:
                    hit_frame = pd.DataFrame(hit_rows)
                    hit_a, hit_b, hit_c = st.columns(3)
                    market_filter = hit_a.selectbox("Market", list(market_specs), key="recent_hit_market")
                    min_hits = hit_b.selectbox("Minimum hits", ["Any", "5/5", "4/5+", "3/5+", "10/10", "8/10+", "6/10+"], key="recent_hit_min")
                    player_filter = hit_c.text_input("Find player", key="recent_hit_player")
                    custom_threshold = st.number_input("Custom prop line", min_value=0.5, max_value=100.0, value=18.0, step=0.5, key="recent_hit_custom_line")
                    # Add a user-defined threshold for every player on the current slate.
                    custom_stat = market_specs[market_filter][0]
                    custom_rows = []
                    for _, player_row in current_players.drop_duplicates("PLAYER_KEY").iterrows():
                        player_games = recent_logs[recent_logs["PLAYER_KEY"] == player_row["PLAYER_KEY"]].head(10)
                        if player_games.empty or custom_stat not in player_games.columns:
                            continue
                        values = pd.to_numeric(player_games[custom_stat], errors="coerce").fillna(0).tolist()
                        row = {"Player": player_row["player"], "Team": player_row["team"], "Opponent": player_row["opponent"], "Market": market_filter, "Prop": f"{custom_threshold:g}+"}
                        for window in (5, 10):
                            sample = values[:window]
                            count = sum(value >= custom_threshold for value in sample)
                            row[f"L{window} Hits"] = f"{count}/{len(sample)}"
                            row[f"L{window} Rate"] = round(count / len(sample) * 100, 1) if sample else None
                        for idx in range(10):
                            row[f"G{idx+1}"] = values[idx] if idx < len(values) else None
                        custom_rows.append(row)
                    if custom_rows:
                        hit_frame = pd.concat([hit_frame, pd.DataFrame(custom_rows)], ignore_index=True)
                    filtered_hits = hit_frame[hit_frame["Market"] == market_filter].copy()
                    if player_filter:
                        filtered_hits = filtered_hits[filtered_hits["Player"].str.contains(player_filter, case=False, na=False)]
                    if min_hits == "5/5":
                        filtered_hits = filtered_hits[pd.to_numeric(filtered_hits["L5 Rate"], errors="coerce") == 100]
                    elif min_hits == "4/5+":
                        filtered_hits = filtered_hits[pd.to_numeric(filtered_hits["L5 Rate"], errors="coerce") >= 80]
                    elif min_hits == "3/5+":
                        filtered_hits = filtered_hits[pd.to_numeric(filtered_hits["L5 Rate"], errors="coerce") >= 60]
                    elif min_hits == "10/10":
                        filtered_hits = filtered_hits[pd.to_numeric(filtered_hits["L10 Rate"], errors="coerce") == 100]
                    elif min_hits == "8/10+":
                        filtered_hits = filtered_hits[pd.to_numeric(filtered_hits["L10 Rate"], errors="coerce") >= 80]
                    elif min_hits == "6/10+":
                        filtered_hits = filtered_hits[pd.to_numeric(filtered_hits["L10 Rate"], errors="coerce") >= 60]
                    tab5, tab10 = st.tabs(["Last 5 games", "Last 10 games"])
                    with tab5:
                        view5 = filtered_hits[["Player", "Team", "Opponent", "Prop", "L5 Hits", "L5 Rate", "G1", "G2", "G3", "G4", "G5"]].copy()
                        view5 = view5.rename(columns={"L5 Hits":"Hits", "L5 Rate":"Hit rate (%)", "G1":"Most recent", "G2":"Game 2", "G3":"Game 3", "G4":"Game 4", "G5":"Oldest"})
                        st.dataframe(view5.sort_values(["Hit rate (%)", "Hits"], ascending=False), width="stretch", hide_index=True)
                    with tab10:
                        view10 = filtered_hits[["Player", "Team", "Opponent", "Prop", "L10 Hits", "L10 Rate", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"]].copy()
                        view10 = view10.rename(columns={"L10 Hits":"Hits", "L10 Rate":"Hit rate (%)", "G1":"Most recent", "G10":"Oldest"})
                        st.dataframe(view10.sort_values(["Hit rate (%)", "Hits"], ascending=False), width="stretch", hide_index=True)
                    st.caption("Game columns run from most recent to oldest. Hit rates use each player's last available appearances, not calendar days. Historical streaks do not guarantee the next result.")
                    st.download_button("Download prop hit-rate history", filtered_hits.to_csv(index=False), "nba_edge_last_5_last_10_prop_hits.csv", "text/csv")
                else:
                    st.info("No matching recent player game logs were found for today's slate.")
            except Exception as hit_exc:
                st.warning("Recent prop hit rates could not be loaded.")
                st.caption(str(hit_exc))

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

            # Compact three-market board: show each player's listed PTS/REB/AST
            # line beside the matching model projection in one row.
            core_markets = {
                "points": ("points_projection", "Points"),
                "rebounds": ("rebounds_projection", "Rebounds"),
                "assists": ("assists_projection", "Assists"),
            }
            core = compare[compare["market"].isin(core_markets)].copy()
            if not core.empty:
                core["Player"] = core.get("player_line", core.get("player"))
                core["Team"] = core.get("team_line", core.get("team"))
                core["Opponent"] = core.get("opponent_line", core.get("opponent"))
                core["Projection"] = core.apply(
                    lambda item: pd.to_numeric(item.get(core_markets[item["market"]][0]), errors="coerce"),
                    axis=1,
                )
                core["Line"] = pd.to_numeric(core["line"], errors="coerce")
                core = core.dropna(subset=["Projection", "Line"])
                if not core.empty:
                    def display_values(values):
                        unique = sorted({float(value) for value in values if pd.notna(value)})
                        return ", ".join(f"{value:g}" for value in unique) if unique else "—"
                    board_rows = []
                    for keys, player_rows in core.groupby(["Player", "Team", "Opponent"], dropna=False):
                        board = {"Player": keys[0], "Team": keys[1], "Opponent": keys[2]}
                        for market_key, (_, label) in core_markets.items():
                            market_rows = player_rows[player_rows["market"] == market_key]
                            board[f"{label} line"] = display_values(market_rows["Line"])
                            board[f"{label} projection"] = (
                                round(float(market_rows["Projection"].iloc[0]), 1)
                                if not market_rows.empty else "—"
                            )
                        board_rows.append(board)
                    st.subheader("📋 Points, rebounds & assists: lines vs projections")
                    st.caption("Each listed prop line is shown beside NBA Edge's projection. Multiple available lines are listed together; a dash means the feed did not provide that market.")
                    st.dataframe(
                        pd.DataFrame(board_rows).sort_values(["Team", "Player"]),
                        width="stretch", hide_index=True,
                    )

            if compare.empty:
                st.warning("No prop lines passed the player-and-team identity check. The app will not force a match.")
            # Use the model's residual spread to estimate over/under probabilities.
            market_to_col = {
                "points": ("points_projection",),
                "rebounds": ("rebounds_projection",),
                "assists": ("assists_projection",),
                "threes": ("threes_projection",),
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
                hard_max = {"points": 60.0, "rebounds": 30.0, "assists": 25.0, "threes": 15.0,
                            "points + rebounds": 85.0, "points + assists": 85.0,
                            "rebounds + assists": 55.0, "points + rebounds + assists": 110.0}
                # Reject lines that are too far from the projection to be
                # credible same-market comparisons. The previous relative
                # limits admitted extreme mismatches (e.g. 20.5 REB vs 11.0).
                max_gap = {
                    "points": 10.0, "rebounds": 4.0, "assists": 3.5, "threes": 5.0,
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
                minimum_sd = {"points": 5.0, "rebounds": 2.5, "assists": 2.0, "threes": 1.23}
                if market in minimum_sd:
                    sd = max(residual_spreads.get(market, minimum_sd[market]), minimum_sd[market])
                else:
                    component_markets = [key for key in ("points", "rebounds", "assists")
                                         if key in market.replace(" + ", "|").split("|")]
                    sd = math.sqrt(sum(max(residual_spreads.get(key, minimum_sd[key]), minimum_sd[key]) ** 2
                                         for key in component_markets))
                # For made-threes half-point lines, the line itself is the
                # continuity-corrected boundary (e.g. 2.5 means 3+ made).
                # Retain the existing threshold convention for other markets.
                boundary = float(line) if market == "threes" else float(line) + 0.5
                z = (boundary - float(projection)) / sd
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
                market_filter = st.multiselect("Markets", ["Points", "Rebounds", "Assists", "Threes"], default=["Points", "Rebounds", "Assists", "Threes"], key="line_markets")
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
                _pick_markets = {"Points", "Rebounds", "Assists", "Threes"}
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

# The scheduled workflow writes a compact snapshot so the dashboard can show
# the latest report even when nobody has opened the app to trigger settlement.
REPORT_SNAPSHOT = DATA / "accuracy_report.json"
if REPORT_SNAPSHOT.exists():
    try:
        _accuracy_report = json.loads(REPORT_SNAPSHOT.read_text(encoding="utf-8"))
        _report_windows = _accuracy_report.get("windows", {})
        _report_7d = _report_windows.get("7d", {})
        _report_30d = _report_windows.get("30d", {})
        _report_daily = _report_windows.get("daily", {})
        st.caption(
            f"Automatic report for {_accuracy_report.get('report_date', 'unknown date')} "
            f"· Generated {_accuracy_report.get('generated_at_utc', 'time unavailable')} UTC"
        )
        if _accuracy_report.get("settlement_status") != "settled":
            st.info("The latest automatic report has no settled player rows yet. It will populate after final box scores are available.")
        st.caption("Mean absolute error is shown separately for each market because points, rebounds, assists and threes use different units.")
        _market_report_rows = []
        for _market_name in ("Points", "Rebounds", "Assists", "Three-pointers made"):
            _d = _report_daily.get("markets", {}).get(_market_name, {})
            _w = _report_7d.get("markets", {}).get(_market_name, {})
            _m = _report_30d.get("markets", {}).get(_market_name, {})
            _market_report_rows.append({
                "Market": _market_name,
                "Yesterday MAE": _d.get("mae"),
                "Yesterday players": _d.get("players", 0),
                "7-day MAE": _w.get("mae"),
                "7-day players": _w.get("players", 0),
                "30-day MAE": _m.get("mae"),
                "30-day players": _m.get("players", 0),
            })
        st.dataframe(pd.DataFrame(_market_report_rows), width="stretch", hide_index=True)
        _daily_pick = _report_daily.get("prop_picks", {})
        if _daily_pick.get("settled", 0):
            _daily_hit = _daily_pick.get("hit_rate")
            st.caption(
                f"Yesterday's archived prop picks: "
                f"{_daily_pick.get('wins', 0)}-{_daily_pick.get('losses', 0)}-{_daily_pick.get('pushes', 0)}"
                + (f" · {_daily_hit:.1f}% hit rate (pushes excluded)" if _daily_hit is not None else "")
            )
    except Exception as _report_error:
        st.warning(f"Could not read the latest automatic accuracy report: {_report_error}")
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
        ("Three-pointers made", "threes_projection", "actual_threes", "threes_absolute_error"),
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
                ("Three-pointers made", "threes_absolute_error"),
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
        _actual_map = {"Points":"actual_points","Rebounds":"actual_rebounds","Assists":"actual_assists","Threes":"actual_threes"}
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
    for _market_name in ["Points","Rebounds","Assists","Threes"]:
        _subset=_graded[_graded["market"].astype(str).str.lower().eq(_market_name.lower())]
        _mw=int((_subset["Actual result"]=="Win").sum())
        _ml=int((_subset["Actual result"]=="Loss").sum())
        _mp=int((_subset["Actual result"]=="Push").sum())
        # Count-based calculation avoids dtype/coercion failures in older persisted pick-history CSVs.
        _benchmark=(_mw * 0.91) - _ml
        _decided = _mw + _ml
        _hit_rate = (_mw / _decided * 100) if _decided else None
        _market_rows.append({
            "Market":_market_name,
            "Settled":_mw + _ml + _mp,
            "Record":f"{_mw}-{_ml}-{_mp}",
            "Hit rate":f"{_hit_rate:.1f}%" if _hit_rate is not None else "Awaiting results",
            "1.91 P/L":f"{_benchmark:+.2f}u",
        })
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