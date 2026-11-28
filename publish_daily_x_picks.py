"""Select and publish NBA Edge's best daily single-stat props to X.

Only posts picks when historical calibration supports them. On days with no
qualified selections, publishes a transparent no-picks update instead.
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import joblib

from prop_lines import fetch_current_lines

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
PICKS_PATH = DATA / "prop_pick_history.csv"
POSTED_PATH = DATA / "x_posted_dates.json"
POST_LOG = DATA / "x_post_history.csv"
MARKETS = ("points", "rebounds", "assists", "threes")
PROJECTION_COLS = {
    "points": "points_projection", "rebounds": "rebounds_projection",
    "assists": "assists_projection", "threes": "threes_projection",
}
MIN_SD = {"points": 5.0, "rebounds": 2.5, "assists": 2.0, "threes": 1.23}
MAX_GAP = {"points": 10.0, "rebounds": 4.0, "assists": 3.5, "threes": 5.0}


def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def wilson_lower(wins, total, z=1.2816):
    """80% one-sided Wilson lower confidence bound."""
    if total <= 0:
        return 0.0
    p = wins / total
    den = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (centre - spread) / den)


def load_calibration():
    """Build market/confidence-bin hit-rate estimates from settled prior picks."""
    if not PICKS_PATH.exists():
        return {}
    history = pd.read_csv(PICKS_PATH)
    required = {"market", "confidence_pct", "Actual result"}
    if not required.issubset(history.columns):
        return {}
    history["market"] = history["market"].astype(str).str.lower()
    base_confidence = pd.to_numeric(history["confidence_pct"], errors="coerce")
    if "raw_confidence_pct" in history.columns:
        history["confidence_pct"] = pd.to_numeric(
            history["raw_confidence_pct"], errors="coerce"
        ).fillna(base_confidence)
    else:
        history["confidence_pct"] = base_confidence
    history["result"] = history["Actual result"].astype(str).str.lower()
    history = history[
        history["market"].isin(MARKETS)
        & history["confidence_pct"].between(50, 99)
        & history["result"].isin(["win", "loss"])
    ].copy()
    calibration = {}
    for market in MARKETS:
        subset = history[history["market"] == market]
        if len(subset) < 60:
            continue
        calibration[market] = []
        for lower in range(50, 100, 5):
            band = subset[subset["confidence_pct"].between(lower, lower + 4.999)]
            if len(band) < 20:
                continue
            wins = int((band["result"] == "win").sum())
            calibration[market].append({
                "low": lower, "high": lower + 5,
                "n": len(band), "wins": wins,
                "lower": wilson_lower(wins, len(band)),
                "hit_rate": wins / len(band),
            })
    return calibration


def calibrated_probability(market, raw_probability, calibration):
    """Return a conservative empirical lower bound; never fall back to raw confidence."""
    bands = calibration.get(market, [])
    if not bands:
        return None, 0
    raw_pct = raw_probability * 100
    candidates = [b for b in bands if b["low"] <= raw_pct < b["high"]]
    if not candidates:
        return None, 0
    band = candidates[0]
    return band["lower"], band["n"]


def normal_over_probability(projection, line, sd, market):
    boundary = line if market == "threes" else line + 0.5
    z = (boundary - projection) / sd
    return min(max(0.5 * (1 + math.erf(-z / math.sqrt(2))), 0.05), 0.95)


def load_residual_spreads():
    spreads = {}
    for market in MARKETS:
        path = MODELS / f"{market}_model.joblib"
        if path.exists():
            pack = joblib.load(path)
            spreads[market] = max(float(pack.get("residual_std", MIN_SD[market])), MIN_SD[market])
    return spreads


def select_picks(game_date):
    predictions_path = DATA / "daily_predictions.csv"
    if not predictions_path.exists():
        print("Daily predictions are missing; publishing a no-picks status update.")
        return []
    predictions = pd.read_csv(predictions_path)
    if predictions.empty:
        return []
    predictions = predictions[predictions["game_date"].astype(str) == game_date].copy()
    if predictions.empty:
        return []

    names = predictions["player"].dropna().astype(str).unique().tolist()
    try:
        lines = fetch_current_lines(names)
    except Exception as exc:
        print(f"Prop-line feed unavailable; publishing a no-picks status update: {exc}")
        return []
    if lines is None or lines.empty:
        return []

    predictions["player_key"] = predictions["player"].map(norm)
    predictions["team_key"] = predictions["team"].fillna("").astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    lines["player_key"] = lines["player"].map(norm)
    lines["team_key"] = lines["team"].fillna("").astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    merged = lines.merge(
        predictions,
        on="player_key", how="inner", suffixes=("_line", "_projection")
    )
    if merged.empty:
        return []

    now = datetime.now(timezone.utc)
    calibration = load_calibration()
    spreads = load_residual_spreads()
    candidates = []
    for row in merged.itertuples(index=False):
        market = str(getattr(row, "market", "")).lower()
        if market not in MARKETS or market not in spreads:
            continue
        player_team = str(getattr(row, "team_projection", "") or "").upper()
        line_team = str(getattr(row, "team_line", "") or "").upper()
        if player_team and line_team and player_team != line_team:
            continue
        status = str(getattr(row, "player_status", "")).lower()
        if any(term in status for term in ("out", "doubtful", "inactive", "suspended")):
            continue
        start = pd.to_datetime(getattr(row, "game_time_utc", None), utc=True, errors="coerce")
        if pd.notna(start) and start.to_pydatetime() <= now:
            continue
        projection = pd.to_numeric(getattr(row, PROJECTION_COLS[market]), errors="coerce")
        line = pd.to_numeric(getattr(row, "line"), errors="coerce")
        if pd.isna(projection) or pd.isna(line) or line < 0.5:
            continue
        if abs(float(projection) - float(line)) > MAX_GAP[market]:
            continue
        minutes = pd.to_numeric(getattr(row, "expected_minutes", getattr(row, "minutes_baseline", 0)), errors="coerce")
        if pd.isna(minutes) or float(minutes) < 18:
            continue
        p_over = normal_over_probability(float(projection), float(line), spreads[market], market)
        side = "Over" if p_over >= 0.5 else "Under"
        raw = max(p_over, 1 - p_over)
        calibrated, sample = calibrated_probability(market, raw, calibration)
        # A pick must clear an empirical lower-bound threshold, not merely a
        # normal-distribution estimate. Insufficient calibration means no pick.
        if calibrated is None or calibrated < 0.55 or sample < 20:
            continue
        candidates.append({
            "game_date": game_date,
            "game_time_utc": str(getattr(row, "game_time_utc", "")),
            "player": str(getattr(row, "player_projection")),
            "team": player_team or line_team,
            "opponent": str(getattr(row, "opponent_projection", "")),
            "market": market,
            "side": side,
            "line": float(line),
            "projection": float(projection),
            "confidence_pct": round(calibrated * 100, 1),
            "raw_confidence_pct": round(raw * 100, 1),
            "calibration_sample": int(sample),
            "Sky Bet odds": np.nan,
            "Actual result": "Pending",
            "Benchmark result": "Pending",
        })
    candidates.sort(key=lambda x: (x["confidence_pct"], x["calibration_sample"]), reverse=True)
    # Avoid stacking multiple markets on the same player in the daily shortlist.
    selected, used_players = [], set()
    for pick in candidates:
        key = norm(pick["player"])
        if key in used_players:
            continue
        selected.append(pick)
        used_players.add(key)
        if len(selected) == 3:
            break
    return selected


def append_pick_history(picks):
    if not picks:
        return
    new = pd.DataFrame(picks)
    if PICKS_PATH.exists():
        old = pd.read_csv(PICKS_PATH)
        all_picks = pd.concat([old, new], ignore_index=True)
    else:
        all_picks = new
    keys = ["game_date", "player", "team", "opponent", "market", "side", "line"]
    all_picks = all_picks.drop_duplicates(keys, keep="last")
    all_picks.to_csv(PICKS_PATH, index=False)


def build_post(game_date, picks):
    if not picks:
        return (
            f"NBA EDGE | {game_date}\n"
            "No picks cleared our historical calibration and reliability filters today. "
            "No forced selections. We track the results, not just the wins."
        )
    lines = [f"NBA EDGE | {game_date}", "Today's top calibrated player props:"]
    for i, pick in enumerate(picks, 1):
        market = pick["market"].upper()
        lines.append(
            f"{i}. {pick['player']} ({pick['team']}) {pick['side']} "
            f"{pick['line']:g} {market} | proj {pick['projection']:.1f} | "
            f"cal. {pick['confidence_pct']:.0f}%"
        )
    lines.append("Model estimates, not guarantees. Results tracked.")
    post = "\n".join(lines)
    # X's standard post limit is 280 characters for this account format.
    while len(post) > 280 and len(picks) > 2:
        picks.pop()
        return build_post(game_date, picks)
    if len(post) > 280:
        post = post[:276].rstrip() + "..."
    return post


def main():
    game_date = os.environ.get("NBA_EDGE_GAME_DATE") or datetime.now(
        timezone.utc
    ).astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).date().isoformat()
    api_key = os.environ.get("X_API_KEY", "").strip()
    api_secret = os.environ.get("X_API_SECRET", "").strip()
    access_token = os.environ.get("X_ACCESS_TOKEN", "").strip()
    access_token_secret = os.environ.get("X_ACCESS_TOKEN_SECRET", "").strip()
    if not all((api_key, api_secret, access_token, access_token_secret)):
        print("X OAuth 1.0a secrets are incomplete; X publishing skipped.")
        return
    from requests_oauthlib import OAuth1
    auth = OAuth1(api_key, api_secret, access_token, access_token_secret)
    POSTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    posted = json.loads(POSTED_PATH.read_text()) if POSTED_PATH.exists() else {}
    if game_date in posted:
        print(f"An NBA Edge X post already exists for {game_date}; skipping duplicate.")
        return

    try:
        picks = select_picks(game_date)
    except Exception as exc:
        print(f"Pick selection failed; publishing a no-picks status update: {exc}")
        picks = []
    post_text = build_post(game_date, picks)
    response = requests.post(
        "https://api.x.com/2/tweets",
        auth=auth,
        headers={"Content-Type": "application/json"},
        json={"text": post_text},
        timeout=30,
    )
    if response.status_code >= 300:
        raise RuntimeError(f"X post failed ({response.status_code}): {response.text[:500]}")
    append_pick_history(picks)
    payload = response.json()
    post_id = (payload.get("data") or {}).get("id")
    if not post_id:
        raise RuntimeError(f"X returned no post ID: {payload}")
    posted[game_date] = {"post_id": str(post_id), "posted_at_utc": datetime.now(timezone.utc).isoformat()}
    POSTED_PATH.write_text(json.dumps(posted, indent=2) + "\n")
    log_row = pd.DataFrame([{
        "game_date": game_date, "post_id": str(post_id),
        "posted_at_utc": posted[game_date]["posted_at_utc"],
        "qualified_picks": len(picks), "post_text": post_text,
    }])
    old_log = pd.read_csv(POST_LOG) if POST_LOG.exists() else pd.DataFrame()
    pd.concat([old_log, log_row], ignore_index=True).to_csv(POST_LOG, index=False)
    print(f"Published NBA Edge daily X post {post_id} with {len(picks)} qualified picks.")


if __name__ == "__main__":
    main()
