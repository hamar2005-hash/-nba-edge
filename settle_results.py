"""Settle archived NBA Edge projections and prop picks against ESPN box scores."""
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import argparse, re
import pandas as pd
import requests

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
DATA = Path("data")
PREDICTIONS = DATA / "prediction_history.csv"
SETTLED = DATA / "settled_results.csv"
PICK_HISTORY = DATA / "prop_pick_history.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NBAEdge/1.0)"}
MARKETS = {"points": "PTS", "rebounds": "REB", "assists": "AST", "threes": "FG3M"}
PICK_ACTUALS = {"points": "actual_points", "rebounds": "actual_rebounds", "assists": "actual_assists", "threes": "actual_threes"}

def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

def grade_pick(side, line, actual):
    """Grade a single-stat pick; exact-line outcomes are pushes."""
    if pd.isna(actual) or pd.isna(line):
        return "Pending"
    if float(actual) == float(line):
        return "Push"
    if (str(side).strip().lower() == "over" and float(actual) > float(line)) or (
        str(side).strip().lower() == "under" and float(actual) < float(line)
    ):
        return "Win"
    return "Loss"

def update_pick_history(settled, path=PICK_HISTORY):
    """Apply final box scores to archived single-stat picks using date, player and team."""
    path = Path(path)
    if not path.exists() or settled.empty:
        return 0
    picks = pd.read_csv(path)
    if picks.empty:
        return 0
    picks["player_key"] = picks["player"].map(norm)
    settled = settled.copy()
    settled["player_key"] = settled["player"].map(norm)
    updated = 0
    for market, actual_col in PICK_ACTUALS.items():
        if actual_col not in settled.columns:
            continue
        mask = picks["market"].astype(str).str.lower().eq(market)
        if not mask.any():
            continue
        actuals = settled[["game_date", "player_key", "team", actual_col]].drop_duplicates(
            ["game_date", "player_key", "team"], keep="last"
        )
        joined = picks.loc[mask, ["game_date", "player_key", "team", "line", "side"]].merge(
            actuals, on=["game_date", "player_key", "team"], how="left", validate="many_to_one"
        )
        results = [grade_pick(row.side, row.line, getattr(row, actual_col)) for row in joined.itertuples(index=False)]
        picks.loc[mask, "Actual result"] = results
        picks.loc[mask, "Benchmark result"] = results
        updated += sum(result != "Pending" for result in results)
    picks.drop(columns=["player_key"], errors="ignore").to_csv(path, index=False)
    return updated

def get_json(url, params=None):
    response = requests.get(url, params=params, headers=HEADERS, timeout=25)
    response.raise_for_status()
    return response.json()

def actuals_from_summary(summary, event_id, game_date):
    rows = []
    boxscore = summary.get("boxscore", {})
    for team_block in boxscore.get("players", []):
        team = team_block.get("team", {})
        for stat_group in team_block.get("statistics", []):
            names = [str(x).upper() for x in stat_group.get("names", [])]
            if not names:
                continue
            for athlete in stat_group.get("athletes", []):
                person = athlete.get("athlete", {})
                values = athlete.get("stats", [])
                if not person.get("displayName"):
                    continue
                stats = dict(zip(names, values))
                def number(key):
                    raw = stats.get(key, "0")
                    try:
                        return float(str(raw).replace(",", "").split(":")[0]) if raw not in ("", "-", None) else 0.0
                    except (TypeError, ValueError):
                        return 0.0
                rows.append({
                    "game_date": game_date, "event_id": str(event_id),
                    "player": person["displayName"], "player_key": norm(person["displayName"]),
                    "team": team.get("abbreviation", ""),
                    "actual_points": number("PTS"), "actual_rebounds": number("REB"),
                    "actual_assists": number("AST"), "actual_threes": number("FG3M"),
                })
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Date YYYY-MM-DD; defaults to yesterday in US Eastern time")
    parser.add_argument("--history", default=str(PREDICTIONS))
    parser.add_argument("--out", default=str(SETTLED))
    parser.add_argument("--picks", default=str(PICK_HISTORY))
    args = parser.parse_args()
    target = args.date or (datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)).isoformat()
    history_path = Path(args.history)
    if not history_path.exists():
        print(f"No archived predictions found at {history_path}; nothing to settle.")
        return
    history = pd.read_csv(history_path)
    history = history[history["game_date"].astype(str) == target].copy()
    if history.empty:
        print(f"No archived predictions for {target}; nothing to settle.")
        return
    scoreboard = get_json(f"{ESPN}/scoreboard", {"dates": target.replace("-", "")})
    actual_rows = []
    for event in scoreboard.get("events", []):
        event_id = event.get("id")
        status = ((event.get("status") or {}).get("type") or {}).get("completed", False)
        if not event_id or not status:
            continue
        summary = get_json(f"{ESPN}/summary", {"event": event_id})
        actual_rows.extend(actuals_from_summary(summary, event_id, target))
    if not actual_rows:
        print(f"No completed box scores available for {target}; will retry on the next scheduled run.")
        return
    actuals = pd.DataFrame(actual_rows).drop_duplicates(["game_date", "player_key", "team"], keep="last")
    history["player_key"] = history["player"].map(norm)
    # Team is part of the join key to prevent a same-name or stale-roster cross-match.
    merged = history.merge(
        actuals, on=["game_date", "player_key", "team"], how="inner",
        suffixes=("", "_actual"), validate="many_to_one"
    )
    if merged.empty:
        print(f"No player/team names matched ESPN box scores for {target}.")
        return
    for market in MARKETS:
        merged[f"{market}_error"] = merged[f"{market}_projection"] - merged[f"actual_{market}"]
        merged[f"{market}_absolute_error"] = merged[f"{market}_error"].abs()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    combined = pd.concat([old, merged], ignore_index=True)
    keys = [c for c in ["game_date", "player", "team", "opponent"] if c in combined]
    combined = combined.drop_duplicates(keys, keep="last")
    combined.to_csv(out_path, index=False)
    graded = update_pick_history(combined[combined["game_date"].astype(str) == target], args.picks)
    print(f"Settled {len(merged)} player rows for {target}; graded {graded} archived prop picks; saved {len(combined)} total rows to {out_path}.")

if __name__ == "__main__":
    main()
