"""Settle archived NBA Edge projections against ESPN final box scores."""
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
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NBAEdge/1.0)"}
MARKETS = {"points": "PTS", "rebounds": "REB", "assists": "AST"}

def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

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
                    "game_date": game_date,
                    "event_id": str(event_id),
                    "player": person["displayName"],
                    "player_key": norm(person["displayName"]),
                    "team": team.get("abbreviation", ""),
                    "actual_points": number("PTS"),
                    "actual_rebounds": number("REB"),
                    "actual_assists": number("AST"),
                })
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Date YYYY-MM-DD; defaults to yesterday in US Eastern time")
    parser.add_argument("--history", default=str(PREDICTIONS))
    parser.add_argument("--out", default=str(SETTLED))
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
    merged = history.merge(actuals, on=["game_date", "player_key"], how="inner", suffixes=("", "_actual"))
    if merged.empty:
        print(f"No player names matched ESPN box scores for {target}.")
        return
    for market in MARKETS:
        merged[f"actual_{market}"] = merged[f"actual_{market}"]
        merged[f"{market}_error"] = merged[f"{market}_projection"] - merged[f"actual_{market}"]
        merged[f"{market}_absolute_error"] = merged[f"{market}_error"].abs()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    combined = pd.concat([old, merged], ignore_index=True)
    keys = [c for c in ["game_date", "player", "team", "opponent"] if c in combined]
    combined = combined.drop_duplicates(keys, keep="last")
    combined.to_csv(out_path, index=False)
    print(f"Settled {len(merged)} player rows for {target}; saved {len(combined)} total rows to {out_path}.")

if __name__ == "__main__":
    main()
