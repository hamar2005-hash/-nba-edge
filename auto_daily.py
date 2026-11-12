"""Create today's NBA player projections from ESPN schedule/rosters and saved ML models.

No bookmaker lines or odds are requested or used. Projections are model estimates
based on the latest historical feature snapshot bundled with the trained models.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse, json, re, sys
import joblib
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NBAEdge/1.0)"}

def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

def get_json(url, params=None):
    response = requests.get(url, params=params, headers=HEADERS, timeout=25)
    response.raise_for_status()
    return response.json()

def flatten_athletes(payload):
    athletes = payload.get("athletes", [])
    out = []
    for group in athletes:
        if isinstance(group, dict) and "items" in group:
            out.extend(group.get("items", []))
        elif isinstance(group, dict) and group.get("id"):
            out.append(group)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Date YYYY-MM-DD; defaults to today's US Eastern date")
    parser.add_argument("--models", default="models")
    parser.add_argument("--out", default="data/daily_predictions.csv")
    args = parser.parse_args()

    game_date = args.date or datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    date_key = game_date.replace("-", "")
    models_dir = Path(args.models)
    feature_path = models_dir / "latest_player_features.csv"
    if not feature_path.exists():
        raise SystemExit(f"Missing {feature_path}. The trained model bundle must include latest_player_features.csv.")

    features = pd.read_csv(feature_path)
    if "PLAYER_NAME" not in features or "PLAYER_ID" not in features:
        raise SystemExit("The feature snapshot is missing PLAYER_NAME or PLAYER_ID.")
    features["player_key"] = features["PLAYER_NAME"].map(norm)
    features = features.drop_duplicates("player_key", keep="last").set_index("player_key", drop=False)

    packs = {}
    for market in ("points", "rebounds", "assists"):
        model_path = models_dir / f"{market}_model.joblib"
        if not model_path.exists():
            raise SystemExit(f"Missing trained model: {model_path}")
        packs[market] = joblib.load(model_path)

    scoreboard = get_json(f"{ESPN}/scoreboard", {"dates": date_key})
    events = scoreboard.get("events", [])
    rows = []
    skipped_teams = []
    for event in events:
        competition = (event.get("competitions") or [{}])[0]
        competitors = competition.get("competitors", [])
        if len(competitors) != 2:
            continue
        for competitor in competitors:
            team = competitor.get("team", {})
            team_id = team.get("id")
            team_abbr = team.get("abbreviation", "")
            opponent = next((c.get("team", {}).get("abbreviation", "") for c in competitors if c is not competitor), "")
            home = int(competitor.get("homeAway") == "home")
            if not team_id:
                continue
            try:
                roster_payload = get_json(f"{ESPN}/teams/{team_id}/roster", {"season": scoreboard.get("season", {}).get("year", "")})
                roster = flatten_athletes(roster_payload)
            except Exception as exc:
                skipped_teams.append(f"{team_abbr}: {exc}")
                continue

            for athlete in roster:
                name = athlete.get("fullName") or athlete.get("displayName")
                if not name:
                    continue
                status = athlete.get("status", "")
                if isinstance(status, dict):
                    status_type = status.get("type", "")
                    if isinstance(status_type, dict):
                        status_text = str(status_type.get("name", status.get("name", ""))).lower()
                    else:
                        status_text = str(status_type or status.get("name", "")).lower()
                else:
                    status_text = str(status or "").lower()
                if athlete.get("active") is False or any(term in status_text for term in ("out", "injured reserve", "suspended")):
                    continue
                key = norm(name)
                if key not in features.index:
                    continue
                base = features.loc[key]
                if isinstance(base, pd.DataFrame):
                    base = base.iloc[-1]
                record = {
                    "game_date": game_date,
                    "game": event.get("name", ""),
                    "game_time_utc": event.get("date", ""),
                    "player": name,
                    "team": team_abbr,
                    "opponent": opponent,
                    "home": home,
                    "player_status": status_text or "Not supplied",
                }
                for market, pack in packs.items():
                    xrow = {}
                    for column in pack["feature_cols"]:
                        if column == "HOME":
                            value = home
                        elif column == "REST_DAYS":
                            last_date = pd.to_datetime(base.get("GAME_DATE"), errors="coerce")
                            value = max(0, min(14, (pd.Timestamp(game_date) - last_date).days)) if pd.notna(last_date) else base.get(column, 0)
                        elif column == "GAMES_PLAYED_BEFORE":
                            value = base.get(column, 0)
                        else:
                            value = base.get(column, 0)
                        xrow[column] = value
                    projection = float(pack["model"].predict(pd.DataFrame([xrow], columns=pack["feature_cols"]))[0])
                    record[f"{market}_projection"] = round(max(0.0, projection), 1)
                record["minutes_baseline"] = round(float(base.get("MIN_L5", base.get("MIN_L10", 0)) or 0), 1)
                rows.append(record)

    if not rows:
        details = "; ".join(skipped_teams)
        raise SystemExit(f"No player projections generated for {game_date}. ESPN returned {len(events)} games. Check roster feed and feature snapshot. {details}")
    result = pd.DataFrame(rows).drop_duplicates(["game_date", "player", "team", "opponent"])
    result = result.sort_values(["game_time_utc", "game", "team", "player"])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Generated {len(result)} player projections across {len(events)} scheduled games for {game_date}.")
    print(f"Saved to {output}")
    if skipped_teams:
        print("Roster feed issues: " + "; ".join(skipped_teams))

if __name__ == "__main__":
    main()
