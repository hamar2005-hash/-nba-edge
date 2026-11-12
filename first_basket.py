"""First made field-goal scorer model using ESPN play-by-play and box scores.

This is a first-basket (first made field goal, not first free throw) market.
The prediction is a smoothed empirical ranking conditional on players appearing
on the ESPN roster; probabilities are normalized model scores, not calibrated odds.
"""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path
import argparse, re, time
import numpy as np
import pandas as pd
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NBAEdge/1.0)"}
DATA = Path("data")
GAMES = DATA / "first_basket_games.csv"
PLAYER_HISTORY = DATA / "first_basket_player_history.csv"

def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

def get_json(url, params=None):
    response = requests.get(url, params=params, headers=HEADERS, timeout=25)
    response.raise_for_status()
    return response.json()

def is_first_made_field_goal(play):
    if not play.get("scoringPlay"):
        return False
    score_value = play.get("scoreValue")
    try:
        if int(score_value) not in (2, 3):
            return False
    except (TypeError, ValueError):
        return False
    text = " ".join(str(play.get(k, "")) for k in ("text", "shortText", "type"))
    return "free throw" not in text.lower()

def first_scorer(summary):
    plays = summary.get("plays") or []
    def order(play):
        period = play.get("period", 1)
        if isinstance(period, dict):
            period = period.get("number", 1)
        sequence = play.get("sequenceNumber", 0)
        try:
            period = int(period)
        except (TypeError, ValueError):
            period = 1
        try:
            sequence = int(sequence)
        except (TypeError, ValueError):
            sequence = 0
        return period, sequence
    for play in sorted(plays, key=order):
        if not is_first_made_field_goal(play):
            continue
        participants = play.get("participants") or []
        for participant in participants:
            athlete = participant.get("athlete") or {}
            name = athlete.get("displayName") or athlete.get("fullName")
            if name:
                return name
        text = str(play.get("text") or "")
        match = re.search(r"^(.+?)\s+(?:makes|hits|dunks|lays?\s+it\s+in)", text, re.I)
        if match:
            return match.group(1).strip()
    return None

def starters_from_summary(summary, game_date, event_id):
    rows = []
    for team_block in (summary.get("boxscore") or {}).get("players", []):
        team = (team_block.get("team") or {}).get("abbreviation", "")
        for group in team_block.get("statistics", []):
            for athlete in group.get("athletes", []):
                person = athlete.get("athlete") or {}
                name = person.get("displayName") or person.get("fullName")
                if not name:
                    continue
                rows.append({
                    "game_date": game_date, "event_id": str(event_id),
                    "player": name, "player_key": norm(name), "team": team,
                    "starter": bool(athlete.get("starter")),
                    "first_basket": False,
                })
    return rows

def collect(start_date: date, end_date: date, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
    done = set(existing.get("event_id", pd.Series(dtype=str)).astype(str))
    new_rows = []
    day = start_date
    while day <= end_date:
        date_key = day.strftime("%Y%m%d")
        try:
            scoreboard = get_json(f"{BASE}/scoreboard", {"dates": date_key})
        except Exception as exc:
            print(f"Scoreboard {day}: {exc}")
            day += timedelta(days=1)
            continue
        for event in scoreboard.get("events", []):
            event_id = str(event.get("id", ""))
            if not event_id or event_id in done:
                continue
            status = ((event.get("status") or {}).get("type") or {}).get("completed", False)
            if not status:
                continue
            try:
                summary = get_json(f"{BASE}/summary", {"event": event_id})
                scorer = first_scorer(summary)
                rows = starters_from_summary(summary, day.isoformat(), event_id)
                scorer_key = norm(scorer)
                for row in rows:
                    row["first_basket"] = bool(scorer_key and row["player_key"] == scorer_key)
                if rows:
                    new_rows.extend(rows)
                    done.add(event_id)
            except Exception as exc:
                print(f"Summary {event_id}: {exc}")
            time.sleep(0.25)
        day += timedelta(days=1)
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(["event_id", "player_key", "team"], keep="last")
        combined.to_csv(out, index=False)
        print(f"Saved {len(combined)} starter-game records across {combined.event_id.nunique()} games.")
    else:
        print("No completed game starter records were collected.")

def build_history(games_path: Path, out: Path):
    games = pd.read_csv(games_path)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["starter"] = games["starter"].astype(str).str.lower().isin(["true", "1", "yes"])
    games["first_basket"] = games["first_basket"].astype(str).str.lower().isin(["true", "1", "yes"])
    starters = games[games["starter"]].copy()
    if starters.empty:
        raise SystemExit("No starter records found; cannot build first-basket history.")
    agg = starters.groupby(["player", "player_key", "team"], as_index=False).agg(
        starts=("event_id", "nunique"),
        first_baskets=("first_basket", "sum"),
        last_start=("game_date", "max"),
    )
    agg["first_basket_rate"] = (agg["first_baskets"] + 1.0) / (agg["starts"] + 12.0)
    agg.to_csv(out, index=False)
    print(f"Built first-basket history for {len(agg)} player-team combinations.")

def predict(predictions_path: Path, history_path: Path, out: Path):
    slate = pd.read_csv(predictions_path)
    hist = pd.read_csv(history_path)
    hist["player_key"] = hist["player_key"].astype(str)
    slate["player_key"] = slate["player"].map(norm)
    slate["team"] = slate["team"].astype(str)
    hist["team"] = hist["team"].astype(str)
    slate = slate.merge(hist, on=["player_key", "team"], how="left")
    slate["starts"] = pd.to_numeric(slate["starts"], errors="coerce").fillna(0)
    slate["first_baskets"] = pd.to_numeric(slate["first_baskets"], errors="coerce").fillna(0)
    slate["first_basket_rate"] = pd.to_numeric(slate["first_basket_rate"], errors="coerce").fillna(1/13)
    # Historical starts and recency provide a proxy for expected starting-lineup probability.
    slate["last_start"] = pd.to_datetime(slate["last_start"], errors="coerce")
    slate["days_since_start"] = (pd.Timestamp.utcnow().tz_localize(None) - slate["last_start"]).dt.days.clip(lower=0)
    recency_weight = np.exp(-slate["days_since_start"].fillna(365) / 45.0)
    slate["start_weight"] = np.minimum(1.0, slate["starts"] / 12.0) * recency_weight
    slate["raw_score"] = slate["start_weight"] * slate["first_basket_rate"]
    # Avoid presenting inactive/bench players as realistic first-basket candidates.
    slate.loc[slate["starts"] < 3, "raw_score"] = 0
    slate["model_confidence"] = slate.groupby("game")["raw_score"].transform(
        lambda x: (x / x.sum() * 100) if x.sum() > 0 else x * 0
    )
    result = slate[slate["model_confidence"] > 0].copy()
    result = result.sort_values(["game", "model_confidence"], ascending=[True, False])
    result["rank"] = result.groupby("game").cumcount() + 1
    result = result[result["rank"] <= 5]
    if result.empty:
        raise SystemExit("No players have enough historical starter data to make first-basket rankings.")
    cols = ["game_date", "game", "player", "team", "opponent", "rank", "model_confidence", "starts", "first_baskets", "first_basket_rate"]
    result[cols].rename(columns={"model_confidence": "estimated_chance_pct"}).to_csv(out, index=False)
    print(f"Generated first-basket rankings for {result.game.nunique()} games; saved {len(result)} candidates to {out}.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["collect", "history", "predict"], required=True)
    p.add_argument("--start", default="2024-10-22")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--games", default=str(GAMES))
    p.add_argument("--history", default=str(PLAYER_HISTORY))
    p.add_argument("--predictions", default="data/daily_predictions.csv")
    p.add_argument("--out", default="data/first_basket_predictions.csv")
    args = p.parse_args()
    if args.mode == "collect":
        collect(date.fromisoformat(args.start), date.fromisoformat(args.end), Path(args.games))
    elif args.mode == "history":
        build_history(Path(args.games), Path(args.history))
    else:
        predict(Path(args.predictions), Path(args.history), Path(args.out))

if __name__ == "__main__":
    main()
