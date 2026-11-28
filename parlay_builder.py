"""Build conservative three-leg NBA prop parlays from user-entered bookmaker odds."""
from __future__ import annotations

from itertools import combinations
import math
import re

import pandas as pd


def _probability(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip().replace("%", "")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1:
        number /= 100.0
    return number if 0 < number < 1 else None


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def _game_key(row):
    team = _text(row.get("Team", "")).upper()
    opponent = _text(row.get("Opponent", "")).upper()
    game_date = _text(row.get("Game date", ""))
    game_time = _text(row.get("Game time UTC", ""))
    if team and opponent:
        matchup = "|".join(sorted((team, opponent)))
        return f"{game_date}|{matchup}"
    return f"{game_date}|{game_time}|{team}|{opponent}"


def build_three_leg_parlays(candidates, max_parlays=3, min_edge_pct=3.0):
    """Return up to three positive-value parlays, with one leg per game.

    Each leg must have a real decimal price supplied by the user and positive
    standalone expected value. Joint probability is the product of leg
    probabilities, so games are deliberately kept separate to reduce (not
    eliminate) dependence. This is a screening estimate, not a calibrated
    betting probability.
    """
    if candidates is None or candidates.empty:
        return []

    legs = []
    seen = set()
    for _, row in candidates.iterrows():
        odds = pd.to_numeric(row.get("Sky Bet odds"), errors="coerce")
        if pd.isna(odds) or float(odds) <= 1.01:
            continue
        side = str(row.get("Best side", "")).strip()
        probability_column = "Over probability" if side.lower() == "over" else "Under probability"
        probability = _probability(row.get(probability_column))
        if probability is None:
            probability = _probability(row.get("Estimated confidence"))
        if probability is None or probability <= 0.5:
            continue
        odds = float(odds)
        if probability * odds <= 1.0:
            continue
        player = str(row.get("Player", "")).strip()
        market = str(row.get("Market", "")).strip()
        line = pd.to_numeric(row.get("Prop line"), errors="coerce")
        if not player or not market or pd.isna(line):
            continue
        player_key = re.sub(r"[^a-z0-9]", "", player.lower())
        identity = (player_key, market.lower(), side.lower(), float(line))
        if identity in seen:
            continue
        seen.add(identity)
        legs.append({
            "Player": player,
            "Team": str(row.get("Team", "")),
            "Opponent": str(row.get("Opponent", "")),
            "Game date": str(row.get("Game date", "")),
            "Game time UTC": str(row.get("Game time UTC", "")),
            "Market": market,
            "Best side": side,
            "Prop line": float(line),
            "Sky Bet odds": odds,
            "probability": probability,
            "player_key": player_key,
            "game_key": _game_key(row),
        })

    opportunities = []
    for combo in combinations(legs, 3):
        if len({leg["game_key"] for leg in combo}) != 3:
            continue
        if len({leg["player_key"] for leg in combo}) != 3:
            continue
        model_probability = math.prod(leg["probability"] for leg in combo)
        combined_odds = math.prod(leg["Sky Bet odds"] for leg in combo)
        implied_probability = 1.0 / combined_odds
        edge_pct = (model_probability - implied_probability) * 100.0
        expected_value_pct = (model_probability * combined_odds - 1.0) * 100.0
        if edge_pct < float(min_edge_pct) or expected_value_pct <= 0:
            continue
        opportunities.append({
            "legs": list(combo),
            "combined_odds": combined_odds,
            "model_probability_pct": model_probability * 100.0,
            "implied_probability_pct": implied_probability * 100.0,
            "edge_pct": edge_pct,
            "expected_value_pct": expected_value_pct,
            "_players": {leg["player_key"] for leg in combo},
        })

    opportunities.sort(
        key=lambda item: (item["expected_value_pct"], item["edge_pct"]),
        reverse=True,
    )
    selected = []
    for opportunity in opportunities:
        # Keep the daily shortlist meaningfully varied: no more than one shared
        # player across any two published parlay suggestions.
        if any(len(opportunity["_players"] & item["_players"]) > 1 for item in selected):
            continue
        selected.append(opportunity)
        if len(selected) >= max(1, int(max_parlays)):
            break
    for item in selected:
        item.pop("_players", None)
    return selected
