"""Fast read-only NBA prop line fetcher using Stat Pick's public API.

The slate endpoint discovers players; one concurrent player-index request per
player returns all covered markets and their current lines. Odds/prices are
never retained, returned, or displayed.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import requests
import pandas as pd

BASE = "https://api.statpick.ai"
HEADERS = {"User-Agent": "NBAEdge/1.0 (player prop line comparison)", "Accept": "application/json"}
MARKET_ALIASES = {
    "points": {"points", "point", "pts", "player_points", "points_scored"},
    "rebounds": {"rebounds", "rebound", "reb", "player_rebounds"},
    "assists": {"assists", "assist", "ast", "player_assists"},
}

def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

def slugify(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")

def get_json(path):
    response = requests.get(BASE + path, headers=HEADERS, timeout=(5, 12))
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(payload.get("message", "Stat Pick API returned an error"))
    return payload.get("data", payload) if isinstance(payload, dict) else payload

def walk(value, parent_market=""):
    if isinstance(value, dict):
        market = (value.get("statSlug") or value.get("marketSlug") or
                  value.get("stat_slug") or value.get("market") or
                  value.get("stat") or parent_market)
        yield value, str(market or "")
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from walk(child, str(key) if key else str(market or ""))
    elif isinstance(value, list):
        for child in value:
            yield from walk(child, parent_market)

def number(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None

def market_name(hint):
    raw = norm(hint)
    for name, aliases in MARKET_ALIASES.items():
        if raw in {norm(alias) for alias in aliases}:
            return name
    return None

def discover_players(slate, wanted_keys=None):
    found = {}
    for item, _ in walk(slate):
        name = (item.get("playerName") or item.get("fullName") or
                item.get("displayName") or item.get("player_name"))
        slug = item.get("playerSlug") or item.get("player_slug") or item.get("slug")
        if not name:
            continue
        name = str(name)
        key = norm(name)
        if wanted_keys and key not in wanted_keys:
            continue
        slug = str(slug or slugify(name))
        team = (item.get("teamAbbreviation") or item.get("team") or
                item.get("team_abbreviation") or "")
        found[(key, slug)] = (name, slug, str(team))
    return list(found.values())

def parse_player_index(payload, player, team, slug):
    rows = []
    seen = set()
    for item, hint in walk(payload):
        # Stat Pick documents statSlug on each market record, with line on the same record.
        market = market_name(item.get("statSlug") or item.get("stat_slug") or
                             item.get("marketSlug") or item.get("market") or
                             item.get("stat") or hint)
        if not market:
            continue
        line = None
        for field in ("line", "propLine", "currentLine", "current_line", "marketLine"):
            if field in item:
                line = number(item.get(field))
                if line is not None:
                    break
        if line is None:
            continue
        key = (market, line)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "player": player,
            "player_key": norm(player),
            "player_slug": slug,
            "team": team,
            "opponent": item.get("opponentAbbreviation") or item.get("opponent") or "",
            "market": market,
            "line": line,
            "line_source": "Stat Pick",
            "line_updated_at": item.get("updatedAt") or item.get("updated_at") or "",
        })
    return rows

def fetch_one(candidate):
    name, slug, team = candidate
    payload = get_json(f"/api/nba/prop-pages/player/{slug}")
    return parse_player_index(payload, name, team, slug)

def fetch_current_lines(player_names=None):
    # The slate is a compact index; fetch only players present in today's NBA Edge projections.
    slate = get_json("/api/nba/prop-pages/slate")
    wanted = {norm(name) for name in player_names} if player_names else None
    candidates = discover_players(slate, wanted)
    if not candidates:
        raise RuntimeError("Stat Pick's current slate did not contain any matching players.")

    lines, errors = [], []
    # Parallel player-index requests avoid the old sequential 2–3 requests per player.
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(fetch_one, candidate): candidate[0] for candidate in candidates}
        for future in as_completed(futures):
            try:
                lines.extend(future.result())
            except Exception as exc:
                errors.append(f"{futures[future]}: {exc}")

    if not lines:
        raise RuntimeError(
            "No parseable points/rebounds/assists lines were returned for today's players. "
            "The slate may not be posted yet or Stat Pick's response format may have changed. "
            + "; ".join(errors[:3])
        )
    frame = pd.DataFrame(lines).drop_duplicates(["player_key", "market", "line"])
    return frame
