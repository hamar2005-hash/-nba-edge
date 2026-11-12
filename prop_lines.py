"""Read current NBA player prop lines from Stat Pick's public API.

The slate endpoint provides each listed player's headline market and line;
the player index provides the player's other covered markets. Prices and
bookmaker names are deliberately discarded.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import requests
import pandas as pd

BASE = "https://api.statpick.ai"
HEADERS = {"User-Agent": "NBAEdge/1.0 (player prop line comparison)", "Accept": "application/json"}
ALIASES = {
    "points": {"points", "point", "pts", "playerpoints", "pointsprop", "playerpointsprop"},
    "rebounds": {"rebounds", "rebound", "reb", "playerrebounds", "reboundsprop", "playerreboundsprop"},
    "assists": {"assists", "assist", "ast", "playerassists", "assistsprop", "playerassistsprop"},
}
MARKET_FIELDS = ("statSlug", "stat_slug", "marketSlug", "market_slug", "marketName", "market_name", "statName", "stat_name", "market", "stat", "name", "slug")
LINE_FIELDS = ("currentLine", "current_line", "propLine", "prop_line", "marketLine", "market_line", "line", "lineValue", "line_value", "displayLine", "display_line")

def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

def slugify(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")

def get_json(path):
    response = requests.get(BASE + path, headers=HEADERS, timeout=(5, 15))
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(payload.get("message", "Stat Pick API returned an error"))
    return payload.get("data", payload) if isinstance(payload, dict) else payload

def as_number(value):
    if isinstance(value, dict):
        for key in ("value", "line", "current", "number"):
            if key in value:
                return as_number(value[key])
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None

def market_name(value):
    raw = norm(value)
    if raw in ALIASES["points"] or raw.endswith("points"):
        return "points"
    if raw in ALIASES["rebounds"] or raw.endswith("rebounds"):
        return "rebounds"
    if raw in ALIASES["assists"] or raw.endswith("assists"):
        return "assists"
    return None

def walk(value, parent_key=""):
    if isinstance(value, dict):
        yield value, parent_key
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from walk(child, str(key))
    elif isinstance(value, list):
        for child in value:
            yield from walk(child, parent_key)

def discover_players(slate, wanted_keys=None):
    found = {}
    for item, _ in walk(slate):
        name = item.get("playerName") or item.get("fullName") or item.get("displayName") or item.get("player_name")
        slug = item.get("playerSlug") or item.get("player_slug") or item.get("slug")
        if not name:
            continue
        name = str(name)
        key = norm(name)
        if wanted_keys and key not in wanted_keys:
            continue
        slug = str(slug or slugify(name))
        team = item.get("teamAbbreviation") or item.get("team_abbreviation") or item.get("team") or ""
        found[(key, slug)] = (name, slug, str(team))
    return list(found.values())

def extract_rows(payload, player, team, slug):
    rows, seen = [], set()
    for item, parent_key in walk(payload):
        market = None
        for field in MARKET_FIELDS:
            value = item.get(field)
            if isinstance(value, dict):
                value = value.get("slug") or value.get("name") or value.get("label")
            if value:
                market = market_name(value)
                if market:
                    break
        if not market:
            market = market_name(parent_key)
        if not market:
            continue
        line = None
        for field in LINE_FIELDS:
            if field in item:
                line = as_number(item.get(field))
                if line is not None:
                    break
        if line is None:
            for key in ("headlineMarket", "headline_market", "current", "prop", "marketData", "market_data"):
                child = item.get(key)
                if isinstance(child, dict):
                    for field in LINE_FIELDS:
                        if field in child:
                            line = as_number(child[field])
                            if line is not None:
                                break
                if line is not None:
                    break
        if line is None:
            continue
        key = (market, line)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "player": player, "player_key": norm(player), "player_slug": slug,
            "team": team, "opponent": item.get("opponentAbbreviation") or item.get("opponent") or "",
            "market": market, "line": line, "line_source": "Stat Pick",
            "line_updated_at": item.get("updatedAt") or item.get("updated_at") or "",
        })
    return rows

def fetch_one(candidate):
    name, slug, team = candidate
    return extract_rows(get_json(f"/api/nba/prop-pages/player/{slug}"), name, team, slug)

def fetch_current_lines(player_names=None):
    slate = get_json("/api/nba/prop-pages/slate")
    wanted = {norm(name) for name in player_names} if player_names else None
    candidates = discover_players(slate, wanted)
    if not candidates:
        raise RuntimeError("Stat Pick's slate contained no players matching today's NBA Edge projections.")
    slate_rows = []
    for name, slug, team in candidates:
        slate_rows.extend(extract_rows(slate, name, team, slug))
    detail_rows, errors = [], []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_one, candidate): candidate[0] for candidate in candidates}
        for future in as_completed(futures):
            try:
                detail_rows.extend(future.result())
            except Exception as exc:
                errors.append(f"{futures[future]}: {exc}")
    all_rows = slate_rows + detail_rows
    if not all_rows:
        raise RuntimeError("Stat Pick returned matching players but no readable PTS/REB/AST lines. Its response fields may have changed or today's markets may not be posted. " + "; ".join(errors[:3]))
    return pd.DataFrame(all_rows).drop_duplicates(["player_key", "market", "line"])
