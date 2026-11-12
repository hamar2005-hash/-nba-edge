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
MARKET_FIELDS = ("statSlug", "stat_slug", "marketSlug", "market_slug", "marketName", "market_name", "statName", "stat_name", "marketType", "market_type", "statType", "stat_type", "market", "stat", "name", "label", "slug")
LINE_FIELDS = ("currentLine", "current_line", "propLine", "prop_line", "marketLine", "market_line", "line", "lineValue", "line_value", "displayLine", "display_line", "lineNumber", "line_number")

def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

def slugify(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")

def get_json(path):
    response = requests.get(BASE + path, headers=HEADERS, timeout=(2, 4))
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
    name_fields = ("playerName", "fullName", "displayName", "player_name", "name")
    slug_fields = ("playerSlug", "player_slug", "slug", "playerId", "player_id")
    for item, _ in walk(slate):
        player_obj = item.get("player") if isinstance(item.get("player"), dict) else {}
        name = next((item.get(k) for k in name_fields if isinstance(item.get(k), str) and item.get(k).strip()), None)
        name = name or next((player_obj.get(k) for k in name_fields if isinstance(player_obj.get(k), str) and player_obj.get(k).strip()), None)
        if not name and isinstance(item.get("player"), str):
            name = item.get("player")
        if not name:
            continue
        name = str(name).strip()
        key = norm(name)
        if wanted_keys and key not in wanted_keys:
            continue
        slug = next((item.get(k) for k in slug_fields if item.get(k) not in (None, "")), None)
        slug = slug or next((player_obj.get(k) for k in slug_fields if player_obj.get(k) not in (None, "")), None)
        slug = str(slug or slugify(name))
        team = item.get("teamAbbreviation") or item.get("team_abbreviation") or item.get("teamAbbr") or item.get("team") or player_obj.get("teamAbbreviation") or player_obj.get("teamAbbr") or player_obj.get("team") or ""
        found[(key, slug)] = (name, slug, str(team))
    return list(found.values())

def extract_rows(payload, player, team, slug):
    rows, seen = [], set()
    for item, parent_key in walk(payload):
        market = None
        for field in MARKET_FIELDS:
            value = item.get(field)
            if isinstance(value, dict):
                value = value.get("slug") or value.get("name") or value.get("label") or value.get("statSlug") or value.get("stat_slug")
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
    """Fetch the slate once and extract its posted headline lines.

    Deliberately do not fan out to one player endpoint per candidate: a full
    NBA slate can otherwise keep the Streamlit page in a running state for
    a long time. The public slate endpoint already includes each player's
    headline market and current line.
    """
    slate = get_json("/api/nba/prop-pages/slate")
    wanted = {norm(name) for name in player_names} if player_names else None

    # Discover without filtering first. Stat Pick may include the player name
    # in a nested player object, and this lets us compare after parsing.
    candidates = discover_players(slate)
    rows = []
    for name, slug, team in candidates:
        rows.extend(extract_rows(slate, name, team, slug))

    # If the slate shape is unusual, still try parsing the full document once
    # under a neutral label; this captures payloads with flat market records.
    if not rows:
        rows = extract_rows(slate, "", "", "")

    if wanted:
        rows = [row for row in rows if row.get("player_key") in wanted]

    if not rows:
        top_keys = ", ".join(sorted(slate.keys())[:30]) if isinstance(slate, dict) else type(slate).__name__
        raise RuntimeError(
            "Stat Pick responded, but NBA Edge could not identify any matching PTS/REB/AST prop lines "
            f"from the slate response. Top-level response keys: {top_keys}. "
            "No player-detail requests were made, so this check should finish quickly."
        )

    result = pd.DataFrame(rows)
    return result.drop_duplicates(["player_key", "market", "line"])
