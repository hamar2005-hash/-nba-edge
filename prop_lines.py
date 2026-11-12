"""Fetch current NBA player prop lines from Stat Pick's public read-only API.

Only player, team, market and line are retained. Any prices/odds in source
responses are deliberately discarded and never displayed or saved.
"""
from datetime import datetime
from zoneinfo import ZoneInfo
import re
import requests

BASE = "https://api.statpick.ai"
HEADERS = {"User-Agent": "NBAEdge/1.0 (player projection comparison)", "Accept": "application/json"}
MARKET_ALIASES = {
    "points": {"points", "point", "pts", "player_points", "points_scored"},
    "rebounds": {"rebounds", "rebound", "reb", "reb", "player_rebounds"},
    "assists": {"assists", "assist", "ast", "player_assists"},
}

def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

def slugify(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")

def get_json(path):
    response = requests.get(BASE + path, headers=HEADERS, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(payload.get("message", "Stat Pick API returned an error"))
    return payload.get("data", payload) if isinstance(payload, dict) else payload

def walk(value, parent_market=""):
    if isinstance(value, dict):
        market = value.get("statSlug") or value.get("marketSlug") or value.get("stat_slug") or value.get("market") or value.get("stat") or parent_market
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

def parse_market_rows(payload, player, team, player_slug):
    rows = []
    seen = set()
    for item, market_hint in walk(payload):
        raw_market = norm(market_hint)
        market = next((name for name, aliases in MARKET_ALIASES.items() if raw_market in {norm(x) for x in aliases}), None)
        if not market:
            continue
        line = None
        for key in ("line", "propLine", "currentLine", "current_line", "marketLine", "value"):
            if key in item:
                line = number(item.get(key))
                if line is not None:
                    break
        if line is None:
            continue
        opponent = item.get("opponent") or item.get("opponentAbbreviation") or ""
        key = (player, market, line)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "player": player,
            "player_key": norm(player),
            "player_slug": player_slug,
            "team": team,
            "opponent": opponent,
            "market": market,
            "line": line,
            "line_source": "Stat Pick",
            "line_updated_at": item.get("updatedAt") or item.get("updated_at") or "",
        })
    return rows

def fetch_current_lines():
    slate = get_json("/api/nba/prop-pages/slate")
    candidates = []
    for item, _ in walk(slate):
        name = item.get("playerName") or item.get("fullName") or item.get("displayName") or item.get("player_name")
        slug = item.get("playerSlug") or item.get("player_slug") or item.get("slug")
        if name and not slug:
            slug = slugify(name)
        if not name or not slug:
            continue
        team = item.get("teamAbbreviation") or item.get("team") or item.get("team_abbreviation") or ""
        candidates.append((str(name), str(slug), str(team)))

    unique = {}
    for name, slug, team in candidates:
        unique[(norm(name), slug)] = (name, slug, team)

    lines = []
    errors = []
    for name, slug, team in unique.values():
        try:
            detail = get_json(f"/api/nba/prop-pages/player/{slug}")
            found = parse_market_rows(detail, name, team, slug)
            if not found:
                # Some player index payloads expose a market slug but omit line values.
                for item, _ in walk(detail):
                    stat_slug = item.get("statSlug") or item.get("stat_slug") or item.get("slug")
                    if not stat_slug:
                        continue
                    stat_slug = str(stat_slug)
                    if norm(stat_slug) not in {norm(x) for aliases in MARKET_ALIASES.values() for x in aliases}:
                        continue
                    try:
                        stat_data = get_json(f"/api/nba/prop-pages/player/{slug}/stat/{stat_slug}")
                        found.extend(parse_market_rows(stat_data, name, team, slug))
                    except Exception:
                        continue
            lines.extend(found)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if not lines:
        raise RuntimeError("Stat Pick returned no parseable NBA points/rebounds/assists lines. API schema or today's coverage may have changed. " + "; ".join(errors[:3]))
    import pandas as pd
    frame = pd.DataFrame(lines).drop_duplicates(["player_key", "market", "line"])
    return frame
