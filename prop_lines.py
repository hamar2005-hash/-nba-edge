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

def iter_player_records(slate):
    """Yield each actual slate player record exactly once.

    Team objects also have names/slugs, so require player-specific fields to
    avoid accidentally treating team metadata as player records.
    """
    name_fields = ("playerName", "fullName", "displayName", "player_name")
    slug_fields = ("playerSlug", "player_slug", "slug", "playerId", "player_id")
    for item, _ in walk(slate):
        player_obj = item.get("player") if isinstance(item.get("player"), dict) else {}
        name = next((item.get(k) for k in name_fields if isinstance(item.get(k), str) and item.get(k).strip()), None)
        name = name or next((player_obj.get(k) for k in name_fields if isinstance(player_obj.get(k), str) and player_obj.get(k).strip()), None)
        if not name and isinstance(item.get("player"), str):
            name = item.get("player")
        if not name:
            continue
        # Stat Pick's slate exposes playerSlug and headline/statCount; this
        # guard excludes nested team objects and arbitrary named metadata.
        if not any(key in item for key in (*slug_fields, "headline", "statCount")):
            continue
        name = str(name).strip()
        slug = next((item.get(k) for k in slug_fields if item.get(k) not in (None, "")), None)
        slug = slug or next((player_obj.get(k) for k in slug_fields if player_obj.get(k) not in (None, "")), None)
        slug = str(slug or slugify(name))
        team = (
            item.get("teamAbbreviation") or item.get("team_abbreviation")
            or item.get("teamAbbr") or item.get("team")
            or player_obj.get("teamAbbreviation") or player_obj.get("teamAbbr")
            or player_obj.get("team") or ""
        )
        yield name, slug, str(team), item


def discover_players(slate, wanted_keys=None):
    players = []
    seen = set()
    for name, slug, team, _record in iter_player_records(slate):
        key = norm(name)
        if wanted_keys and key not in wanted_keys:
            continue
        identity = (key, slug)
        if identity not in seen:
            seen.add(identity)
            players.append((name, slug, team))
    return players

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

def parse_slate(slate, player_names=None):
    """Parse player lines from each player's own record, never the whole slate."""
    wanted = {norm(name) for name in player_names} if player_names else None
    rows = []
    for name, slug, team, player_record in iter_player_records(slate):
        if wanted and norm(name) not in wanted:
            continue
        rows.extend(extract_rows(player_record, name, team, slug))
    if not rows:
        return pd.DataFrame(columns=[
            "player", "player_key", "player_slug", "team", "opponent",
            "market", "line", "line_source", "line_updated_at",
        ])
    return pd.DataFrame(rows).drop_duplicates(["player_key", "market", "line"])


def fetch_current_lines(player_names=None):
    """Fetch current lines for today's projected players, with safe fallbacks.

    The slate endpoint only supplies a headline market per player. The player
    detail endpoint supplies all supported markets. Keep the number of calls
    bounded and parallel, and retain headline lines for any failed detail call.
    """
    slate = get_json("/api/nba/prop-pages/slate")
    wanted = {norm(name) for name in player_names} if player_names else None
    candidates = discover_players(slate, wanted)
    if not candidates:
        raise RuntimeError("No Stat Pick slate players matched today's NBA Edge projections.")

    headline = parse_slate(slate, player_names)
    detailed_rows = []
    failed = []
    # The current Stat Pick slate is small; cap requests so a large projection
    # file cannot create a request storm or hold Streamlit indefinitely.
    candidates = candidates[:60]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_one, candidate): candidate[0] for candidate in candidates}
        for future in as_completed(futures):
            try:
                detailed_rows.extend(future.result())
            except Exception as exc:
                failed.append((futures[future], str(exc)[:160]))

    if detailed_rows:
        details = pd.DataFrame(detailed_rows)
        details = details.drop_duplicates(["player_key", "market", "line"])
        # Use all-market detail results for players where endpoint succeeded;
        # retain the slate headline only for candidates with detail failures.
        successful = set(details["player_key"].dropna().astype(str))
        if not headline.empty and failed:
            failed_keys = {norm(name) for name, _ in failed}
            fallback = headline[headline["player_key"].isin(failed_keys)]
            result = pd.concat([details, fallback], ignore_index=True)
        else:
            result = details
    else:
        result = headline

    if result.empty:
        summary = "; ".join(f"{name}: {error}" for name, error in failed[:3])
        raise RuntimeError(
            "Stat Pick responded but no current PTS/REB/AST lines could be extracted. "
            + summary
        )
    result = result.drop_duplicates(["player_key", "market", "line"])
    result.attrs["detail_players_loaded"] = int(result["player_key"].nunique())
    result.attrs["detail_failures"] = len(failed)
    return result
