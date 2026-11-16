"""Build NBA Edge's canonical player-game log file from a stable Hugging Face archive.

Historical seasons (2021-22 through 2025-26) are downloaded as small, versioned
Parquet files. The current season is optionally supplemented from one NBA Stats
request; failure there never blocks the historical backtest.

The output deliberately matches the NBA Stats column names consumed by
train_models.prepare(), so the model feature pipeline remains unchanged.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import argparse
import time

import pandas as pd
import requests

DATASET = "Sangameshsb/nba-game-logs"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/game_logs"
REQUIRED_SOURCE = {
    "game_id", "game_date", "season", "player_id", "player_name",
    "team", "opponent", "home", "minutes", "pts", "reb", "ast",
}
STAT_MAP = {
    "minutes": "MIN", "pts": "PTS", "reb": "REB", "ast": "AST",
    "fga": "FGA", "fg3a": "FG3A", "fta": "FTA", "tov": "TOV",
    "stl": "STL", "blk": "BLK", "fgm": "FGM", "fg3m": "FG3M",
    "ftm": "FTM", "oreb": "OREB", "dreb": "DREB", "pf": "PF",
    "plus_minus": "PLUS_MINUS",
}


def normalise_logs(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert archive or NBA Stats rows to train_models.prepare() schema."""
    if frame.empty:
        return pd.DataFrame()
    lower = {str(c).lower(): c for c in frame.columns}
    if REQUIRED_SOURCE.issubset(lower):
        src = frame.rename(columns={v: k for k, v in lower.items()}).copy()
        out = pd.DataFrame({
            "GAME_ID": src["game_id"].astype(str),
            "GAME_DATE": pd.to_datetime(src["game_date"], errors="coerce"),
            "PLAYER_ID": pd.to_numeric(src["player_id"], errors="coerce"),
            "PLAYER_NAME": src["player_name"].astype(str),
            "TEAM_ABBREVIATION": src["team"].astype(str),
            "SEASON": src["season"].astype(str),
            "MATCHUP": [
                f"{team} {'vs.' if bool(home) else '@'} {opp}"
                for team, opp, home in zip(src["team"], src["opponent"], src["home"])
            ],
        })
        for source, target in STAT_MAP.items():
            out[target] = pd.to_numeric(src[source], errors="coerce") if source in src else pd.NA
        out["DATA_SOURCE"] = "huggingface_archive"
    else:
        # NBA Stats already uses the canonical uppercase schema.
        out = frame.copy()
        if "GAME_ID" in out:
            out["GAME_ID"] = out["GAME_ID"].astype(str)
        if "GAME_DATE" in out:
            out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")
        if "DATA_SOURCE" not in out:
            out["DATA_SOURCE"] = "nba_stats_current_season"
        for col in ["PLAYER_ID", "MIN", "PTS", "REB", "AST", *STAT_MAP.values()]:
            if col in out:
                out[col] = pd.to_numeric(out[col], errors="coerce")
    needed = {"GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "MATCHUP", "MIN", "PTS", "REB", "AST"}
    missing = needed - set(out.columns)
    if missing:
        raise ValueError(f"Downloaded player logs are missing canonical columns: {sorted(missing)}")
    out = out.dropna(subset=["GAME_DATE", "PLAYER_ID", "PTS", "MIN"]).copy()
    out = out[out["MIN"] > 0]
    return out


def download_archive_season(season: str, cache_dir: Path, session: requests.Session) -> pd.DataFrame:
    cache = cache_dir / f"player_logs_{season}.csv"
    if cache.exists():
        cached = pd.read_csv(cache, low_memory=False)
        if not cached.empty:
            print(f"Cached {season}: {len(cached):,} player-game rows")
            return cached

    url = f"{BASE_URL}/game_logs_{season}.parquet"
    last_error = None
    for attempt in range(3):
        try:
            response = session.get(url, timeout=(15, 90))
            response.raise_for_status()
            frame = pd.read_parquet(BytesIO(response.content), engine="pyarrow")
            lower = {str(c).lower() for c in frame.columns}
            missing = REQUIRED_SOURCE - lower
            if missing:
                raise ValueError(f"{season} archive missing columns: {sorted(missing)}")
            result = normalise_logs(frame)
            if result.empty:
                raise ValueError(f"{season} archive returned no usable rows")
            result.to_csv(cache, index=False)
            print(f"Downloaded {season}: {len(result):,} player-game rows")
            return result
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                wait = 3 * (attempt + 1)
                print(f"{season} download attempt {attempt + 1} failed: {exc}; retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"Could not download required historical season {season}: {last_error}")


def fetch_current_season(season: str) -> pd.DataFrame:
    """Best-effort single request for current-season-to-date rows."""
    try:
        from nba_api.stats.endpoints import leaguegamelog
        endpoint = leaguegamelog.LeagueGameLog(
            player_or_team_abbreviation="P",
            season=season,
            season_type_all_star="Regular Season",
            league_id="00",
            timeout=25,
        )
        frame = endpoint.get_data_frames()[0]
        result = normalise_logs(frame)
        print(f"Current season {season}: {len(result):,} rows from NBA Stats")
        return result
    except Exception as exc:
        print(f"Current-season supplement unavailable ({exc}). Continuing with complete archived seasons.")
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", default="2021-22")
    parser.add_argument("--end-season", default="2025-26")
    parser.add_argument("--current-season", default="2026-27")
    parser.add_argument("--out", default="data/player_game_logs.csv")
    parser.add_argument("--skip-current-season", action="store_true")
    args = parser.parse_args()

    start_year = int(args.start_season[:4])
    end_year = int(args.end_season[:4])
    if end_year < start_year:
        raise SystemExit("--end-season must not be earlier than --start-season")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = out.parent / "season_cache"
    cache_dir.mkdir(exist_ok=True)

    frames = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "NBA-Edge historical backtest/1.0"})
        for year in range(start_year, end_year + 1):
            season = f"{year}-{str(year + 1)[-2:]}"
            frames.append(download_archive_season(season, cache_dir, session))

    if not args.skip_current_season:
        current = fetch_current_season(args.current_season)
        if not current.empty:
            frames.append(current)

    data = pd.concat(frames, ignore_index=True, sort=False)
    data["GAME_ID"] = data["GAME_ID"].astype(str)
    data = data.drop_duplicates(["PLAYER_ID", "GAME_ID"], keep="last")
    data = data.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    # Guard against a partial archive silently producing a misleading backtest.
    archived = data[data.get("DATA_SOURCE", pd.Series(index=data.index, dtype=str)) == "huggingface_archive"]
    expected = end_year - start_year + 1
    observed_seasons = archived["SEASON"].nunique() if "SEASON" in archived else 0
    if archived.empty or observed_seasons < expected:
        raise SystemExit(f"Historical archive incomplete: expected {expected} seasons, got {observed_seasons}")

    data.to_csv(out, index=False)
    print(f"Combined {len(data):,} unique player-game rows -> {out}")
    print(f"Date range: {data.GAME_DATE.min().date()} to {data.GAME_DATE.max().date()}")
    print(f"Historical archive rows: {len(archived):,}")
    print("NBA Stats was not used for the historical backfill.")


if __name__ == "__main__":
    main()
