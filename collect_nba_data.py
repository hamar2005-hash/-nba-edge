"""Download NBA player game logs in monthly chunks with resumable local caching.

NBA Stats can throttle or block cloud environments. Successful months are cached,
so rerunning the script only retries missing months. This does not bypass provider
access controls; if NBA Stats remains unreachable, use a permitted data export.
"""
from pathlib import Path
import argparse, calendar, time
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start-season', default='2021-22')
    p.add_argument('--end-season', default='2026-27')
    p.add_argument('--out', default='data/player_game_logs.csv')
    p.add_argument('--sleep', type=float, default=2.0)
    p.add_argument('--retries', type=int, default=2)
    args = p.parse_args()
    try:
        from nba_api.stats.endpoints import leaguegamelog
    except ImportError as e:
        raise SystemExit('Install dependencies first: pip install -r requirements.txt') from e

    start_year, end_year = int(args.start_season[:4]), int(args.end_season[:4])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = out.parent / 'monthly_cache'; cache_dir.mkdir(exist_ok=True)
    frames, failures = [], []

    for season_year in range(start_year, end_year + 1):
        season = f'{season_year}-{str(season_year + 1)[-2:]}'
        # Regular season months: Oct-Dec in season start year, Jan-Jun next year.
        periods = [(season_year, m) for m in range(10, 13)] + [(season_year + 1, m) for m in range(1, 7)]
        for year, month in periods:
            first = f'{year}-{month:02d}-01'
            last = f'{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}'
            cache = cache_dir / f'player_logs_{year}_{month:02d}.csv'
            if cache.exists():
                try:
                    cached = pd.read_csv(cache)
                    if not cached.empty:
                        frames.append(cached); print(f'Cached {first}: {len(cached):,} rows'); continue
                except Exception:
                    cache.unlink(missing_ok=True)
            print(f'Downloading {season}: {first} to {last}')
            error = None
            for attempt in range(args.retries + 1):
                try:
                    endpoint = leaguegamelog.LeagueGameLog(
                        player_or_team_abbreviation='P', season=season,
                        season_type_all_star='Regular Season', league_id='00',
                        date_from_nullable=first, date_to_nullable=last,
                        timeout=35, headers=HEADERS)
                    chunk = endpoint.get_data_frames()[0]
                    if not chunk.empty:
                        chunk.to_csv(cache, index=False); frames.append(chunk)
                        print(f'  saved {len(chunk):,} rows')
                    else:
                        print('  no games returned for this month')
                    error = None; break
                except Exception as exc:
                    error = exc
                    if attempt < args.retries:
                        wait = 5 * (attempt + 1)
                        print(f'  attempt {attempt + 1} failed: {exc}; retrying in {wait}s')
                        time.sleep(wait)
            if error:
                failures.append(f'{first}: {error}')
                print(f'  skipped {first}; it can be retried on the next run')
            time.sleep(args.sleep)

    if not frames:
        raise SystemExit('No data downloaded. NBA Stats is unreachable from this session; do not keep retrying repeatedly.')
    data = pd.concat(frames, ignore_index=True)
    if {'PLAYER_ID', 'GAME_ID'}.issubset(data.columns):
        data = data.drop_duplicates(['PLAYER_ID', 'GAME_ID'])
    data.to_csv(out, index=False)
    print(f'Combined {len(data):,} player-game rows -> {out}')
    if failures:
        failure_path = out.parent / 'download_failures.txt'
        failure_path.write_text('\n'.join(failures))
        print(f'{len(failures)} month(s) failed; successful months remain cached in {cache_dir}.')
    else:
        print('All requested months downloaded successfully.')

if __name__ == '__main__':
    main()
