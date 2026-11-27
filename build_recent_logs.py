"""Create a compact recent-game snapshot for player prop hit-rate displays."""
from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/player_game_logs.csv")
    parser.add_argument("--out", default="data/recent_player_game_logs.csv")
    args = parser.parse_args()
    source = Path(args.input)
    if not source.exists():
        raise SystemExit(f"Missing source game logs: {source}")
    logs = pd.read_csv(source, low_memory=False, parse_dates=["GAME_DATE"])
    required = {"PLAYER_ID", "PLAYER_NAME", "GAME_ID", "GAME_DATE", "MIN", "PTS", "REB", "AST", "FG3M"}
    missing = required - set(logs.columns)
    if missing:
        raise SystemExit("Game logs missing required columns: " + ", ".join(sorted(missing)))
    logs["MIN"] = pd.to_numeric(logs["MIN"], errors="coerce")
    logs = logs.dropna(subset=["PLAYER_ID", "GAME_DATE"])
    logs = logs[logs["MIN"].fillna(0) > 0].copy()
    logs = logs.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"], ascending=[True, False, False])
    recent = logs.groupby("PLAYER_ID", sort=False, group_keys=False).head(10)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    recent.to_csv(out, index=False)
    print(f"Saved {len(recent):,} recent player-game rows for {recent['PLAYER_ID'].nunique():,} players to {out}")


if __name__ == "__main__":
    main()
