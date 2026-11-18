"""Build the latest pre-game opponent defensive feature snapshot for daily inference."""
from pathlib import Path

import pandas as pd


def build_latest_team_defense_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one latest pre-game defensive feature row per defending team.

    In the prepared player-game table, OPPONENT_ABBREVIATION identifies the
    defending team and OPP_ALLOWED_* columns are shifted rolling/season features
    calculated before that historical game. Keeping the latest such row per
    opponent gives auto_daily.py a current team-keyed lookup without using the
    game being predicted.
    """
    feature_cols = sorted(
        column for column in frame.columns
        if column.startswith("OPP_ALLOWED_")
        and column.endswith(("_L5", "_L10", "_SEASON_PRE"))
    )
    output_cols = ["TEAM_ABBREVIATION", *feature_cols]
    required = {"GAME_DATE", "OPPONENT_ABBREVIATION"}
    if not required.issubset(frame.columns) or not feature_cols:
        return pd.DataFrame(columns=output_cols)

    rows = frame.dropna(subset=["GAME_DATE", "OPPONENT_ABBREVIATION"]).copy()
    rows["TEAM_ABBREVIATION"] = rows["OPPONENT_ABBREVIATION"].astype(str).str.upper().str.strip()
    rows = rows[rows["TEAM_ABBREVIATION"] != ""]
    if rows.empty:
        return pd.DataFrame(columns=output_cols)

    sort_cols = ["GAME_DATE"]
    if "GAME_ID" in rows.columns:
        sort_cols.append("GAME_ID")
    rows = rows.sort_values(sort_cols)
    latest = rows.groupby("TEAM_ABBREVIATION", sort=False, as_index=False).tail(1)
    return (
        latest[output_cols]
        .drop_duplicates("TEAM_ABBREVIATION", keep="last")
        .sort_values("TEAM_ABBREVIATION")
        .reset_index(drop=True)
    )


def save_latest_team_defense_snapshot(frame: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Write the team-keyed opponent-defense snapshot and return it."""
    snapshot = build_latest_team_defense_snapshot(frame)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(destination, index=False)
    return snapshot
