"""Regression tests for the daily opponent-defense feature snapshot."""
import pandas as pd

from team_defense_snapshot import build_latest_team_defense_snapshot


def test_latest_pre_game_defense_row_is_selected_per_opponent():
    frame = pd.DataFrame([
        {"GAME_DATE": "2026-11-01", "GAME_ID": "001", "OPPONENT_ABBREVIATION": "BOS",
         "OPP_ALLOWED_3PA_L5": 31.0, "OPP_ALLOWED_EFG_L10": 0.51},
        {"GAME_DATE": "2026-11-03", "GAME_ID": "003", "OPPONENT_ABBREVIATION": "BOS",
         "OPP_ALLOWED_3PA_L5": 34.0, "OPP_ALLOWED_EFG_L10": 0.53},
        {"GAME_DATE": "2026-11-02", "GAME_ID": "002", "OPPONENT_ABBREVIATION": "NYK",
         "OPP_ALLOWED_3PA_L5": 29.0, "OPP_ALLOWED_EFG_L10": 0.49},
    ])

    snapshot = build_latest_team_defense_snapshot(frame).set_index("TEAM_ABBREVIATION")

    assert set(snapshot.index) == {"BOS", "NYK"}
    assert snapshot.loc["BOS", "OPP_ALLOWED_3PA_L5"] == 34.0
    assert snapshot.loc["BOS", "OPP_ALLOWED_EFG_L10"] == 0.53
    assert snapshot.loc["NYK", "OPP_ALLOWED_3PA_L5"] == 29.0


def test_empty_or_incomplete_input_returns_schema_safe_empty_snapshot():
    empty = build_latest_team_defense_snapshot(pd.DataFrame())
    assert list(empty.columns) == ["TEAM_ABBREVIATION"]

    missing_features = build_latest_team_defense_snapshot(pd.DataFrame([
        {"GAME_DATE": "2026-11-01", "OPPONENT_ABBREVIATION": "BOS"}
    ]))
    assert list(missing_features.columns) == ["TEAM_ABBREVIATION"]
    assert missing_features.empty


if __name__ == "__main__":
    test_latest_pre_game_defense_row_is_selected_per_opponent()
    test_empty_or_incomplete_input_returns_schema_safe_empty_snapshot()
    print("Opponent-defense snapshot tests passed.")
