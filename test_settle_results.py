"""Unit tests for NBA Edge single-stat pick settlement."""
import tempfile
from pathlib import Path
import pandas as pd
from settle_results import grade_pick, update_pick_history

def test_grade_pick_win_loss_push_pending():
    assert grade_pick("Over", 20.5, 21) == "Win"
    assert grade_pick("Under", 20.5, 20) == "Win"
    assert grade_pick("Over", 20.0, 20) == "Push"
    assert grade_pick("Under", 20.5, 21) == "Loss"
    assert grade_pick("Over", 20.5, None) == "Pending"

def test_update_pick_history_matches_team_and_grades():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "picks.csv"
        pd.DataFrame([
            {"game_date":"2026-11-18","player":"Example Player","team":"BOS","market":"points","side":"Over","line":19.5,"Actual result":"Pending","Benchmark result":"Pending"},
            {"game_date":"2026-11-18","player":"Example Player","team":"NYK","market":"points","side":"Under","line":19.5,"Actual result":"Pending","Benchmark result":"Pending"},
        ]).to_csv(path,index=False)
        settled = pd.DataFrame([
            {"game_date":"2026-11-18","player":"Example Player","team":"BOS","actual_points":24},
        ])
        assert update_pick_history(settled,path) == 1
        result = pd.read_csv(path)
        assert result.loc[0,"Actual result"] == "Win"
        assert result.loc[1,"Actual result"] == "Pending"

if __name__ == "__main__":
    test_grade_pick_win_loss_push_pending()
    test_update_pick_history_matches_team_and_grades()
    print("Settlement tests passed.")
