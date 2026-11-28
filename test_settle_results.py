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


def test_update_pick_history_grades_threes_over_under_and_push():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "picks.csv"
        pd.DataFrame([
            {"game_date":"2026-11-18","player":"Shooter One","team":"BOS","market":"threes","side":"Over","line":2.5,"Actual result":"Pending","Benchmark result":"Pending"},
            {"game_date":"2026-11-18","player":"Shooter Two","team":"NYK","market":"threes","side":"Under","line":2.5,"Actual result":"Pending","Benchmark result":"Pending"},
            {"game_date":"2026-11-18","player":"Shooter Three","team":"MIA","market":"threes","side":"Over","line":3.0,"Actual result":"Pending","Benchmark result":"Pending"},
        ]).to_csv(path,index=False)
        settled = pd.DataFrame([
            {"game_date":"2026-11-18","player":"Shooter One","team":"BOS","actual_threes":4},
            {"game_date":"2026-11-18","player":"Shooter Two","team":"NYK","actual_threes":1},
            {"game_date":"2026-11-18","player":"Shooter Three","team":"MIA","actual_threes":3},
        ])
        assert update_pick_history(settled,path) == 3
        result = pd.read_csv(path)
        assert result["Actual result"].tolist() == ["Win","Win","Push"]
        assert result["Benchmark result"].tolist() == ["Win","Win","Push"]


def test_update_pick_history_preserves_other_dates():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "picks.csv"
        pd.DataFrame([
            {"game_date":"2026-11-17","player":"Earlier Shooter","team":"BOS","market":"threes","side":"Over","line":2.5,"Actual result":"Win","Benchmark result":"Win"},
            {"game_date":"2026-11-18","player":"Tonight Shooter","team":"NYK","market":"threes","side":"Under","line":2.5,"Actual result":"Pending","Benchmark result":"Pending"},
        ]).to_csv(path,index=False)
        settled = pd.DataFrame([
            {"game_date":"2026-11-18","player":"Tonight Shooter","team":"NYK","actual_threes":1},
        ])
        assert update_pick_history(settled,path) == 1
        result = pd.read_csv(path)
        assert result["Actual result"].tolist() == ["Win","Win"]

if __name__ == "__main__":
    test_grade_pick_win_loss_push_pending()
    test_update_pick_history_matches_team_and_grades()
    test_update_pick_history_grades_threes_over_under_and_push()
    test_update_pick_history_preserves_other_dates()
    print("Settlement tests passed.")
