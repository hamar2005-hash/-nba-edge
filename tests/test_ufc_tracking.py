import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ufc_tracking import record_predictions, record_best_prices, settle_predictions


class UFCTrackingTests(unittest.TestCase):
    def test_records_and_grades_winner_method_and_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.csv"
            event = {
                "id": "event-1", "name": "Test UFC", "date": "2026-11-20T20:00:00Z"
            }
            winners = pd.DataFrame([{
                "Bout": "Fighter A vs Fighter B",
                "Predicted winner": "Fighter A",
                "Confidence": 68.0,
            }])
            auxiliary = pd.DataFrame([{
                "Bout": "Fighter A vs Fighter B",
                "Most likely method": "Decision",
                "Over 1.5 %": 75.0,
                "Over 2.5 %": 40.0,
            }])
            saved = record_predictions(
                path, event, winners, auxiliary_predictions=auxiliary,
                now_utc=pd.Timestamp("2026-11-19T12:00:00Z"),
            )
            self.assertEqual(saved, 1)

            results = pd.DataFrame([{
                "Date": "20/11/2026", "Fighter 1": "Fighter A",
                "Fighter 2": "Fighter B", "Result": 1,
                "Method": "Decision - Unanimous",
            }])
            stats = pd.DataFrame([
                {"Date": "20/11/2026", "Name": "Fighter A", "Time": 900},
                {"Date": "20/11/2026", "Name": "Fighter B", "Time": 900},
            ])
            changed = settle_predictions(path, results, stats)
            self.assertEqual(changed, 1)
            history = pd.read_csv(path).iloc[0]
            self.assertEqual(history["status"], "Correct")
            self.assertEqual(history["method_status"], "Correct")
            self.assertEqual(history["over_1_5_status"], "Correct")
            self.assertEqual(history["over_2_5_status"], "Incorrect")
            self.assertEqual(history["actual_duration_seconds"], 900)

    def test_records_round_prices_and_one_unit_profit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.csv"
            event = {"id": "event-3", "name": "Test UFC", "date": "2026-11-20T20:00:00Z"}
            winners = pd.DataFrame([{
                "Bout": "Fighter A vs Fighter B", "Predicted winner": "Fighter A",
                "Confidence": 65.0,
            }])
            auxiliary = pd.DataFrame([{
                "Bout": "Fighter A vs Fighter B", "Most likely method": "Decision",
                "Over 1.5 %": 75.0, "Over 2.5 %": 35.0,
            }])
            record_predictions(path, event, winners, auxiliary_predictions=auxiliary,
                               now_utc=pd.Timestamp("2026-11-19T12:00:00Z"))
            odds = [{
                "home_team": "Fighter A", "away_team": "Fighter B",
                "bookmakers": [{
                    "title": "Test Book",
                    "markets": [{
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 1.5, "price": 1.8},
                            {"name": "Under", "point": 1.5, "price": 2.0},
                            {"name": "Over", "point": 2.5, "price": 2.2},
                            {"name": "Under", "point": 2.5, "price": 1.6},
                        ],
                    }],
                }],
            }]
            saved = record_best_prices(path, odds)
            self.assertEqual(saved, 2)
            results = pd.DataFrame([{
                "Date": "20/11/2026", "Fighter 1": "Fighter A",
                "Fighter 2": "Fighter B", "Result": 1, "Method": "Decision",
            }])
            stats = pd.DataFrame([
                {"Date": "20/11/2026", "Name": "Fighter A", "Time": 900},
                {"Date": "20/11/2026", "Name": "Fighter B", "Time": 900},
            ])
            settle_predictions(path, results, stats)
            row = pd.read_csv(path).iloc[0]
            self.assertEqual(row["over_1_5_odds"], 1.8)
            self.assertEqual(row["over_2_5_odds"], 1.6)
            self.assertEqual(row["over_1_5_bookmaker"], "Test Book")
            self.assertEqual(row["over_2_5_bookmaker"], "Test Book")
            self.assertAlmostEqual(row["over_1_5_profit"], 0.8)
            self.assertAlmostEqual(row["over_2_5_profit"], -1.0)

    def test_does_not_record_after_event_start(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.csv"
            event = {"id": "event-2", "date": "2026-11-20T20:00:00Z"}
            predictions = pd.DataFrame([{
                "Bout": "A vs B", "Predicted winner": "A", "Confidence": 60
            }])
            saved = record_predictions(
                path, event, predictions,
                now_utc=pd.Timestamp("2026-11-20T20:01:00Z"),
            )
            self.assertEqual(saved, 0)

    def test_no_contest_is_void_and_not_a_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.csv"
            event = {"id": "event-nc", "name": "Test UFC", "date": "2026-11-20T20:00:00Z"}
            winners = pd.DataFrame([{
                "Bout": "Fighter A vs Fighter B", "Predicted winner": "Fighter A",
                "Confidence": 70.0,
            }])
            auxiliary = pd.DataFrame([{
                "Bout": "Fighter A vs Fighter B", "Most likely method": "Decision",
                "Over 1.5 %": 70.0, "Over 2.5 %": 60.0,
            }])
            record_predictions(path, event, winners, auxiliary_predictions=auxiliary,
                               now_utc=pd.Timestamp("2026-11-19T12:00:00Z"))
            results = pd.DataFrame([{
                "Date": "20/11/2026", "Fighter 1": "Fighter A",
                "Fighter 2": "Fighter B", "Result": 4, "Method": "Overturned",
            }])
            # Even if duration data exists, a no-contest must void every market.
            stats = pd.DataFrame([
                {"Date": "20/11/2026", "Name": "Fighter A", "Time": 900},
                {"Date": "20/11/2026", "Name": "Fighter B", "Time": 900},
            ])
            changed = settle_predictions(path, results, stats)
            self.assertEqual(changed, 1)
            row = pd.read_csv(path).iloc[0]
            self.assertEqual(row["status"], "Void")
            self.assertEqual(row["method_status"], "Void")
            self.assertEqual(row["over_1_5_status"], "Void")
            self.assertEqual(row["over_2_5_status"], "Void")
            self.assertTrue(pd.isna(row["over_1_5_profit"]))
            self.assertTrue(pd.isna(row["over_2_5_profit"]))
            self.assertTrue(pd.isna(row["actual_duration_seconds"]))

    def test_does_not_capture_odds_after_event_start(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.csv"
            event = {"id": "event-live", "name": "Test UFC", "date": "2026-11-15T20:00:00Z"}
            winners = pd.DataFrame([{
                "Bout": "Fighter A vs Fighter B", "Predicted winner": "Fighter A",
                "Confidence": 65.0,
            }])
            record_predictions(path, event, winners,
                               now_utc=pd.Timestamp("2026-11-14T12:00:00Z"))
            odds = [{
                "home_team": "Fighter A", "away_team": "Fighter B",
                "bookmakers": [{"title": "Live Book", "markets": [{
                    "key": "h2h", "outcomes": [
                        {"name": "Fighter A", "price": 1.8},
                        {"name": "Fighter B", "price": 2.1},
                    ],
                }]}],
            }]
            saved = record_best_prices(path, odds)
            self.assertEqual(saved, 0)
            row = pd.read_csv(path).iloc[0]
            self.assertTrue(pd.isna(row["predicted_odds"]))


if __name__ == "__main__":
    unittest.main()
