import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ufc_tracking import record_predictions, settle_predictions


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


if __name__ == "__main__":
    unittest.main()
