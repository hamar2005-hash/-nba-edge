import unittest
from datetime import date, timedelta

import pandas as pd

from ufc_model import build_chronological_rows, train_and_evaluate


def synthetic_results():
    rows = []
    fighters = [f"Fighter {i:02d}" for i in range(12)]
    first_day = date(2018, 1, 1)
    for event in range(20):
        event_day = first_day + timedelta(days=event * 28)
        for bout in range(20):
            a = fighters[(bout * 3 + event) % len(fighters)]
            b = fighters[(bout * 3 + event + 5) % len(fighters)]
            if a == b:
                b = fighters[(fighters.index(b) + 1) % len(fighters)]
            rows.append({
                "Date": event_day.isoformat(),
                "Fighter 1": a,
                "Fighter 2": b,
                "Result": 1 if bout % 2 == 0 else 2,
                "Method": "Decision",
            })
    return pd.DataFrame(rows)


class UFCChronologyTests(unittest.TestCase):
    def test_draws_are_excluded_from_binary_training_rows(self):
        data = synthetic_results().iloc[:40].copy()
        data.loc[data.index[0], "Result"] = 3
        frame, _ = build_chronological_rows(data)
        self.assertEqual(len(frame), 39)
        self.assertTrue(set(frame["target"].unique()).issubset({0, 1}))

    def test_train_calibration_and_test_are_separate_event_periods(self):
        _, metrics, _, _ = train_and_evaluate(synthetic_results())
        self.assertGreater(metrics["train_fights"], 0)
        self.assertGreater(metrics["calibration_fights"], 0)
        self.assertGreater(metrics["test_fights"], 0)
        self.assertLess(metrics["train_end"], metrics["calibration_start"])
        self.assertLess(metrics["calibration_end"], metrics["test_start"])
        self.assertLessEqual(metrics["train_start"], metrics["train_end"])
        self.assertLessEqual(metrics["test_start"], metrics["test_end"])


if __name__ == "__main__":
    unittest.main()
