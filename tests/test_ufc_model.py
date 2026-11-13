import unittest
from datetime import date, timedelta

import pandas as pd

from ufc_model import audit_results, build_chronological_rows, train_and_evaluate


def synthetic_results():
    rows = []
    fighters = [f"Fighter {i:02d}" for i in range(40)]
    first_day = date(2018, 1, 1)
    for event in range(24):
        event_day = first_day + timedelta(days=event * 28)
        rotated = fighters[event:] + fighters[:event]
        for bout in range(20):
            a, b = rotated[bout], rotated[39 - bout]
            rows.append({
                "Date": event_day.isoformat(),
                "Fighter 1": a,
                "Fighter 2": b,
                "Result": 1 if (bout + event) % 2 == 0 else 2,
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

    def test_fighter_identity_normalization_merges_punctuation_variants(self):
        data = pd.DataFrame([
            {"Date": "2020-01-01", "Fighter 1": "O'Malley", "Fighter 2": "Opponent", "Result": 1},
            {"Date": "2020-02-01", "Fighter 1": "OMalley", "Fighter 2": "Other", "Result": 2},
        ])
        _, states = build_chronological_rows(data)
        self.assertEqual(sum("omalley" in name.replace(" ", "") for name in states), 1)

    def test_audit_flags_duplicate_bouts_and_same_day_repeats(self):
        data = synthetic_results().iloc[:2].copy()
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        audit = audit_results(data)
        self.assertEqual(audit["duplicate_bout_rows"], 1)
        self.assertEqual(audit["same_day_repeat_fighters"], 2)

    def test_train_calibration_and_test_are_separate_event_periods(self):
        _, metrics, _, _ = train_and_evaluate(synthetic_results())
        self.assertIn("elo_brier", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertGreater(metrics["train_fights"], 0)
        self.assertGreater(metrics["calibration_fights"], 0)
        self.assertGreater(metrics["test_fights"], 0)
        self.assertLess(metrics["train_end"], metrics["calibration_start"])
        self.assertLess(metrics["calibration_end"], metrics["test_start"])
        self.assertLessEqual(metrics["train_start"], metrics["train_end"])
        self.assertLessEqual(metrics["test_start"], metrics["test_end"])


if __name__ == "__main__":
    unittest.main()
