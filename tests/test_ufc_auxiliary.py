import unittest
from datetime import date, timedelta

import pandas as pd

from ufc_model import (
    AUX_METHOD_FEATURES,
    build_auxiliary_training_rows,
    predict_auxiliary_matchups,
    train_method_and_round_models,
)


def synthetic_history():
    results, stats = [], []
    methods = ("KO/TKO", "Submission", "Decision - Unanimous")
    durations = (300, 440, 600, 740, 900, 1200, 1500)
    start = date(2010, 1, 1)
    for event_idx in range(120):
        event_date = start + timedelta(days=event_idx * 7)
        date_text = event_date.strftime("%d/%m/%Y")
        for bout_idx in range(6):
            n = (event_idx * 12 + bout_idx * 2) % 240
            f1, f2 = f"Fighter {n}", f"Fighter {n + 1}"
            method = methods[(event_idx + bout_idx) % len(methods)]
            duration = durations[(event_idx * 6 + bout_idx) % len(durations)]
            results.append({
                "Date": date_text, "Fighter 1": f1, "Fighter 2": f2,
                "Result": 1 if (event_idx + bout_idx) % 2 else 2,
                "Method": method,
            })
            for fighter_offset, fighter in enumerate((f1, f2)):
                stats.append({
                    "Date": date_text, "Name": fighter, "Time": duration,
                    "Knockdowns": (event_idx + bout_idx + fighter_offset) % 2,
                    "Sig Strikes Landed": 12 + bout_idx + fighter_offset,
                    "Sig Strikes Attempted": 28 + bout_idx + fighter_offset,
                    "Sig Strikes Absorbed": 10 + event_idx % 4,
                    "Takedowns": bout_idx % 3,
                    "Takedown Attempts": 1 + bout_idx % 4,
                    "Submission Attempts": (event_idx + fighter_offset) % 3,
                    "Clinch Strikes": 2 + bout_idx,
                    "Ground Strikes": 1 + event_idx % 5,
                })
    return pd.DataFrame(results), pd.DataFrame(stats)


class AuxiliaryUFCModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.stats = synthetic_history()

    def test_features_are_pre_fight_and_duration_is_label_only(self):
        frame, snapshot = build_auxiliary_training_rows(self.results, self.stats)
        self.assertGreater(len(frame), 600)
        self.assertTrue(set(AUX_METHOD_FEATURES).issubset(frame.columns))
        self.assertTrue(frame["duration_seconds"].notna().all())
        self.assertNotIn("duration_seconds", AUX_METHOD_FEATURES)
        self.assertEqual(frame.iloc[0]["sig_landed_rate_mean"], 0.0)
        self.assertTrue({"ko_win", "sub_win", "decision_win", "sig_landed_rate", "takedown_accuracy"}.issubset(
            set(snapshot[next(iter(snapshot))].keys())
        ))
        self.assertGreater(frame.iloc[-1]["sig_landed_rate_mean"], 0)

    def test_models_train_and_predict(self):
        models, metrics, snapshot, frame = train_method_and_round_models(
            self.results, self.stats
        )
        self.assertTrue(metrics["method"]["available"], metrics["method"])
        self.assertTrue(metrics["rounds"]["1.5"]["available"], metrics["rounds"]["1.5"])
        self.assertTrue(metrics["rounds"]["2.5"]["available"], metrics["rounds"]["2.5"])
        predictions = predict_auxiliary_matchups(models, snapshot, [{
            "Fighter 1": "Fighter 1", "Fighter 2": "Fighter 2"
        }])
        self.assertEqual(len(predictions), 1)
        self.assertIn("Most likely method", predictions.columns)
        for line in ("1.5", "2.5"):
            self.assertIn(f"Over {line} %", predictions.columns)
            self.assertIn(f"Under {line} %", predictions.columns)
            total = predictions.iloc[0][f"Over {line} %"] + predictions.iloc[0][f"Under {line} %"]
            self.assertAlmostEqual(total, 100.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
