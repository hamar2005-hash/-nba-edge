import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import publish_daily_x_picks as publisher


class DailyPublisherTests(unittest.TestCase):
    def test_wilson_lower_bound_is_conservative(self):
        lower = publisher.wilson_lower(70, 100)
        self.assertGreater(lower, 0.5)
        self.assertLess(lower, 0.7)

    def test_over_probability_increases_with_projection(self):
        low = publisher.normal_over_probability(18, 19.5, 5, "points")
        high = publisher.normal_over_probability(25, 19.5, 5, "points")
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_no_pick_post_is_transparent(self):
        post = publisher.build_post("2026-11-28", [])
        self.assertIn("No picks cleared", post)
        self.assertIn("No forced selections", post)

    def test_calibration_uses_settled_historical_results(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "picks.csv"
            frame = pd.DataFrame({
                "market": ["points"] * 100,
                "confidence_pct": [70.0] * 100,
                "raw_confidence_pct": [72.0] * 100,
                "Actual result": ["Win"] * 70 + ["Loss"] * 30,
            })
            frame.to_csv(path, index=False)
            with patch.object(publisher, "PICKS_PATH", path):
                calibration = publisher.load_calibration()
            self.assertIn("points", calibration)
            band = next(x for x in calibration["points"] if x["low"] == 70)
            self.assertEqual(band["n"], 100)
            self.assertAlmostEqual(band["hit_rate"], 0.7)


if __name__ == "__main__":
    unittest.main()
