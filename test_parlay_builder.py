import unittest

import pandas as pd

from parlay_builder import build_three_leg_parlays


class ThreeLegParlayTests(unittest.TestCase):
    def candidates(self):
        return pd.DataFrame([
            {"Player":"Player A","Team":"BOS","Opponent":"NYK","Game date":"2026-11-28","Game time UTC":"2026-11-28T23:00:00Z","Market":"Points","Best side":"Over","Prop line":19.5,"Over probability":"65.0%","Under probability":"35.0%","Estimated confidence":65.0,"Sky Bet odds":2.0},
            {"Player":"Player B","Team":"MIA","Opponent":"ORL","Game date":"2026-11-28","Game time UTC":"2026-11-28T23:30:00Z","Market":"Rebounds","Best side":"Over","Prop line":5.5,"Over probability":"65.0%","Under probability":"35.0%","Estimated confidence":65.0,"Sky Bet odds":2.0},
            {"Player":"Player C","Team":"LAL","Opponent":"GSW","Game date":"2026-11-28","Game time UTC":"2026-11-29T01:00:00Z","Market":"Assists","Best side":"Under","Prop line":6.5,"Over probability":"35.0%","Under probability":"65.0%","Estimated confidence":65.0,"Sky Bet odds":2.0},
        ])

    def test_builds_three_leg_value_parlay_from_three_games(self):
        parlays = build_three_leg_parlays(self.candidates())
        self.assertEqual(len(parlays), 1)
        self.assertEqual(len(parlays[0]["legs"]), 3)
        self.assertAlmostEqual(parlays[0]["combined_odds"], 8.0)
        self.assertGreater(parlays[0]["edge_pct"], 3.0)

    def test_missing_odds_never_get_invented(self):
        frame = self.candidates()
        frame["Sky Bet odds"] = float("nan")
        self.assertEqual(build_three_leg_parlays(frame), [])

    def test_same_game_legs_are_not_combined(self):
        frame = self.candidates()
        frame.loc[1, "Opponent"] = "BOS"
        frame.loc[1, "Team"] = "NYK"
        self.assertEqual(build_three_leg_parlays(frame), [])


if __name__ == "__main__":
    unittest.main()
