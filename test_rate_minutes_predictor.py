import unittest
import numpy as np
from rate_minutes_predictor import RateMinutesPredictor


class FixedModel:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


class RateMinutesPredictorTests(unittest.TestCase):
    def test_minutes_rate_and_direct_blend(self):
        model = RateMinutesPredictor(
            FixedModel(30), FixedModel(0.2), ["feature"], "rebounds",
            direct_model=FixedModel(8), direct_weight=0.5
        )
        details = model.predict_with_details(__import__("pandas").DataFrame({"feature": [1, 2]}))
        np.testing.assert_allclose(details["expected_minutes"], [30, 30])
        np.testing.assert_allclose(details["rate_projection"], [6, 6])
        np.testing.assert_allclose(details["projection"], [7, 7])

    def test_assist_rate_is_capped(self):
        model = RateMinutesPredictor(FixedModel(40), FixedModel(2), ["feature"], "assists")
        result = model.predict(__import__("pandas").DataFrame({"feature": [1]}))
        np.testing.assert_allclose(result, [26])


if __name__ == "__main__":
    unittest.main()
