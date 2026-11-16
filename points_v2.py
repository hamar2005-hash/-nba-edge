"""Points predictor combining expected minutes, scoring rate, and direct points signal."""
import numpy as np


class PointsV2Predictor:
    """Predict points from role-adjusted minutes/rate with optional direct-model blend."""

    def __init__(self, minutes_model, rate_model, feature_cols, direct_model=None, direct_weight=0.0):
        self.minutes_model = minutes_model
        self.rate_model = rate_model
        self.feature_cols = list(feature_cols)
        self.direct_model = direct_model
        self.direct_weight = float(np.clip(direct_weight, 0.0, 1.0))

    def predict_with_details(self, X):
        frame = X.reindex(columns=self.feature_cols)
        minutes = np.clip(self.minutes_model.predict(frame), 0.0, 48.0)
        points_per_minute = np.clip(self.rate_model.predict(frame), 0.0, 1.5)
        rate_points = np.maximum(0.0, minutes * points_per_minute)
        direct_points = None
        if self.direct_model is not None and self.direct_weight > 0:
            direct_points = np.maximum(0.0, self.direct_model.predict(frame))
            points = (1.0 - self.direct_weight) * rate_points + self.direct_weight * direct_points
        else:
            points = rate_points
        return {
            "points": points,
            "rate_points": rate_points,
            "direct_points": direct_points,
            "expected_minutes": minutes,
            "points_per_minute": points_per_minute,
        }

    def predict(self, X):
        return self.predict_with_details(X)["points"]
