"""Dedicated points model: forecast playing time and scoring rate separately."""
import numpy as np
import pandas as pd


class PointsV2Predictor:
    """Predict points as expected minutes multiplied by expected scoring rate."""

    def __init__(self, minutes_model, rate_model, feature_cols):
        self.minutes_model = minutes_model
        self.rate_model = rate_model
        self.feature_cols = list(feature_cols)

    def predict_with_details(self, X):
        frame = X.reindex(columns=self.feature_cols)
        minutes = np.clip(self.minutes_model.predict(frame), 0.0, 48.0)
        points_per_minute = np.clip(self.rate_model.predict(frame), 0.0, 1.5)
        points = np.maximum(0.0, minutes * points_per_minute)
        return {
            "points": points,
            "expected_minutes": minutes,
            "points_per_minute": points_per_minute,
        }

    def predict(self, X):
        return self.predict_with_details(X)["points"]
