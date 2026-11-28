"""Generic minutes × per-minute-rate model for NBA rebound and assist forecasts."""
import numpy as np


class RateMinutesPredictor:
    """Blend a role-aware minutes×rate estimate with a direct stat forecast."""

    def __init__(self, minutes_model, rate_model, feature_cols, market,
                 direct_model=None, direct_weight=0.0):
        self.minutes_model = minutes_model
        self.rate_model = rate_model
        self.feature_cols = list(feature_cols)
        self.market = str(market)
        self.direct_model = direct_model
        self.direct_weight = float(np.clip(direct_weight, 0.0, 1.0))
        self.rate_cap = 0.85 if self.market == "rebounds" else 0.65

    def predict_with_details(self, X):
        frame = X.reindex(columns=self.feature_cols)
        minutes = np.clip(self.minutes_model.predict(frame), 0.0, 48.0)
        rate = np.clip(self.rate_model.predict(frame), 0.0, self.rate_cap)
        rate_projection = np.maximum(0.0, minutes * rate)
        direct_projection = None
        if self.direct_model is not None and self.direct_weight > 0:
            direct_projection = np.maximum(0.0, self.direct_model.predict(frame))
            projection = ((1.0 - self.direct_weight) * rate_projection
                          + self.direct_weight * direct_projection)
        else:
            projection = rate_projection
        return {
            "projection": projection,
            "rate_projection": rate_projection,
            "direct_projection": direct_projection,
            "expected_minutes": minutes,
            "per_minute_rate": rate,
        }

    def predict(self, X):
        return self.predict_with_details(X)["projection"]
