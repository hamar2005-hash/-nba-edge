"""Same-player NBA Bet Builder candidates.

The joint probabilities are experimental bivariate-normal estimates based on
model projections and each market model's holdout residual spread. They are
not calibrated sportsbook probabilities and no prices are scraped or inferred.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKETS = {
    "points": {"column": "points_projection", "steps": [5, 10, 15, 20, 25, 30]},
    "rebounds": {"column": "rebounds_projection", "steps": [3, 5, 7, 10, 12]},
    "assists": {"column": "assists_projection", "steps": [2, 4, 6, 8, 10]},
}
CORRELATIONS = {
    ("points", "rebounds"): 0.18,
    ("points", "assists"): 0.32,
    ("rebounds", "assists"): 0.12,
}


def _residual_spread(model_dir, market):
    try:
        import joblib
        pack = joblib.load(model_dir / f"{market}_model.joblib")
        value = float(pack.get("residual_std", np.nan))
        if np.isfinite(value) and value > 0:
            return value
    except Exception:
        pass
    return {"points": 6.0, "rebounds": 3.0, "assists": 2.5}[market]


def generate_same_player_builders(predictions: pd.DataFrame, model_dir, simulations=30000) -> pd.DataFrame:
    """Create two-leg same-player milestones and estimate joint hit probability."""
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(20261113)
    spreads = {market: _residual_spread(model_dir, market) for market in MARKETS}
    rows = []

    for _, player in predictions.iterrows():
        available = []
        for market, spec in MARKETS.items():
            mean = pd.to_numeric(pd.Series([player.get(spec["column"])]), errors="coerce").iloc[0]
            if pd.notna(mean) and float(mean) > 0:
                available.append((market, float(mean), spec["steps"]))
        if len(available) < 2:
            continue

        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                m1, mean1, steps1 = available[i]
                m2, mean2, steps2 = available[j]
                rho = CORRELATIONS.get((m1, m2), CORRELATIONS.get((m2, m1), 0.15))
                covariance = [[spreads[m1] ** 2, rho * spreads[m1] * spreads[m2]],
                              [rho * spreads[m1] * spreads[m2], spreads[m2] ** 2]]
                draws = rng.multivariate_normal([mean1, mean2], covariance, size=simulations)
                for line1 in steps1:
                    if line1 > mean1 + 1.25 * spreads[m1] or line1 < mean1 - 2.25 * spreads[m1]:
                        continue
                    for line2 in steps2:
                        if line2 > mean2 + 1.25 * spreads[m2] or line2 < mean2 - 2.25 * spreads[m2]:
                            continue
                        p1 = draws[:, 0] >= line1
                        p2 = draws[:, 1] >= line2
                        joint = float(np.mean(p1 & p2))
                        if not 0.50 <= joint <= 0.82:
                            continue
                        rows.append({
                            "game_date": player.get("game_date"),
                            "game": player.get("game"),
                            "player": player.get("player"),
                            "team": player.get("team"),
                            "opponent": player.get("opponent"),
                            "leg_1": f"{int(line1)}+ {m1}",
                            "leg_2": f"{int(line2)}+ {m2}",
                            "projection_1": round(mean1, 1),
                            "projection_2": round(mean2, 1),
                            "joint_probability_pct": round(joint * 100, 1),
                            "estimated_fair_decimal_odds": round(1 / joint, 2),
                            "model_type": "Experimental joint estimate",
                        })

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["game", "joint_probability_pct"], ascending=[True, False]
    )
    # Keep the best few sensible alternatives per player and game, avoiding
    # a flood of near-identical milestone combinations.
    result = result.drop_duplicates(["game", "player", "leg_1", "leg_2"])
    return result.groupby(["game", "player"], group_keys=False).head(4).reset_index(drop=True)
