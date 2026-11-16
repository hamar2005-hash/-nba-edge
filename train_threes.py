"""Train and validate the dedicated NBA three-pointers-made model."""
from pathlib import Path
import argparse, json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from train_models import prepare, BASE_FEATURES, POINT_EXTRA

FEATURES = [
    "MIN_LAST", "MIN_L3", "MIN_STD_L5", "MIN_STD_L10", "MIN_TREND_5V10", "MIN_SEASON_PRE",
    "FG3M_L5", "FG3M_L10", "FG3M_SEASON_PRE",
    "FG3A_L5", "FG3A_L10", "FG3A_SEASON_PRE",
    "THREE_PCT_L5", "THREE_PCT_L10", "THREE_PCT_SEASON_PRE",
    "THREE_PA_RATE_L5", "THREE_PA_RATE_L10", "THREE_PA_RATE_SEASON_PRE",
    "USAGE_PROXY_L5", "USAGE_PROXY_L10", "USAGE_PROXY_SEASON_PRE",
    "PTS_PER_MIN_L5", "PTS_PER_MIN_L10", "PTS_PER_MIN_SEASON_PRE",
    "OPP_ALLOWED_3PA_L5", "OPP_ALLOWED_3PA_L10", "OPP_ALLOWED_3PA_SEASON_PRE",
    "OPP_ALLOWED_3P_PCT_L5", "OPP_ALLOWED_3P_PCT_L10", "OPP_ALLOWED_3P_PCT_SEASON_PRE",
    "REST_DAYS", "HOME", "GAMES_PLAYED_BEFORE",
]

def make_model(seed=42):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=300, learning_rate=.045, max_leaf_nodes=15,
            l2_regularization=2.5, random_state=seed
        ))
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/player_game_logs.csv")
    parser.add_argument("--outdir", default="models")
    args = parser.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    df = prepare(args.input)
    features = [c for c in FEATURES if c in df.columns]
    missing = sorted(set(FEATURES) - set(features))
    if missing:
        raise SystemExit("Missing required 3PM features: " + ", ".join(missing))
    data = df.dropna(subset=["FG3M"]).copy()
    data = data[(data["GAMES_PLAYED_BEFORE"] >= 2) & (data["MIN"] > 0)]
    dates = np.sort(data["GAME_DATE"].dropna().unique())
    if len(dates) < 30:
        raise SystemExit("Not enough distinct game dates to train the 3PM model.")
    cutoff = dates[max(1, int(len(dates) * .8))]
    train = data[data.GAME_DATE < cutoff]
    test = data[data.GAME_DATE >= cutoff]
    if len(train) < 1000 or len(test) < 100:
        raise SystemExit(f"Insufficient 3PM rows: train={len(train)}, test={len(test)}")
    model = make_model()
    model.fit(train[features], train["FG3M"])
    pred = np.maximum(0, model.predict(test[features]))
    residual = test["FG3M"].to_numpy() - pred
    mae = float(mean_absolute_error(test["FG3M"], pred))
    rmse = float(np.sqrt(mean_squared_error(test["FG3M"], pred)))
    # Compare against a lagged last-five/season average on exactly the same rows.
    baseline = test["FG3M_L5"].fillna(test["FG3M_SEASON_PRE"]).fillna(train["FG3M"].median()).to_numpy()
    baseline_mae = float(mean_absolute_error(test["FG3M"], baseline))
    if mae >= baseline_mae:
        raise SystemExit(f"3PM candidate failed baseline check: model MAE={mae:.4f}, baseline MAE={baseline_mae:.4f}")
    pack = {
        "model": model, "feature_cols": features, "target": "FG3M", "market": "threes",
        "model_version": "dedicated_threes_made_v1", "residual_std": float(np.std(residual, ddof=1)),
        "holdout_mae": mae, "holdout_rmse": rmse, "holdout_start": str(pd.Timestamp(cutoff).date()),
    }
    joblib.dump(pack, out / "threes_model.joblib")
    latest = df.sort_values("GAME_DATE").groupby("PLAYER_ID", as_index=False).tail(1)
    identity = ["PLAYER_ID", "PLAYER_NAME"] + (["TEAM_ABBREVIATION"] if "TEAM_ABBREVIATION" in latest else []) + ["GAME_DATE"]
    snapshot_cols = list(dict.fromkeys(
        identity + [c for c in BASE_FEATURES + POINT_EXTRA + features if c in latest.columns]
    ))
    latest[snapshot_cols].to_csv(out / "latest_player_features.csv", index=False)
    report = {
        "market": "three-pointers made", "model_version": pack["model_version"],
        "train_rows": int(len(train)), "holdout_rows": int(len(test)),
        "holdout_start": pack["holdout_start"], "feature_count": len(features),
        "candidate_mae": mae, "candidate_rmse": rmse,
        "last5_or_season_baseline_mae": baseline_mae,
        "beats_baseline": mae < baseline_mae,
        "residual_std": pack["residual_std"],
    }
    (out / "threes_model_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
