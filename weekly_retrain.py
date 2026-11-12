"""Weekly candidate retraining with a chronological holdout and incumbent-model gate.

Historical game logs are incrementally cached by the GitHub Actions workflow.
Each market is replaced only when the candidate has lower MAE on the same
chronological holdout. The latest pregame feature snapshot is refreshed either way.
"""
from pathlib import Path
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from train_models import prepare, TARGETS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
CANDIDATES = ROOT / "candidate_models"
STATS = ['PTS','REB','AST','MIN','FGA','FG3A','FTA','TOV','STL','BLK']

def train_candidate(train, test, features, market, target):
    tr = train.dropna(subset=[target])
    te = test.dropna(subset=[target])
    tr = tr[tr.GAMES_PLAYED_BEFORE >= 2]
    te = te[te.GAMES_PLAYED_BEFORE >= 2]
    if len(tr) < 1000 or len(te) < 100:
        raise RuntimeError(f"Insufficient {market} rows: train={len(tr)}, holdout={len(te)}")
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('regressor', HistGradientBoostingRegressor(
            max_iter=250, learning_rate=.055, max_leaf_nodes=15,
            l2_regularization=2.0, random_state=42
        ))
    ])
    model.fit(tr[features], tr[target])
    pred = model.predict(te[features])
    residual = te[target].to_numpy() - pred
    return {
        "model": model, "feature_cols": features, "target": target, "market": market,
        "residual_std": float(np.std(residual, ddof=1)),
    }, {
        "train_rows": int(len(tr)), "test_rows": int(len(te)),
        "mae": float(mean_absolute_error(te[target], pred)),
        "rmse": float(np.sqrt(mean_squared_error(te[target], pred))),
    }, te

def main():
    source = DATA / "player_game_logs.csv"
    if not source.exists():
        raise SystemExit("Historical player game logs are missing.")
    df = prepare(source)
    dates = np.sort(df.GAME_DATE.dropna().unique())
    if len(dates) < 30:
        raise SystemExit("Not enough distinct game dates for a safe weekly retrain.")
    cutoff = dates[max(1, int(len(dates) * .8))]
    train = df[df.GAME_DATE < cutoff]
    test = df[df.GAME_DATE >= cutoff]
    features = [c for c in df.columns if c.endswith(('_L5','_L10','_SEASON_PRE'))]
    features = list(dict.fromkeys(features + ['REST_DAYS','HOME','GAMES_PLAYED_BEFORE']))
    MODELS.mkdir(exist_ok=True)
    CANDIDATES.mkdir(exist_ok=True)
    report = {"trained_at_utc": pd.Timestamp.utcnow().isoformat(), "holdout_start": str(pd.Timestamp(cutoff).date()), "markets": {}}
    accepted_any = False

    for market, target in TARGETS.items():
        candidate, candidate_metrics, holdout = train_candidate(train, test, features, market, target)
        incumbent_path = MODELS / f"{market}_model.joblib"
        incumbent_mae = None
        if incumbent_path.exists():
            incumbent = joblib.load(incumbent_path)
            incumbent_features = incumbent.get("feature_cols", features)
            incumbent_test = holdout.dropna(subset=[target])
            incumbent_test = incumbent_test[incumbent_test.GAMES_PLAYED_BEFORE >= 2]
            try:
                incumbent_pred = incumbent["model"].predict(incumbent_test[incumbent_features])
                incumbent_mae = float(mean_absolute_error(incumbent_test[target], incumbent_pred))
            except Exception as exc:
                print(f"{market}: incumbent could not be scored on common holdout ({exc}); candidate not auto-accepted.")
        accept = incumbent_mae is not None and candidate_metrics["mae"] < incumbent_mae
        if accept:
            joblib.dump(candidate, incumbent_path)
            accepted_any = True
        report["markets"][market] = {
            "incumbent_holdout_mae": incumbent_mae,
            "candidate_holdout_mae": candidate_metrics["mae"],
            "candidate_holdout_rmse": candidate_metrics["rmse"],
            "candidate_accepted": bool(accept),
            "train_rows": candidate_metrics["train_rows"],
            "holdout_rows": candidate_metrics["test_rows"],
        }
        print(f"{market}: incumbent MAE={incumbent_mae}; candidate MAE={candidate_metrics['mae']:.4f}; accepted={accept}")

    # Refresh features with all completed games, even when a model is retained.
    latest = df.sort_values('GAME_DATE').groupby('PLAYER_ID', as_index=False).tail(1)
    cols = ['PLAYER_ID','PLAYER_NAME'] + (['TEAM_ABBREVIATION'] if 'TEAM_ABBREVIATION' in latest else []) + ['GAME_DATE'] + features
    latest[cols].to_csv(MODELS / "latest_player_features.csv", index=False)
    report["any_model_updated"] = accepted_any
    (MODELS / "weekly_retrain_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
