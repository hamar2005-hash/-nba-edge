"""Multi-market chronological audit for NBA Edge (research only; never promotes models)."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from train_models import prepare, TARGETS, BASE_FEATURES, POINT_EXTRA
from weekly_retrain import fit_points_v2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
FOLDS = 3


def market_features(df, market):
    base = [c for c in BASE_FEATURES if c in df.columns]
    if market == "points":
        return list(dict.fromkeys(base + [c for c in POINT_EXTRA if c in df.columns]))
    extras = []
    if market == "rebounds":
        # Lagged opportunity/role proxies available in standard player box scores.
        extras = ["MIN_LAST", "MIN_L3", "MIN_STD_L5", "MIN_STD_L10",
                  "MIN_TREND_5V10", "MIN_SEASON_PRE"]
        for stat in ("REB", "MIN", "FGA", "FG3A", "FTA", "BLK", "STL"):
            extras += [f"{stat}_L5", f"{stat}_L10", f"{stat}_SEASON_PRE"]
    elif market == "assists":
        extras = ["MIN_LAST", "MIN_L3", "MIN_STD_L5", "MIN_STD_L10",
                  "MIN_TREND_5V10", "MIN_SEASON_PRE"]
        for stat in ("AST", "MIN", "FGA", "TOV", "PTS"):
            extras += [f"{stat}_L5", f"{stat}_L10", f"{stat}_SEASON_PRE"]
    return list(dict.fromkeys(base + [c for c in extras if c in df.columns]))


def score(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "bias": float(np.mean(np.asarray(pred) - np.asarray(y))),
            "rows": int(len(y))}


def main():
    source = DATA / "player_game_logs.csv"
    if not source.exists():
        raise SystemExit("Historical player game logs are missing.")
    df = prepare(source)
    dates = np.sort(df.GAME_DATE.dropna().unique())
    if len(dates) < 80:
        raise SystemExit(f"Need at least 80 distinct game dates for a 3-fold audit; found {len(dates)}.")
    # Three expanding-window, forward-only test periods. No random shuffling.
    boundaries = [int(len(dates) * x) for x in (0.55, 0.70, 0.85, 1.0)]
    report = {"audit_type": "three_fold_expanding_chronological",
              "folds": [], "markets": {}, "data": {
                  "rows": int(len(df)), "players": int(df.PLAYER_ID.nunique()),
                  "first_date": str(pd.Timestamp(dates[0]).date()),
                  "last_date": str(pd.Timestamp(dates[-1]).date())}}
    detail_rows = []
    for fold_no in range(FOLDS):
        train_end, test_end = boundaries[fold_no + 1], boundaries[fold_no + 2]
        train_dates = dates[:train_end]
        test_dates = dates[train_end:test_end]
        if not len(test_dates):
            continue
        train = df[df.GAME_DATE <= train_dates[-1]]
        test = df[df.GAME_DATE.isin(test_dates)]
        fold_info = {"fold": fold_no + 1,
                     "train_end": str(pd.Timestamp(train_dates[-1]).date()),
                     "test_start": str(pd.Timestamp(test_dates[0]).date()),
                     "test_end": str(pd.Timestamp(test_dates[-1]).date())}
        report["folds"].append(fold_info)
        for market, target in TARGETS.items():
            features = market_features(df, market)
            tr = train.dropna(subset=[target]).copy()
            te = test.dropna(subset=[target]).copy()
            tr = tr[tr.GAMES_PLAYED_BEFORE >= 2]
            te = te[te.GAMES_PLAYED_BEFORE >= 2]
            if market == "points":
                tr = tr[(tr.MIN > 0) & tr.PTS.notna()]
                te = te[(te.MIN > 0) & te.PTS.notna()]
            if len(tr) < 1000 or len(te) < 100:
                raise RuntimeError(f"{market} fold {fold_no+1}: insufficient rows train={len(tr)} test={len(te)}")
            if market == "points":
                candidate, _, scored = fit_points_v2(tr, te, features)
                candidate_pred = candidate["model"].predict(scored[features])
            else:
                candidate_model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("regressor", HistGradientBoostingRegressor(
                        max_iter=300, learning_rate=.045, max_leaf_nodes=15,
                        l2_regularization=2.5, random_state=42 + fold_no))
                ])
                candidate_model.fit(tr[features], tr[target])
                scored = te
                candidate_pred = candidate_model.predict(scored[features])
            y = scored[target].to_numpy()
            candidate_score = score(y, candidate_pred)
            # Last-five and season-to-date naive forecasts are mandatory sanity checks.
            l5_col = f"{target}_L5"
            season_col = f"{target}_SEASON_PRE"
            fallback = float(tr[target].median())
            l5 = scored[l5_col].fillna(scored[season_col]).fillna(fallback).to_numpy()
            season = scored[season_col].fillna(fallback).to_numpy()
            naive_scores = {"last5_or_season": score(y, l5), "season_average": score(y, season)}
            incumbent_score = None
            incumbent_path = MODELS / f"{market}_model.joblib"
            if incumbent_path.exists():
                try:
                    incumbent = joblib.load(incumbent_path)
                    old_features = incumbent.get("feature_cols", features)
                    inc_pred = incumbent["model"].predict(scored[old_features])
                    incumbent_score = score(y, inc_pred)
                except Exception as exc:
                    incumbent_score = {"error": str(exc)}
            # Error diagnostics by actual playing-time band and recent role trend.
            minute = scored["MIN"].to_numpy()
            bands = {"under_15": minute < 15, "15_to_24": (minute >= 15) & (minute < 25),
                     "25_to_34": (minute >= 25) & (minute < 35), "35_plus": minute >= 35}
            subgroup = {}
            abs_err = np.abs(candidate_pred - y)
            for name, mask in bands.items():
                if int(mask.sum()) >= 30:
                    subgroup[name] = {"rows": int(mask.sum()), "mae": float(abs_err[mask].mean())}
            entry = {"market": market, "fold": fold_no + 1,
                     "train_rows": int(len(tr)), "test_rows": int(len(scored)),
                     "features": int(len(features)), "candidate": candidate_score,
                     "naive_baselines": naive_scores, "incumbent": incumbent_score,
                     "candidate_mae_by_actual_minutes": subgroup}
            report.setdefault("fold_results", []).append(entry)
            report["markets"].setdefault(market, []).append(entry)
            for idx, (_, row) in enumerate(scored.iterrows()):
                detail_rows.append({"market": market, "fold": fold_no + 1,
                    "game_date": str(pd.Timestamp(row.GAME_DATE).date()),
                    "player": row.PLAYER_NAME, "actual": float(y[idx]),
                    "candidate": float(candidate_pred[idx]),
                    "error": float(candidate_pred[idx] - y[idx]),
                    "abs_error": float(abs_err[idx]),
                    "actual_minutes": float(row.MIN)})
    for market, entries in report["markets"].items():
        report["markets"][market] = {
            "folds": len(entries),
            "mean_candidate_mae": float(np.mean([e["candidate"]["mae"] for e in entries])),
            "mean_candidate_rmse": float(np.mean([e["candidate"]["rmse"] for e in entries])),
            "mean_last5_baseline_mae": float(np.mean([e["naive_baselines"]["last5_or_season"]["mae"] for e in entries])),
            "mean_season_baseline_mae": float(np.mean([e["naive_baselines"]["season_average"]["mae"] for e in entries])),
            "mean_incumbent_mae": (float(np.mean([e["incumbent"]["mae"] for e in entries
                if e["incumbent"] and "mae" in e["incumbent"]]))
                if any(e["incumbent"] and "mae" in e["incumbent"] for e in entries) else None),
            "fold_detail": entries}
    MODELS.mkdir(exist_ok=True)
    (MODELS / "multi_market_audit.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(detail_rows).to_csv(MODELS / "multi_market_audit_errors.csv", index=False)
    print(json.dumps(report, indent=2))
    print(f"Saved audit report and {len(detail_rows)} player-level error rows.")


if __name__ == "__main__":
    main()
