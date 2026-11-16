"""Leakage-safe chronological experiment for historical UFC performance features."""
from collections import defaultdict, deque
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ufc_model

BASE_URL = "https://raw.githubusercontent.com/Sam-Armstrong/UFC-Almanac/main/data/"
STAT_COLS = {
    "kd": "Knockdowns", "sig_landed": "Sig Strikes Landed",
    "sig_attempted": "Sig Strikes Attempted", "sig_absorbed": "Sig Strikes Absorbed",
    "td": "Takedowns", "td_attempted": "Takedown Attempts",
    "sub_attempts": "Submission Attempts", "clinch": "Clinch Strikes",
    "ground": "Ground Strikes", "td_against": "Got Taken Down",
}
RATE_KEYS = tuple(STAT_COLS)


def fighter_key(value):
    return ufc_model._fighter_key(value)


def build_past_performance(results, stats, base_frame):
    """Each bout receives aggregates from strictly earlier calendar dates."""
    s = stats.copy()
    s["_date"] = pd.to_datetime(s["Date"], dayfirst=True, errors="coerce").dt.normalize()
    s["_fighter"] = s["Name"].map(fighter_key)
    for key, col in STAT_COLS.items():
        s[key] = pd.to_numeric(s[col], errors="coerce") if col in s else 0.0
    s["_seconds"] = pd.to_numeric(s["Time"], errors="coerce").clip(lower=1)
    s = s[s["_date"].notna() & s["_fighter"].ne("") & s["_seconds"].notna()]
    # Rates are normalized by elapsed minutes, preventing long fights dominating solely by duration.
    for key in RATE_KEYS:
        s[key + "_pm"] = s[key] / (s["_seconds"] / 60.0).clip(lower=1.0)
    # Same-day duplicates cannot safely be assigned to a particular bout.
    s = s.drop_duplicates(["_date", "_fighter"], keep=False)
    by_date = {d: g for d, g in s.groupby("_date")}
    history = defaultdict(lambda: deque(maxlen=10))
    rows = []
    for day, group in base_frame.groupby("date", sort=True):
        for _, bout in group.iterrows():
            pair = []
            for fighter in (fighter_key(bout["fighter1"]), fighter_key(bout["fighter2"])):
                past = list(history[fighter])
                values = {}
                for key in RATE_KEYS:
                    seq = [float(item[key + "_pm"]) for item in past if pd.notna(item[key + "_pm"])]
                    values[key] = float(np.mean(seq)) if seq else np.nan
                pair.append(values)
            row = {}
            for key in RATE_KEYS:
                a, b = pair
                row[key + "_mean"] = np.nanmean([a[key], b[key]]) if not (np.isnan(a[key]) and np.isnan(b[key])) else np.nan
                row[key + "_diff"] = a[key] - b[key] if np.isfinite(a[key]) and np.isfinite(b[key]) else np.nan
            rows.append(row)
        # Update history only after all bouts on the event date have been featurised.
        today = by_date.get(day, pd.DataFrame())
        for _, stat in today.iterrows():
            history[stat["_fighter"]].append({key + "_pm": stat[key + "_pm"] for key in RATE_KEYS})
    features = pd.DataFrame(rows, index=base_frame.index)
    return features


def evaluate(frame, features, train_idx, cal_idx, test_idx):
    train, cal, test = frame.loc[train_idx], frame.loc[cal_idx], frame.loc[test_idx]
    candidates, scores = [], []
    for c_value in (0.10, 0.25, 0.50, 1.0, 2.0):
        candidate = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2500, class_weight="balanced", C=c_value)),
        ])
        candidate.fit(train[features], train["target"])
        p = np.clip(candidate.predict_proba(cal[features])[:, 1], 1e-6, 1-1e-6)
        scores.append({"C": c_value, "brier": float(brier_score_loss(cal["target"], p)),
                       "log_loss": float(log_loss(cal["target"], p, labels=[0,1]))})
        candidates.append(candidate)
    incumbent = next(i for i, row in enumerate(scores) if row["C"] == .50)
    best = min(range(len(scores)), key=lambda i: (scores[i]["brier"], scores[i]["log_loss"]))
    selected = best if (scores[best]["brier"] < scores[incumbent]["brier"]
                        and scores[best]["log_loss"] < scores[incumbent]["log_loss"]) else incumbent
    model = candidates[selected]
    raw_cal = np.clip(model.predict_proba(cal[features])[:, 1], 1e-6, 1-1e-6)
    logits = np.log(raw_cal / (1-raw_cal)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(logits, cal["target"])
    raw_test = np.clip(model.predict_proba(test[features])[:, 1], 1e-6, 1-1e-6)
    test_logits = np.log(raw_test / (1-raw_test)).reshape(-1, 1)
    probs = calibrator.predict_proba(test_logits)[:, 1]
    pred = (probs >= .5).astype(int)
    return {
        "selected_c": float(scores[selected]["C"]),
        "calibration_candidates": scores,
        "accuracy": float(accuracy_score(test["target"], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test["target"], pred)),
        "brier": float(brier_score_loss(test["target"], probs)),
        "log_loss": float(log_loss(test["target"], probs, labels=[0,1])),
        "test_fights": int(len(test)),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
    }


def main():
    results = pd.read_csv(BASE_URL + "FightResults.csv")
    stats = pd.read_csv(BASE_URL + "FightStats.csv")
    base, _ = ufc_model.build_chronological_rows(results)
    base = base.reset_index(drop=True)
    extra = build_past_performance(results, stats, base).reset_index(drop=True)
    combined = pd.concat([base, extra], axis=1)
    dates = pd.Index(combined["date"].drop_duplicates().sort_values())
    train_end = max(1, int(len(dates)*.70))
    cal_end = min(max(train_end+1, int(len(dates)*.80)), len(dates)-1)
    train_cut, cal_cut = dates[train_end-1], dates[cal_end-1]
    train_idx = combined.index[combined.date <= train_cut]
    cal_idx = combined.index[(combined.date > train_cut) & (combined.date <= cal_cut)]
    test_idx = combined.index[combined.date > cal_cut]
    base_features = list(ufc_model.FEATURES)
    extra_features = base_features + list(extra.columns)
    baseline = evaluate(combined, base_features, train_idx, cal_idx, test_idx)
    candidate = evaluate(combined, extra_features, train_idx, cal_idx, test_idx)
    passed = candidate["brier"] < baseline["brier"] and candidate["log_loss"] < baseline["log_loss"]
    report = {
        "experiment": "past-only rolling performance rates with calibration-period C selection",
        "feature_window": "last 10 prior fights per fighter",
        "same_chronological_split": True,
        "same_test_rows": baseline["test_fights"] == candidate["test_fights"],
        "baseline": baseline, "candidate": candidate,
        "improved_both_brier_and_log_loss": bool(passed),
        "promotion": "candidate eligible for review" if passed else "retain existing winner model",
        "features": extra_features,
        "note": "No current-bout statistics or current profile snapshot values are used as predictors. All same-day bouts are featurised before any same-day statistics update fighter histories."
    }
    out = ROOT / "artifacts" / "ufc_performance_experiment.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["same_test_rows"] or not baseline["test_fights"]:
        raise SystemExit("Invalid paired experiment")
if __name__ == "__main__":
    main()
