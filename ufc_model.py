"""Leakage-safe chronological UFC winner and method validation."""
from collections import defaultdict, deque
import re
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "elo_diff", "wins_diff", "losses_diff", "experience_diff",
    "win_rate_diff", "recent5_diff", "recent10_diff",
    "streak_diff", "days_since_fight_diff",
]

def _state_features(state, day):
    dates = state["dates"]
    days = (day - dates[-1]).days if dates else 730
    return {
        "elo": state["elo"], "wins": state["wins"], "losses": state["losses"],
        "experience": state["wins"] + state["losses"],
        "win_rate": state["wins"] / max(1, state["wins"] + state["losses"]),
        "recent5": float(np.mean(state["recent5"])) if state["recent5"] else 0.5,
        "recent10": float(np.mean(state["recent10"])) if state["recent10"] else 0.5,
        "streak": state["streak"], "days_since_fight": min(days, 730),
    }

def _method_group(value):
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if any(x in s for x in ("ko", "tko", "technical knockout", "doctor stoppage")):
        return "KO/TKO"
    if any(x in s for x in ("submission", "choke", "kimura", "armbar", "heel hook", "guillotine")):
        return "Submission"
    if any(x in s for x in ("decision", "unanimous", "split", "majority")):
        return "Decision"
    return "Other"

def build_chronological_rows(results):
    required = {"Date", "Fighter 1", "Fighter 2", "Result"}
    if not required.issubset(results.columns):
        raise ValueError(f"Results data missing columns: {sorted(required - set(results.columns))}")
    df = results.copy()
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["result"] = pd.to_numeric(df["Result"], errors="coerce")
    df["fighter1"] = df["Fighter 1"].astype(str).str.strip()
    df["fighter2"] = df["Fighter 2"].astype(str).str.strip()
    method_col = next((x for x in ("Method", "method") if x in df.columns), None)
    division_col = next((x for x in ("Weight Class", "weight_class", "Division", "division") if x in df.columns), None)
    df["method_group"] = df[method_col].map(_method_group) if method_col else None
    df["division"] = df[division_col].astype(str).str.strip() if division_col else "Not supplied"
    df = df.dropna(subset=["date", "result"])
    df = df[(df["fighter1"] != "") & (df["fighter2"] != "") & (df["fighter1"] != df["fighter2"])]
    df = df[df["result"].isin([1, 2, 3])].sort_values(["date"], kind="stable")
    states = defaultdict(lambda: {
        "elo": 1500.0, "wins": 0, "losses": 0, "streak": 0,
        "recent5": deque(maxlen=5), "recent10": deque(maxlen=10), "dates": [],
    })
    rows = []
    for day, group in df.groupby("date", sort=True):
        pending = []
        for _, fight in group.iterrows():
            a, b = fight["fighter1"], fight["fighter2"]
            sa, sb = _state_features(states[a], day), _state_features(states[b], day)
            rows.append({
                "date": day,
                "elo_diff": sa["elo"] - sb["elo"],
                "wins_diff": sa["wins"] - sb["wins"],
                "losses_diff": sa["losses"] - sb["losses"],
                "experience_diff": sa["experience"] - sb["experience"],
                "win_rate_diff": sa["win_rate"] - sb["win_rate"],
                "recent5_diff": sa["recent5"] - sb["recent5"],
                "recent10_diff": sa["recent10"] - sb["recent10"],
                "streak_diff": sa["streak"] - sb["streak"],
                "days_since_fight_diff": sa["days_since_fight"] - sb["days_since_fight"],
                "target": int(fight["result"] == 1),
                "outcome": int(fight["result"]),
                "method_group": fight["method_group"],
                "division": fight["division"],
                "fighter1": a, "fighter2": b,
            })
            pending.append((a, b, int(fight["result"])))
        for a, b, outcome in pending:
            if outcome == 3:
                continue
            sa, sb = states[a], states[b]
            ya = 1 if outcome == 1 else 0
            expected_a = 1 / (1 + 10 ** ((sb["elo"] - sa["elo"]) / 400))
            change = 32 * (ya - expected_a)
            sa["elo"] += change
            sb["elo"] -= change
            for state, won in ((sa, ya), (sb, 1 - ya)):
                if won:
                    state["wins"] += 1
                    state["streak"] = max(0, state["streak"]) + 1
                else:
                    state["losses"] += 1
                    state["streak"] = min(0, state["streak"]) - 1
                state["recent5"].append(won)
                state["recent10"].append(won)
                state["dates"].append(day)
    frame = pd.DataFrame(rows)
    # Draws are not wins for either side and must be excluded from binary training.
    frame = frame[frame["outcome"].isin([1, 2])].drop(columns=["outcome"]).reset_index(drop=True)
    return frame, states

def train_and_evaluate(results):
    frame, states = build_chronological_rows(results)
    if len(frame) < 250:
        raise ValueError(f"Only {len(frame)} usable decisive fights; at least 250 are required.")
    # Three chronological partitions: fit, calibrate, then final untouched test.
    n = len(frame)
    train_end = int(n * 0.70)
    calibration_end = int(n * 0.80)
    train = frame.iloc[:train_end].copy()
    calibration = frame.iloc[train_end:calibration_end].copy()
    test = frame.iloc[calibration_end:].copy()
    if any(part["target"].nunique() < 2 for part in (train, calibration, test)):
        raise ValueError("Chronological train/calibration/test split does not contain both winner classes.")

    base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
    ])
    base_model.fit(train[FEATURES], train["target"])

    # Platt scaling: learn a probability correction on the separate calibration period.
    calibration_raw = np.clip(base_model.predict_proba(calibration[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    calibration_logits = np.log(calibration_raw / (1 - calibration_raw)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs")
    calibrator.fit(calibration_logits, calibration["target"])

    class CalibratedProbabilityModel:
        def __init__(self, estimator, probability_calibrator):
            self.estimator = estimator
            self.probability_calibrator = probability_calibrator

        def predict_proba(self, X):
            raw = np.clip(self.estimator.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
            logits = np.log(raw / (1 - raw)).reshape(-1, 1)
            calibrated = self.probability_calibrator.predict_proba(logits)[:, 1]
            return np.column_stack([1 - calibrated, calibrated])

    model = CalibratedProbabilityModel(base_model, calibrator)
    raw_probabilities = base_model.predict_proba(test[FEATURES])[:, 1]
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    baseline = DummyClassifier(strategy="prior").fit(train[FEATURES], train["target"])
    base_probs = baseline.predict_proba(test[FEATURES])[:, 1]
    metrics = {
        "train_fights": int(len(train)), "calibration_fights": int(len(calibration)),
        "test_fights": int(len(test)),
        "calibration_start": str(calibration["date"].min().date()),
        "calibration_end": str(calibration["date"].max().date()),
        "test_start": str(test["date"].min().date()), "test_end": str(test["date"].max().date()),
        "accuracy": float(accuracy_score(test["target"], predictions)),
        "raw_accuracy": float(accuracy_score(test["target"], (raw_probabilities >= 0.5).astype(int))),
        "baseline_accuracy": float(accuracy_score(test["target"], (base_probs >= 0.5).astype(int))),
        "log_loss": float(log_loss(test["target"], probabilities, labels=[0, 1])),
        "raw_log_loss": float(log_loss(test["target"], raw_probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(test["target"], probabilities)),
        "raw_brier": float(brier_score_loss(test["target"], raw_probabilities)),
        "baseline_brier": float(brier_score_loss(test["target"], base_probs)),
    }

    # Threshold audit: accuracy and coverage on the untouched test period.
    threshold_results = []
    confidence = np.maximum(probabilities, 1 - probabilities)
    correct = ((probabilities >= 0.5).astype(int) == test["target"].to_numpy())
    for threshold in (0.55, 0.60, 0.65, 0.70):
        mask = confidence >= threshold
        if mask.any():
            threshold_results.append({
                "threshold": threshold, "fights": int(mask.sum()),
                "coverage": float(mask.mean()),
                "accuracy": float(correct[mask].mean()),
            })
    metrics["thresholds"] = threshold_results

    # Reliability table: compare mean predicted probability with observed win rate.
    calibration = []
    bins = pd.cut(probabilities, bins=[0, .4, .5, .6, .7, .8, 1.000001],
                  labels=["<40%", "40-50%", "50-60%", "60-70%", "70-80%", "80%+"])
    cal_frame = pd.DataFrame({"band": bins.astype(str), "prob": probabilities, "actual": test["target"].to_numpy()})
    for band, group in cal_frame.groupby("band", observed=False):
        if len(group):
            calibration.append({"band": str(band), "fights": int(len(group)),
                                "predicted": float(group["prob"].mean()),
                                "observed": float(group["actual"].mean())})

    # Chronological walk-forward-style reporting over four consecutive holdout slices.
    temporal = []
    for idx, positions in enumerate(np.array_split(np.arange(len(test)), 4), start=1):
        if len(positions):
            chunk = test.iloc[positions]
            pp = probabilities[positions]
            temporal.append({"period": idx, "fights": int(len(chunk)),
                             "accuracy": float(accuracy_score(chunk["target"], (pp >= .5).astype(int))),
                             "start": str(chunk["date"].min().date()), "end": str(chunk["date"].max().date())})

    # Division breakdown is only reported when the source actually provides a division field.
    division = []
    if (test["division"] != "Not supplied").any():
        temp = test.copy()
        temp["pred"] = predictions
        for name, group in temp.groupby("division"):
            if len(group) >= 30:
                division.append({"division": str(name), "fights": int(len(group)),
                                 "accuracy": float(accuracy_score(group["target"], group["pred"]))})

    # Method-of-victory classifier, trained only on the training-era bouts.
    method_metrics = {"available": False, "reason": "The source has no usable method labels."}
    method_train = train[train["method_group"].notna()]
    method_test = test[test["method_group"].notna()]
    if len(method_train) >= 250 and method_train["method_group"].nunique() >= 2:
        seen = set(method_train["method_group"].unique())
        method_test = method_test[method_test["method_group"].isin(seen)]
        if len(method_test) >= 30:
            method_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2500, class_weight="balanced")),
            ])
            method_model.fit(method_train[FEATURES], method_train["method_group"])
            mp = method_model.predict(method_test[FEATURES])
            from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
            actual_methods = method_test["method_group"]
            majority_class = method_train["method_group"].value_counts().idxmax()
            report = classification_report(
                actual_methods, mp, labels=sorted(seen), output_dict=True, zero_division=0
            )
            method_metrics = {
                "available": True, "train_fights": int(len(method_train)),
                "test_fights": int(len(method_test)),
                "accuracy": float(accuracy_score(actual_methods, mp)),
                "majority_baseline_accuracy": float((actual_methods == majority_class).mean()),
                "majority_class": str(majority_class),
                "balanced_accuracy": float(balanced_accuracy_score(actual_methods, mp)),
                "macro_f1": float(f1_score(actual_methods, mp, labels=sorted(seen), average="macro", zero_division=0)),
                "classes": sorted(seen),
                "class_counts": actual_methods.value_counts().to_dict(),
                "per_class": {
                    name: {"precision": float(report[name]["precision"]),
                           "recall": float(report[name]["recall"]),
                           "f1": float(report[name]["f1-score"]),
                           "support": int(report[name]["support"])}
                    for name in sorted(seen)
                },
            }

    metrics["calibration"] = calibration
    metrics["temporal"] = temporal
    metrics["division"] = division
    metrics["division_available"] = bool(division)
    metrics["method"] = method_metrics
    # Keep the calibrated estimator trained on the pre-test era; do not refit it
    # on test outcomes, which would invalidate the reported validation.
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    snapshot = {name: _state_features(state, today) for name, state in states.items()}
    return model, metrics, snapshot, frame
