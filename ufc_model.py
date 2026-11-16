"""Leakage-safe chronological UFC winner and method validation."""
from collections import defaultdict, deque
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def _fighter_key(value):
    """Normalize Unicode, case and punctuation; never fuzzy-merge identities."""
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = "".join(ch for ch in value if ch.isalnum() or ch.isspace())
    return " ".join(value.split())


def audit_results(results):
    """Source-quality counts. Invalid records are reported, not silently repaired."""
    required = {"Date", "Fighter 1", "Fighter 2", "Result"}
    missing = sorted(required - set(results.columns))
    if missing:
        return {"missing_columns": missing}
    dates = pd.to_datetime(results["Date"], dayfirst=True, errors="coerce")
    outcomes = pd.to_numeric(results["Result"], errors="coerce")
    a = results["Fighter 1"].fillna("").astype(str).str.strip().map(_fighter_key)
    b = results["Fighter 2"].fillna("").astype(str).str.strip().map(_fighter_key)
    pairs = pd.DataFrame({"date": dates, "pair": [tuple(sorted(x)) for x in zip(a, b)]})
    appearances = pd.concat([
        pd.DataFrame({"date": dates, "fighter": a}),
        pd.DataFrame({"date": dates, "fighter": b}),
    ])
    appearances = appearances[(appearances["date"].notna()) & (appearances["fighter"] != "")]
    repeats = appearances.groupby(["date", "fighter"]).size()
    usable = dates.notna() & outcomes.isin([1, 2, 3]) & a.ne("") & b.ne("") & a.ne(b)
    return {
        "rows": int(len(results)),
        "invalid_dates": int(dates.isna().sum()),
        "invalid_results": int((~outcomes.isin([1, 2, 3])).sum()),
        "blank_names": int((a.eq("") | b.eq("")).sum()),
        "self_matchups": int((a == b).sum()),
        "duplicate_bout_rows": int(pairs.duplicated().sum()),
        "same_day_repeat_fighters": int((repeats > 1).sum()),
        "usable_rows": int(usable.sum()),
    }


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
    df["fighter1_key"] = df["fighter1"].map(_fighter_key)
    df["fighter2_key"] = df["fighter2"].map(_fighter_key)
    method_col = next((x for x in ("Method", "method") if x in df.columns), None)
    division_col = next((x for x in ("Weight Class", "weight_class", "Division", "division") if x in df.columns), None)
    df["method_group"] = df[method_col].map(_method_group) if method_col else None
    df["division"] = df[division_col].astype(str).str.strip() if division_col else "Not supplied"
    df = df.dropna(subset=["date", "result"])
    df = df[(df["fighter1_key"] != "") & (df["fighter2_key"] != "") & (df["fighter1_key"] != df["fighter2_key"])]
    df = df[df["result"].isin([1, 2, 3])].sort_values(["date"], kind="stable")
    states = defaultdict(lambda: {
        "elo": 1500.0, "wins": 0, "losses": 0, "streak": 0,
        "recent5": deque(maxlen=5), "recent10": deque(maxlen=10), "dates": [],
    })
    rows = []
    for day, group in df.groupby("date", sort=True):
        pending = []
        for _, fight in group.iterrows():
            a, b = fight["fighter1_key"], fight["fighter2_key"]
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
                "fighter1": fight["fighter1"], "fighter2": fight["fighter2"],
            })
            pending.append((a, b, int(fight["result"])))
        # All Elo changes use the same pre-event ratings, then apply together.
        elo_deltas = defaultdict(float)
        for a, b, outcome in pending:
            if outcome == 3:
                continue
            sa, sb = states[a], states[b]
            ya = 1 if outcome == 1 else 0
            expected_a = 1 / (1 + 10 ** ((sb["elo"] - sa["elo"]) / 400))
            change = 32 * (ya - expected_a)
            elo_deltas[a] += change
            elo_deltas[b] -= change
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
        for fighter, delta in elo_deltas.items():
            states[fighter]["elo"] += delta
    frame = pd.DataFrame(rows)
    # Draws are not wins for either side and must be excluded from binary training.
    frame = frame[frame["outcome"].isin([1, 2])].drop(columns=["outcome"]).reset_index(drop=True)
    return frame, states

def train_and_evaluate(results):
    frame, states = build_chronological_rows(results)
    if len(frame) < 250:
        raise ValueError(f"Only {len(frame)} usable decisive fights; at least 250 are required.")
    # Split on whole event dates, never individual fight rows. This prevents
    # bouts from the same card leaking across fit/calibration/test partitions.
    event_dates = pd.Index(frame["date"].drop_duplicates().sort_values())
    if len(event_dates) < 10:
        raise ValueError("Not enough distinct event dates for a reliable chronological split.")
    train_date_end = max(1, int(len(event_dates) * 0.70))
    calibration_date_end = max(train_date_end + 1, int(len(event_dates) * 0.80))
    calibration_date_end = min(calibration_date_end, len(event_dates) - 1)
    train_cutoff = event_dates[train_date_end - 1]
    calibration_cutoff = event_dates[calibration_date_end - 1]
    train = frame[frame["date"] <= train_cutoff].copy()
    calibration = frame[(frame["date"] > train_cutoff) & (frame["date"] <= calibration_cutoff)].copy()
    test = frame[frame["date"] > calibration_cutoff].copy()
    if train.empty or calibration.empty or test.empty:
        raise ValueError("The event-date split produced an empty partition.")
    if any(part["target"].nunique() < 2 for part in (train, calibration, test)):
        raise ValueError("Chronological train/calibration/test split does not contain both winner classes.")

    # Select a challenger using the later calibration period, then retain a
    # completely untouched final test period for the reported performance.
    candidate_rows = []
    candidates = []
    for candidate_c in (0.10, 0.25, 0.50, 1.0, 2.0):
        candidate = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", C=candidate_c
            )),
        ])
        candidate.fit(train[FEATURES], train["target"])
        candidate_probs = np.clip(
            candidate.predict_proba(calibration[FEATURES])[:, 1], 1e-6, 1 - 1e-6
        )
        candidate_rows.append({
            "C": candidate_c,
            "calibration_brier": float(brier_score_loss(calibration["target"], candidate_probs)),
            "calibration_log_loss": float(log_loss(calibration["target"], candidate_probs, labels=[0, 1])),
        })
        candidates.append(candidate)

    incumbent_index = next(i for i, row in enumerate(candidate_rows) if row["C"] == 0.50)
    challenger_index = min(
        range(len(candidate_rows)),
        key=lambda i: (candidate_rows[i]["calibration_brier"], candidate_rows[i]["calibration_log_loss"]),
    )
    incumbent_score = candidate_rows[incumbent_index]
    challenger_score = candidate_rows[challenger_index]
    challenger_improves_both = (
        challenger_score["calibration_brier"] < incumbent_score["calibration_brier"]
        and challenger_score["calibration_log_loss"] < incumbent_score["calibration_log_loss"]
    )
    selected_index = challenger_index if challenger_improves_both else incumbent_index
    selected_c = candidate_rows[selected_index]["C"]
    selection_reason = (
        "Challenger improved both calibration Brier and log loss."
        if challenger_improves_both
        else "Kept incumbent C=0.50 because no challenger improved both calibration metrics."
    )
    base_model = candidates[selected_index]

    # Platt scaling: fit the probability correction on the later calibration period.
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
    elo_probs = 1 / (1 + np.power(10.0, -test["elo_diff"].to_numpy() / 400.0))
    elo_predictions = (elo_probs >= 0.5).astype(int)
    base_predictions = (base_probs >= 0.5).astype(int)
    metrics = {
        "selected_c": float(selected_c),
        "selection_reason": selection_reason,
        "candidate_comparison": candidate_rows,
        "elo_accuracy": float(accuracy_score(test["target"], elo_predictions)),
        "elo_balanced_accuracy": float(balanced_accuracy_score(test["target"], elo_predictions)),
        "elo_log_loss": float(log_loss(test["target"], elo_probs, labels=[0, 1])),
        "elo_brier": float(brier_score_loss(test["target"], elo_probs)),
        "balanced_accuracy": float(balanced_accuracy_score(test["target"], predictions)),
        "baseline_balanced_accuracy": float(balanced_accuracy_score(test["target"], base_predictions)),
        "baseline_log_loss": float(log_loss(test["target"], base_probs, labels=[0, 1])),
        "train_fights": int(len(train)), "calibration_fights": int(len(calibration)),
        "test_fights": int(len(test)),
        "train_start": str(train["date"].min().date()),
        "train_end": str(train["date"].max().date()),
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
        # Stored only for the in-app reliability table; these are untouched test rows.
        "test_probabilities": probabilities.astype(float).tolist(),
        "test_targets": test["target"].astype(int).tolist(),
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

    # Method labels are retained for future research, but the current
    # classifier is deliberately disabled: prior validation was materially
    # worse than a majority-class baseline.
    method_metrics = {
        "available": False,
        "reason": "Disabled after failing majority-baseline and balanced-accuracy checks. Rebuild only after adding verified pre-fight attributes and a new chronological validation.",
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


def predict_matchups(model, snapshot, matchups):
    """Predict every supplied matchup, including low-confidence and debutant bouts."""
    rows = []
    names = ["elo", "wins", "losses", "experience", "win_rate", "recent5", "recent10", "streak", "days_since_fight"]
    neutral = {"elo": 1500.0, "wins": 0, "losses": 0, "experience": 0, "win_rate": 0.5, "recent5": 0.5, "recent10": 0.5, "streak": 0, "days_since_fight": 730}
    for bout in matchups:
        fighter1, fighter2 = str(bout.get("Fighter 1", "")).strip(), str(bout.get("Fighter 2", "")).strip()
        if not fighter1 or not fighter2:
            continue
        key1, key2 = _fighter_key(fighter1), _fighter_key(fighter2)
        a, b = snapshot.get(key1, neutral), snapshot.get(key2, neutral)
        values = {name + "_diff": float(a[name]) - float(b[name]) for name in names}
        probability1 = float(model.predict_proba(pd.DataFrame([values], columns=FEATURES))[0, 1])
        probability1 = min(1.0, max(0.0, probability1))
        confidence = max(probability1, 1.0 - probability1)
        missing = [fighter for fighter, key in ((fighter1, key1), (fighter2, key2)) if key not in snapshot]
        rows.append({"Bout": fighter1 + " vs " + fighter2, "Predicted winner": fighter1 if probability1 >= 0.5 else fighter2, fighter1 + " win %": round(probability1 * 100, 1), fighter2 + " win %": round((1.0 - probability1) * 100, 1), "Confidence": round(confidence * 100, 1), "Data warning": ("No historical profile: " + ", ".join(missing)) if missing else "Both fighters have historical profiles"})
    return pd.DataFrame(rows)


# Independent method-of-victory and fight-duration models. These use only
# fighter history available before each bout; current-fight statistics are
# used only to create labels, never as predictors.
AUX_METHOD_CLASSES = ("KO/TKO", "Submission", "Decision")
AUX_PERFORMANCE_FEATURES = (
    "knockdown_rate", "sig_landed_rate", "sig_absorbed_rate",
    "takedown_rate", "submission_attempt_rate", "ground_strike_rate",
    "clinch_strike_rate", "sig_accuracy", "takedown_accuracy",
)
AUX_METHOD_FEATURES = [
    "experience_mean", "experience_max",
    "ko_win_mean", "ko_win_max", "sub_win_mean", "sub_win_max",
    "decision_win_mean", "decision_win_max",
    "ko_loss_mean", "ko_loss_max", "sub_loss_mean", "sub_loss_max",
    "decision_loss_mean", "decision_loss_max",
    "duration_mean", "duration_max", "recent_duration_mean",
    "finish_rate_mean", "days_since_mean",
] + [
    feature + suffix
    for feature in AUX_PERFORMANCE_FEATURES
    for suffix in ("_mean", "_max", "_diff_abs")
]
AUX_ROUND_LINES = {"1.5": 450.0, "2.5": 750.0}


def _aux_state():
    return {
        "fights": 0, "method_wins": defaultdict(int),
        "method_losses": defaultdict(int), "durations": deque(maxlen=10),
        "performance": deque(maxlen=10), "dates": deque(maxlen=20),
    }


def _aux_rate(count, total, classes=3):
    return float((count + 1) / (total + classes))


def _aux_fighter_features(state, day):
    fights = state["fights"]
    wins_total = sum(state["method_wins"].values())
    losses_total = sum(state["method_losses"].values())
    duration_values = list(state["durations"])
    days_since = (day - state["dates"][-1]).days if state["dates"] else 730
    return {
        "experience": fights,
        "ko_win": _aux_rate(state["method_wins"]["KO/TKO"], wins_total),
        "sub_win": _aux_rate(state["method_wins"]["Submission"], wins_total),
        "decision_win": _aux_rate(state["method_wins"]["Decision"], wins_total),
        "ko_loss": _aux_rate(state["method_losses"]["KO/TKO"], losses_total),
        "sub_loss": _aux_rate(state["method_losses"]["Submission"], losses_total),
        "decision_loss": _aux_rate(state["method_losses"]["Decision"], losses_total),
        "duration": float(np.mean(duration_values)) if duration_values else 600.0,
        "recent_duration": float(np.mean(duration_values[-5:])) if duration_values else 600.0,
        "finish_rate": _aux_rate(
            sum(state["method_wins"][x] + state["method_losses"][x]
                for x in ("KO/TKO", "Submission")),
            fights,
        ),
        "days_since": min(days_since, 730),
        **{
            key: float(np.mean([item.get(key, 0.0) for item in state["performance"]]))
            if state["performance"] else 0.0
            for key in AUX_PERFORMANCE_FEATURES
        },
    }


def _aux_pair_features(state1, state2, day):
    a, b = _aux_fighter_features(state1, day), _aux_fighter_features(state2, day)
    values = {
        "experience_mean": (a["experience"] + b["experience"]) / 2,
        "experience_max": max(a["experience"], b["experience"]),
        "duration_mean": (a["duration"] + b["duration"]) / 2,
        "duration_max": max(a["duration"], b["duration"]),
        "recent_duration_mean": (a["recent_duration"] + b["recent_duration"]) / 2,
        "finish_rate_mean": (a["finish_rate"] + b["finish_rate"]) / 2,
        "days_since_mean": (a["days_since"] + b["days_since"]) / 2,
    }
    for key in ("ko_win", "sub_win", "decision_win", "ko_loss", "sub_loss", "decision_loss"):
        values[f"{key}_mean"] = (a[key] + b[key]) / 2
        values[f"{key}_max"] = max(a[key], b[key])
    for key in AUX_PERFORMANCE_FEATURES:
        values[f"{key}_mean"] = (a[key] + b[key]) / 2
        values[f"{key}_max"] = max(a[key], b[key])
        values[f"{key}_diff_abs"] = abs(a[key] - b[key])
    return {key: float(values[key]) for key in AUX_METHOD_FEATURES}


def _aux_duration_lookup(results, fight_stats):
    """Join post-fight elapsed time and performance by date/fighter for labels and history."""
    required = {"Date", "Name", "Time"}
    if fight_stats is None or not required.issubset(fight_stats.columns):
        return {}
    stats = fight_stats.copy()
    stats["_date"] = pd.to_datetime(stats["Date"], dayfirst=True, errors="coerce").dt.normalize()
    stats["_fighter"] = stats["Name"].map(_fighter_key)
    stats["_seconds"] = pd.to_numeric(stats["Time"], errors="coerce")
    numeric_columns = (
        "Knockdowns", "Sig Strikes Landed", "Sig Strikes Attempted",
        "Sig Strikes Absorbed", "Takedowns", "Takedown Attempts",
        "Submission Attempts", "Clinch Strikes", "Ground Strikes",
    )
    for column in numeric_columns:
        stats[column] = pd.to_numeric(stats[column], errors="coerce").fillna(0) if column in stats.columns else 0.0
    stats = stats[
        stats["_date"].notna() & stats["_fighter"].ne("")
        & stats["_seconds"].between(1, 1500)
    ]
    lookup = {}
    for (day, fighter), group in stats.groupby(["_date", "_fighter"]):
        # If a fighter has multiple rows on the same date, the source cannot
        # identify which bout a stat belongs to, so omit it rather than leak.
        if len(group) != 1:
            continue
        row = group.iloc[0]
        seconds = float(row["_seconds"])
        minutes = max(seconds / 60.0, 1.0)
        sig_attempted = float(row["Sig Strikes Attempted"])
        td_attempted = float(row["Takedown Attempts"])
        lookup[(day, fighter)] = {
            "seconds": seconds,
            "performance": {
                "knockdown_rate": float(row["Knockdowns"]) / minutes,
                "sig_landed_rate": float(row["Sig Strikes Landed"]) / minutes,
                "sig_absorbed_rate": float(row["Sig Strikes Absorbed"]) / minutes,
                "takedown_rate": float(row["Takedowns"]) / minutes,
                "submission_attempt_rate": float(row["Submission Attempts"]) / minutes,
                "ground_strike_rate": float(row["Ground Strikes"]) / minutes,
                "clinch_strike_rate": float(row["Clinch Strikes"]) / minutes,
                "sig_accuracy": float(row["Sig Strikes Landed"]) / max(sig_attempted, 1.0),
                "takedown_accuracy": float(row["Takedowns"]) / max(td_attempted, 1.0),
            },
        }
    return lookup

def build_auxiliary_training_rows(results, fight_stats):
    required = {"Date", "Fighter 1", "Fighter 2", "Result", "Method"}
    if not required.issubset(results.columns):
        raise ValueError(f"Auxiliary models require result columns: {sorted(required - set(results.columns))}")
    duration_lookup = _aux_duration_lookup(results, fight_stats)
    df = results.copy()
    df["_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.normalize()
    df["_f1"] = df["Fighter 1"].fillna("").astype(str).map(_fighter_key)
    df["_f2"] = df["Fighter 2"].fillna("").astype(str).map(_fighter_key)
    df["_result"] = pd.to_numeric(df["Result"], errors="coerce")
    df["_method"] = df["Method"].map(_method_group)
    df = df.dropna(subset=["_date", "_result"]).sort_values("_date", kind="stable")
    states = defaultdict(_aux_state)
    rows = []

    for day, group in df.groupby("_date", sort=True):
        pending_updates = []
        for _, fight in group.iterrows():
            f1, f2, result = fight["_f1"], fight["_f2"], int(fight["_result"])
            if not f1 or not f2 or f1 == f2 or result not in (1, 2):
                continue
            method = fight["_method"]
            stats1 = duration_lookup.get((day, f1), {})
            stats2 = duration_lookup.get((day, f2), {})
            duration_values = [
                item["seconds"] for item in (stats1, stats2) if item.get("seconds") is not None
            ]
            duration = float(np.median(duration_values)) if duration_values else np.nan

            # Method labels can be trained from results alone; duration labels
            # require a matched FightStats time value.
            if method in AUX_METHOD_CLASSES:
                row = _aux_pair_features(states[f1], states[f2], day)
                row.update({"date": day, "method_target": method})
                rows.append(row)
                if np.isfinite(duration):
                    rows[-1]["duration_seconds"] = duration
            pending_updates.append((
                f1, f2, method, duration, result,
                stats1.get("performance"), stats2.get("performance"),
            ))

        # Update only after every bout on the event date has been featurised.
        for f1, f2, method, duration, result, performance1, performance2 in pending_updates:
            winner, loser = (f1, f2) if result == 1 else (f2, f1)
            states[winner]["fights"] += 1
            states[loser]["fights"] += 1
            if method in AUX_METHOD_CLASSES:
                states[winner]["method_wins"][method] += 1
                states[loser]["method_losses"][method] += 1
            if np.isfinite(duration):
                states[f1]["durations"].append(duration)
                states[f2]["durations"].append(duration)
            if performance1:
                states[f1]["performance"].append(performance1)
            if performance2:
                states[f2]["performance"].append(performance2)
            states[f1]["dates"].append(day)
            states[f2]["dates"].append(day)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No usable historical method labels were created.")
    snapshot_day = pd.Timestamp.utcnow().tz_localize(None).normalize()
    snapshot = {
        fighter: _aux_fighter_features(state, snapshot_day)
        for fighter, state in states.items()
    }
    return frame, snapshot


def _chronological_aux_split(frame):
    dates = pd.Index(frame["date"].drop_duplicates().sort_values())
    if len(dates) < 10:
        raise ValueError("At least 10 distinct event dates are required for auxiliary validation.")
    train_end = max(1, int(len(dates) * .70))
    cal_end = min(max(train_end + 1, int(len(dates) * .80)), len(dates) - 1)
    train_cut, cal_cut = dates[train_end - 1], dates[cal_end - 1]
    train = frame[frame["date"] <= train_cut].copy()
    calibration = frame[(frame["date"] > train_cut) & (frame["date"] <= cal_cut)].copy()
    test = frame[frame["date"] > cal_cut].copy()
    if train.empty or calibration.empty or test.empty:
        raise ValueError("Auxiliary chronological split contains an empty partition.")
    return train, calibration, test


def _aux_pipeline(c_value, multiclass=False):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2500, class_weight="balanced", C=c_value
        )),
    ])


def _fit_aux_method(frame):
    usable = frame[frame["method_target"].isin(AUX_METHOD_CLASSES)].copy()
    train, calibration, test = _chronological_aux_split(usable)
    if train["method_target"].nunique() < 3:
        raise ValueError("The training period does not contain all three method classes.")
    candidates, scores = [], []
    for c_value in (0.10, 0.50, 1.0):
        model = _aux_pipeline(c_value, multiclass=True)
        model.fit(train[AUX_METHOD_FEATURES], train["method_target"])
        probs = model.predict_proba(calibration[AUX_METHOD_FEATURES])
        scores.append({
            "C": c_value,
            "calibration_log_loss": float(log_loss(
                calibration["method_target"], probs, labels=list(model.classes_)
            )),
        })
        candidates.append(model)
    incumbent = next(i for i, row in enumerate(scores) if row["C"] == .50)
    best = min(range(len(scores)), key=lambda i: scores[i]["calibration_log_loss"])
    selected = best if scores[best]["calibration_log_loss"] < scores[incumbent]["calibration_log_loss"] else incumbent
    model = candidates[selected]
    # Calibrate each class one-vs-rest on the later calibration window, then
    # renormalize. The untouched test window remains evaluation-only.
    cal_raw = np.clip(model.predict_proba(calibration[AUX_METHOD_FEATURES]), 1e-6, 1 - 1e-6)
    class_calibrators = []
    for class_index, label in enumerate(model.classes_):
        logits = np.log(cal_raw[:, class_index] / (1 - cal_raw[:, class_index])).reshape(-1, 1)
        binary_target = (calibration["method_target"].to_numpy() == label).astype(int)
        if np.unique(binary_target).size == 2:
            calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(logits, binary_target)
            class_calibrators.append(calibrator)
        else:
            class_calibrators.append(None)

    class CalibratedMethodModel:
        def __init__(self, estimator, calibrators):
            self.estimator, self.calibrators = estimator, calibrators
            self.classes_ = estimator.classes_
        def predict_proba(self, X):
            raw = np.clip(self.estimator.predict_proba(X), 1e-6, 1 - 1e-6)
            calibrated = []
            for i, calibrator in enumerate(self.calibrators):
                if calibrator is None:
                    calibrated.append(raw[:, i])
                else:
                    logits = np.log(raw[:, i] / (1 - raw[:, i])).reshape(-1, 1)
                    calibrated.append(calibrator.predict_proba(logits)[:, 1])
            matrix = np.column_stack(calibrated)
            return matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-9, None)
        def predict(self, X):
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
    model = CalibratedMethodModel(model, class_calibrators)
    probabilities = model.predict_proba(test[AUX_METHOD_FEATURES])
    predictions = model.predict(test[AUX_METHOD_FEATURES])
    baseline = DummyClassifier(strategy="prior").fit(train[AUX_METHOD_FEATURES], train["method_target"])
    baseline_probs = baseline.predict_proba(test[AUX_METHOD_FEATURES])
    baseline_pred = baseline.predict(test[AUX_METHOD_FEATURES])
    ll = float(log_loss(test["method_target"], probabilities, labels=list(model.classes_)))
    baseline_ll = float(log_loss(test["method_target"], baseline_probs, labels=list(baseline.classes_)))
    bal_acc = float(balanced_accuracy_score(test["method_target"], predictions))
    base_bal_acc = float(balanced_accuracy_score(test["method_target"], baseline_pred))
    metrics = {
        "available": True, "passed": bool(ll < baseline_ll and bal_acc > base_bal_acc),
        "accuracy": float(accuracy_score(test["method_target"], predictions)),
        "balanced_accuracy": bal_acc, "log_loss": ll,
        "brier": float(brier_score_loss(test["method_target"].map({label: i for i, label in enumerate(model.classes_)}), np.eye(len(model.classes_))[np.array([list(model.classes_).index(x) for x in test["method_target"]])] if False else np.array([probabilities[i, list(model.classes_).index(label)] for i, label in enumerate(test["method_target"])])),
        "baseline_balanced_accuracy": base_bal_acc, "baseline_log_loss": baseline_ll,
        "test_fights": int(len(test)), "selected_c": float(scores[selected]["C"]),
        "candidate_comparison": scores,
        "test_start": str(test["date"].min().date()), "test_end": str(test["date"].max().date()),
        "class_counts": {str(k): int(v) for k, v in test["method_target"].value_counts().items()},
    }
    return model, metrics


def _fit_aux_round(frame, line, seconds):
    usable = frame.dropna(subset=["duration_seconds"]).copy()
    usable["target"] = (usable["duration_seconds"] > seconds).astype(int)
    train, calibration, test = _chronological_aux_split(usable)
    if train["target"].nunique() < 2 or calibration["target"].nunique() < 2 or test["target"].nunique() < 2:
        raise ValueError(f"Not enough over/under examples on both sides of the {line} line.")
    candidates, scores = [], []
    for c_value in (0.10, 0.50, 1.0):
        model = _aux_pipeline(c_value)
        model.fit(train[AUX_METHOD_FEATURES], train["target"])
        probs = model.predict_proba(calibration[AUX_METHOD_FEATURES])[:, 1]
        scores.append({
            "C": c_value,
            "calibration_brier": float(brier_score_loss(calibration["target"], probs)),
            "calibration_log_loss": float(log_loss(calibration["target"], probs, labels=[0, 1])),
        })
        candidates.append(model)
    incumbent = next(i for i, row in enumerate(scores) if row["C"] == .50)
    best = min(range(len(scores)), key=lambda i: (scores[i]["calibration_brier"], scores[i]["calibration_log_loss"]))
    improved = (
        scores[best]["calibration_brier"] < scores[incumbent]["calibration_brier"]
        and scores[best]["calibration_log_loss"] < scores[incumbent]["calibration_log_loss"]
    )
    selected = best if improved else incumbent
    model = candidates[selected]
    cal_raw = np.clip(model.predict_proba(calibration[AUX_METHOD_FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    cal_logits = np.log(cal_raw / (1 - cal_raw)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(cal_logits, calibration["target"])
    class CalibratedRoundModel:
        def __init__(self, estimator, probability_calibrator):
            self.estimator, self.probability_calibrator = estimator, probability_calibrator
        def predict_proba(self, X):
            raw = np.clip(self.estimator.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
            logits = np.log(raw / (1 - raw)).reshape(-1, 1)
            over = self.probability_calibrator.predict_proba(logits)[:, 1]
            return np.column_stack([1 - over, over])
    model = CalibratedRoundModel(model, calibrator)
    probs = model.predict_proba(test[AUX_METHOD_FEATURES])[:, 1]
    pred = (probs >= .5).astype(int)
    baseline = DummyClassifier(strategy="prior").fit(train[AUX_METHOD_FEATURES], train["target"])
    base_probs = baseline.predict_proba(test[AUX_METHOD_FEATURES])[:, 1]
    base_pred = baseline.predict(test[AUX_METHOD_FEATURES])
    ll = float(log_loss(test["target"], probs, labels=[0, 1]))
    base_ll = float(log_loss(test["target"], base_probs, labels=[0, 1]))
    bal_acc = float(balanced_accuracy_score(test["target"], pred))
    base_bal_acc = float(balanced_accuracy_score(test["target"], base_pred))
    metrics = {
        "available": True, "passed": bool(ll < base_ll and bal_acc > base_bal_acc),
        "accuracy": float(accuracy_score(test["target"], pred)),
        "balanced_accuracy": bal_acc, "brier": float(brier_score_loss(test["target"], probs)),
        "log_loss": ll, "baseline_brier": float(brier_score_loss(test["target"], base_probs)),
        "baseline_log_loss": base_ll, "baseline_balanced_accuracy": base_bal_acc,
        "test_fights": int(len(test)), "selected_c": float(scores[selected]["C"]),
        "candidate_comparison": scores,
        "test_start": str(test["date"].min().date()), "test_end": str(test["date"].max().date()),
        "over_rate": float(test["target"].mean()),
    }
    return model, metrics


def train_method_and_round_models(results, fight_stats):
    """Train independent, leakage-safe method and round-duration predictors."""
    frame, snapshot = build_auxiliary_training_rows(results, fight_stats)
    output = {"method": None, "rounds": {}}
    metrics = {"method": {"available": False}, "rounds": {}}
    try:
        output["method"], metrics["method"] = _fit_aux_method(frame)
    except Exception as exc:
        metrics["method"] = {"available": False, "reason": str(exc)}
    for line, seconds in AUX_ROUND_LINES.items():
        try:
            output["rounds"][line], metrics["rounds"][line] = _fit_aux_round(frame, line, seconds)
        except Exception as exc:
            metrics["rounds"][line] = {"available": False, "reason": str(exc)}
    return output, metrics, snapshot, frame


def predict_auxiliary_matchups(models, snapshot, matchups):
    """Return method probabilities and O/U 1.5 and 2.5 probabilities for a card."""
    rows = []
    for bout in matchups or []:
        f1 = str(bout.get("Fighter 1", "")).strip()
        f2 = str(bout.get("Fighter 2", "")).strip()
        if not f1 or not f2:
            continue
        neutral = {
            "experience": 0, "ko_win": 1/3, "sub_win": 1/3, "decision_win": 1/3,
            "ko_loss": 1/3, "sub_loss": 1/3, "decision_loss": 1/3,
            "duration": 600.0, "recent_duration": 600.0, "finish_rate": 1/3, "days_since": 730,
            **{key: 0.0 for key in AUX_PERFORMANCE_FEATURES},
        }
        a = snapshot.get(_fighter_key(f1), neutral)
        b = snapshot.get(_fighter_key(f2), neutral)
        values = {
            "experience_mean": (a["experience"] + b["experience"]) / 2,
            "experience_max": max(a["experience"], b["experience"]),
            "duration_mean": (a["duration"] + b["duration"]) / 2,
            "duration_max": max(a["duration"], b["duration"]),
            "recent_duration_mean": (a["recent_duration"] + b["recent_duration"]) / 2,
            "finish_rate_mean": (a["finish_rate"] + b["finish_rate"]) / 2,
            "days_since_mean": (a["days_since"] + b["days_since"]) / 2,
        }
        for key in ("ko_win", "sub_win", "decision_win", "ko_loss", "sub_loss", "decision_loss"):
            values[key + "_mean"] = (a[key] + b[key]) / 2
            values[key + "_max"] = max(a[key], b[key])
        for key in AUX_PERFORMANCE_FEATURES:
            values[key + "_mean"] = (a[key] + b[key]) / 2
            values[key + "_max"] = max(a[key], b[key])
            values[key + "_diff_abs"] = abs(a[key] - b[key])
        X = pd.DataFrame([{key: float(values[key]) for key in AUX_METHOD_FEATURES}])
        row = {
            "Bout": f"{f1} vs {f2}",
            "Fighter 1 prior fights": int(a["experience"]),
            "Fighter 2 prior fights": int(b["experience"]),
            "Fighter 1 sig strikes/min": round(float(a["sig_landed_rate"]), 2),
            "Fighter 2 sig strikes/min": round(float(b["sig_landed_rate"]), 2),
            "Fighter 1 takedowns/min": round(float(a["takedown_rate"]), 3),
            "Fighter 2 takedowns/min": round(float(b["takedown_rate"]), 3),
            "Combined prior finish rate %": round(
                (float(a["finish_rate"]) + float(b["finish_rate"])) * 50, 1
            ),
            "Average prior fight duration (sec)": round(
                (float(a["duration"]) + float(b["duration"])) / 2, 1
            ),
        }
        method_model = (models or {}).get("method")
        if method_model is not None:
            method_probs = method_model.predict_proba(X)[0]
            for label, probability in zip(method_model.classes_, method_probs):
                row[f"{label} %"] = round(float(probability) * 100, 1)
            if len(method_probs):
                row["Most likely method"] = str(method_model.classes_[int(np.argmax(method_probs))])
        for line, model in (models or {}).get("rounds", {}).items():
            over = float(model.predict_proba(X)[0, 1])
            row[f"Over {line} %"] = round(over * 100, 1)
            row[f"Under {line} %"] = round((1 - over) * 100, 1)
        missing = [name for name in (f1, f2) if _fighter_key(name) not in snapshot]
        row["Style data warning"] = ("Limited history: " + ", ".join(missing)) if missing else ""
        rows.append(row)
    return pd.DataFrame(rows)
