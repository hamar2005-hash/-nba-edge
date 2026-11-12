"""Leakage-safe chronological UFC winner model built from historical results only."""
from collections import defaultdict, deque
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
        "elo": state["elo"],
        "wins": state["wins"],
        "losses": state["losses"],
        "experience": state["wins"] + state["losses"],
        "win_rate": state["wins"] / max(1, state["wins"] + state["losses"]),
        "recent5": float(np.mean(state["recent5"])) if state["recent5"] else 0.5,
        "recent10": float(np.mean(state["recent10"])) if state["recent10"] else 0.5,
        "streak": state["streak"],
        "days_since_fight": min(days, 730),
    }

def build_chronological_rows(results):
    required = {"Date", "Fighter 1", "Fighter 2", "Result"}
    if not required.issubset(results.columns):
        raise ValueError(f"Results data missing columns: {sorted(required - set(results.columns))}")
    df = results.copy()
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["result"] = pd.to_numeric(df["Result"], errors="coerce")
    df["fighter1"] = df["Fighter 1"].astype(str).str.strip()
    df["fighter2"] = df["Fighter 2"].astype(str).str.strip()
    df = df.dropna(subset=["date", "result"])
    df = df[(df["fighter1"] != "") & (df["fighter2"] != "") & (df["fighter1"] != df["fighter2"])]
    df = df[df["result"].isin([1, 2, 3])].sort_values(["date"], kind="stable")
    states = defaultdict(lambda: {
        "elo": 1500.0, "wins": 0, "losses": 0, "streak": 0,
        "recent5": deque(maxlen=5), "recent10": deque(maxlen=10), "dates": [],
    })
    rows = []
    # All bouts on a calendar date use snapshots from before that date.
    for day, group in df.groupby("date", sort=True):
        pending = []
        for _, fight in group.iterrows():
            a, b = fight["fighter1"], fight["fighter2"]
            sa, sb = _state_features(states[a], day), _state_features(states[b], day)
            rows.append({
                "date": day,
                **{f: sa[f] - sb[f] for f in [
                    "elo", "wins", "losses", "experience", "win_rate",
                    "recent5", "recent10", "streak", "days_since_fight"
                ]},
                "target": int(fight["result"] == 1),
                "fighter1": a, "fighter2": b,
            })
            pending.append((a, b, int(fight["result"])))
        # Update after every bout on the same date has been featurised.
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
    return pd.DataFrame(rows), states

def train_and_evaluate(results):
    frame, states = build_chronological_rows(results)
    if len(frame) < 250:
        raise ValueError(f"Only {len(frame)} usable decisive fights; at least 250 are required.")
    split = int(len(frame) * 0.8)
    train, test = frame.iloc[:split], frame.iloc[split:]
    if train["target"].nunique() < 2 or test["target"].nunique() < 2:
        raise ValueError("Chronological split does not contain both winner classes.")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
    ])
    model.fit(train[FEATURES], train["target"])
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    baseline = DummyClassifier(strategy="prior").fit(train[FEATURES], train["target"])
    base_probs = baseline.predict_proba(test[FEATURES])[:, 1]
    metrics = {
        "train_fights": int(len(train)),
        "test_fights": int(len(test)),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "accuracy": float(accuracy_score(test["target"], predictions)),
        "baseline_accuracy": float(accuracy_score(test["target"], (base_probs >= 0.5).astype(int))),
        "log_loss": float(log_loss(test["target"], probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(test["target"], probabilities)),
        "baseline_brier": float(brier_score_loss(test["target"], base_probs)),
    }
    # Refit on all available historical data only after the holdout is scored.
    model.fit(frame[FEATURES], frame["target"])
    snapshot = {name: _state_features(state, pd.Timestamp.utcnow().tz_localize(None).normalize()) for name, state in states.items()}
    return model, metrics, snapshot, frame
