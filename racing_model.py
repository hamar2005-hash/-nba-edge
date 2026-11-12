"""Leakage-aware UK/IRE horse-racing baseline model.

Expected historical CSV: one row per runner, with at least race_date,
horse, finish_position and race_id (or enough fields to identify a race).
Optional columns enrich the model: course, distance, going, field_size,
weight, age, draw, trainer, jockey, class.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ALIASES = {
    "race_date": ["race_date", "date", "meeting_date"],
    "race_id": ["race_id", "raceid", "race"],
    "horse": ["horse", "horse_name", "runner", "horse name"],
    "finish_position": ["finish_position", "position", "pos", "finishing_position", "result"],
    "course": ["course", "track", "racecourse"],
    "distance": ["distance", "dist"],
    "going": ["going", "ground"],
    "field_size": ["field_size", "runners", "number_of_runners"],
    "weight": ["weight", "carried_weight", "lbs"],
    "age": ["age"],
    "draw": ["draw", "stall"],
    "trainer": ["trainer", "trainer_name"],
    "jockey": ["jockey", "jockey_name", "rider"],
    "class": ["class", "race_class"],
}

NUMERIC = ["distance_num", "field_size_num", "weight_num", "age_num", "draw_num",
           "horse_runs_before", "horse_win_rate", "horse_place_rate",
           "horse_last_finish", "horse_avg_finish5", "horse_days_since_run",
           "trainer_win_rate", "jockey_win_rate"]

def _clean_name(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

def standardise_history(frame):
    df = frame.copy()
    df.columns = [_clean_name(c) for c in df.columns]
    rename = {}
    for target, choices in ALIASES.items():
        for choice in choices:
            key = _clean_name(choice)
            if key in df.columns:
                rename[key] = target
                break
    df = df.rename(columns=rename)
    required = ["race_date", "horse", "finish_position"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    for c in ALIASES:
        if c not in df:
            df[c] = np.nan
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce", dayfirst=True)
    df["finish_position"] = pd.to_numeric(
        df["finish_position"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
    # Keep pulled-up, fell and refused runners as non-winners instead of dropping them.
    df["finish_position"] = df["finish_position"].fillna(99)
    df = df.dropna(subset=["race_date", "horse"]).copy()
    df["horse_key"] = df["horse"].astype(str).str.strip().str.lower()
    df["trainer_key"] = df["trainer"].fillna("").astype(str).str.strip().str.lower()
    df["jockey_key"] = df["jockey"].fillna("").astype(str).str.strip().str.lower()
    if df["race_id"].isna().all():
        df["race_id"] = (
            df["race_date"].dt.strftime("%Y-%m-%d") + "|" +
            df["course"].fillna("").astype(str) + "|" +
            df["distance"].fillna("").astype(str) + "|" +
            df["class"].fillna("").astype(str)
        )
    df["race_key"] = df["race_date"].dt.strftime("%Y-%m-%d") + "|" + df["race_id"].astype(str)
    df["win"] = (df["finish_position"] == 1).astype(int)
    df["place"] = (df["finish_position"] <= 3).astype(int)
    for col in ["distance", "field_size", "weight", "age", "draw"]:
        df[col + "_num"] = pd.to_numeric(
            df[col].astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0], errors="coerce"
        )
    if df["field_size_num"].isna().any():
        sizes = df.groupby("race_key")["horse"].transform("count")
        df["field_size_num"] = df["field_size_num"].fillna(sizes)
    return df.sort_values(["race_date", "race_key", "horse_key"]).reset_index(drop=True)

def build_historical_features(raw):
    df = standardise_history(raw)
    horse_runs, trainer_runs, jockey_runs = {}, {}, {}
    rows = []
    for _, row in df.iterrows():
        h = horse_runs.get(row.horse_key, [])
        t = trainer_runs.get(row.trainer_key, []) if row.trainer_key else []
        j = jockey_runs.get(row.jockey_key, []) if row.jockey_key else []
        prev_date = h[-1]["date"] if h else pd.NaT
        rows.append({
            "race_key": row.race_key, "race_date": row.race_date,
            "horse": row.horse, "win": row.win, "place": row.place,
            "distance_num": row.distance_num, "field_size_num": row.field_size_num,
            "weight_num": row.weight_num, "age_num": row.age_num, "draw_num": row.draw_num,
            "horse_runs_before": len(h),
            "horse_win_rate": np.mean([x["win"] for x in h]) if h else np.nan,
            "horse_place_rate": np.mean([x["place"] for x in h]) if h else np.nan,
            "horse_last_finish": h[-1]["finish"] if h else np.nan,
            "horse_avg_finish5": np.mean([x["finish"] for x in h[-5:]]) if h else np.nan,
            "horse_days_since_run": (row.race_date - prev_date).days if h else np.nan,
            "trainer_win_rate": np.mean([x["win"] for x in t[-100:]]) if t else np.nan,
            "jockey_win_rate": np.mean([x["win"] for x in j[-100:]]) if j else np.nan,
            "course": str(row.course), "going": str(row.going), "race_class": str(row["class"]),
        })
        item = {"date": row.race_date, "win": int(row.win), "place": int(row.place),
                "finish": float(row.finish_position)}
        horse_runs.setdefault(row.horse_key, []).append(item)
        if row.trainer_key:
            trainer_runs.setdefault(row.trainer_key, []).append(item)
        if row.jockey_key:
            jockey_runs.setdefault(row.jockey_key, []).append(item)
    return pd.DataFrame(rows)

def train_and_backtest(raw):
    features = build_historical_features(raw)
    features = features.sort_values("race_date").reset_index(drop=True)
    race_dates = features.groupby("race_key")["race_date"].first().sort_values()
    if len(race_dates) < 40:
        raise ValueError(f"Only {len(race_dates)} races found. At least 40 are required for a basic chronological test.")
    split_idx = max(1, int(len(race_dates) * 0.8))
    cutoff = race_dates.iloc[split_idx]
    train = features[features.race_date < cutoff].copy()
    test = features[features.race_date >= cutoff].copy()
    if train["win"].nunique() < 2 or test.empty:
        raise ValueError("Not enough winners or test data after the chronological split.")
    categorical = ["course", "going", "race_class"]
    X = pd.get_dummies(features[NUMERIC + categorical], columns=categorical, dummy_na=True)
    X_train, X_test = X.loc[train.index], X.loc[test.index]
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("classifier", RandomForestClassifier(
            n_estimators=350, min_samples_leaf=8, max_features="sqrt",
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )),
    ])
    model.fit(X_train, train["win"])
    probs = model.predict_proba(X_test)[:, list(model.named_steps["classifier"].classes_).index(1)]
    test_out = test[["race_key", "race_date", "horse", "win", "place"]].copy()
    test_out["win_probability"] = probs
    test_out["rank"] = test_out.groupby("race_key")["win_probability"].rank(method="first", ascending=False)
    top = test_out[test_out["rank"] == 1]
    metrics = {
        "train_races": int(train["race_key"].nunique()),
        "test_races": int(test["race_key"].nunique()),
        "train_runners": int(len(train)), "test_runners": int(len(test)),
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "top_pick_win_rate": float(top["win"].mean()),
        "runner_accuracy": float(accuracy_score(test["win"], probs >= 0.5)),
        "brier": float(brier_score_loss(test["win"], probs)),
        "log_loss": float(log_loss(test["win"], np.clip(probs, 1e-6, 1-1e-6))),
        "base_win_rate": float(test["win"].mean()),
    }
    # Refit on all available history for future racecard scoring.
    model.fit(X, features["win"])
    return model, list(X.columns), metrics, test_out

def score_racecards(model, feature_columns, cards):
    """Conservative initial scorer using only fields available in the live card.
    Historical-form features remain missing until runner history is joined.
    """
    df = cards.copy()
    aliases = {"distance_num": "Distance", "field_size_num": "Field size",
               "weight_num": "Weight", "age_num": "Age", "draw_num": "Draw"}
    for target, source in aliases.items():
        df[target] = pd.to_numeric(df[source], errors="coerce") if source in df else np.nan
    for col in ["horse_runs_before", "horse_win_rate", "horse_place_rate",
                "horse_last_finish", "horse_avg_finish5", "horse_days_since_run",
                "trainer_win_rate", "jockey_win_rate"]:
        df[col] = np.nan
    for target, source in [("course", "Course"), ("going", "Going"), ("race_class", "Class")]:
        df[target] = df[source].astype(str) if source in df else "nan"
    X = pd.get_dummies(df[NUMERIC + ["course", "going", "race_class"]],
                       columns=["course", "going", "race_class"], dummy_na=True)
    X = X.reindex(columns=feature_columns, fill_value=0)
    raw = model.predict_proba(X)[:, list(model.named_steps["classifier"].classes_).index(1)]
    df["Model score"] = raw
    df["Race rank"] = df.groupby(["Course", "Time", "Race ID"])["Model score"].rank(
        method="first", ascending=False
    )
    return df
