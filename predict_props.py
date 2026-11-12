"""Generate PTS/REB/AST projections and over/under probabilities.

This utility deliberately accepts and outputs prop line numbers only. It does
not read, retain, calculate, display, or export bookmaker prices or odds.
"""
from pathlib import Path
import argparse
import math

import joblib
import numpy as np
import pandas as pd

MARKETS = {"points", "rebounds", "assists"}
REQUIRED = {
    "game_date", "player", "team", "opponent",
    "home", "minutes", "market", "line",
}


def cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normalized_name(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--props", default="todays_props.csv")
    parser.add_argument("--models", default="models")
    parser.add_argument("--features", default="models/latest_player_features.csv")
    parser.add_argument("--out", default="predictions.csv")
    args = parser.parse_args()

    props = pd.read_csv(args.props)
    missing = REQUIRED - set(props.columns)
    if missing:
        raise SystemExit(f"Missing required columns in props file: {sorted(missing)}")

    # Explicit column selection prevents optional pricing/bookmaker fields in
    # an input CSV from leaking into the output.
    props = props[sorted(REQUIRED)].copy()
    props["market"] = props["market"].astype(str).str.strip().str.lower()
    props["line"] = pd.to_numeric(props["line"], errors="coerce")
    props["minutes"] = pd.to_numeric(props["minutes"], errors="coerce")
    props["home"] = pd.to_numeric(props["home"], errors="coerce")
    props = props[props["market"].isin(MARKETS) & props["line"].notna()].copy()
    if props.empty:
        raise SystemExit("No valid PTS/REB/AST prop lines were supplied.")

    features = pd.read_csv(args.features)
    if "PLAYER_NAME" not in features or "PLAYER_ID" not in features:
        raise SystemExit("The player feature snapshot is missing PLAYER_NAME or PLAYER_ID.")

    props["player_key"] = normalized_name(props["player"])
    features["player_key"] = normalized_name(features["PLAYER_NAME"])
    merged = props.merge(features, on="player_key", how="left", suffixes=("", "_hist"))

    unknown = merged.loc[merged["PLAYER_ID"].isna(), "player"].drop_duplicates().tolist()
    if unknown:
        print("WARNING — no trained player match (excluded): " + ", ".join(unknown))

    packs = {}
    rows = []
    for _, row in merged.iterrows():
        market = str(row["market"])
        if market not in MARKETS or pd.isna(row.get("PLAYER_ID")):
            continue

        if market not in packs:
            model_path = Path(args.models) / f"{market}_model.joblib"
            if not model_path.is_file():
                print(f"WARNING — no trained model for {market}; excluded.")
                continue
            packs[market] = joblib.load(model_path)

        pack = packs[market]
        feature_cols = pack["feature_cols"]
        xrow = {column: row.get(column, np.nan) for column in feature_cols}
        X = pd.DataFrame([xrow], columns=feature_cols)
        projection = float(pack["model"].predict(X)[0])

        recent_minutes = pd.to_numeric(row.get("MIN_L5", np.nan), errors="coerce")
        listed_minutes = pd.to_numeric(row.get("minutes", np.nan), errors="coerce")
        if pd.notna(recent_minutes) and recent_minutes > 0 and pd.notna(listed_minutes):
            projection *= max(0.5, min(1.5, float(listed_minutes) / float(recent_minutes)))

        line = float(row["line"])
        residual_std = max(float(pack.get("residual_std", 1.0)), 0.5)
        p_over = 1.0 - cdf((line + 0.5 - projection) / residual_std)
        p_over = min(max(p_over, 0.0001), 0.9999)
        p_under = 1.0 - p_over

        rows.append({
            "game_date": row["game_date"],
            "player": row["player"],
            "team": row["team"],
            "opponent": row["opponent"],
            "market": market,
            "prop_line": line,
            "projection": round(max(0.0, projection), 2),
            "projection_gap": round(projection - line, 2),
            "prob_over_pct": round(p_over * 100.0, 1),
            "prob_under_pct": round(p_under * 100.0, 1),
            "estimated_best_side": "Over" if p_over >= p_under else "Under",
        })

    result = pd.DataFrame(rows)
    if result.empty:
        raise SystemExit("No predictions produced. Check player names, markets and trained model files.")

    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved {len(result)} prop comparisons to {args.out}")
    print("Probabilities are model estimates, not guarantees. Validate calibration before relying on them.")


if __name__ == "__main__":
    main()
