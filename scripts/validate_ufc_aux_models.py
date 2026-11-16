"""Train and evaluate UFC method and round models on chronological holdouts."""
import json
from pathlib import Path
import pandas as pd
import ufc_model

BASE = "https://raw.githubusercontent.com/Sam-Armstrong/UFC-Almanac/main/data/"

def main():
    results = pd.read_csv(BASE + "FightResults.csv")
    stats = pd.read_csv(BASE + "FightStats.csv")
    models, metrics, snapshot, frame = ufc_model.train_method_and_round_models(results, stats)
    report = {
        "training_rows": int(len(frame)),
        "fighter_profiles": int(len(snapshot)),
        "method": metrics.get("method", {}),
        "rounds": metrics.get("rounds", {}),
        "calibrated": True,
        "leakage_controls": [
            "All bouts on a date use fighter history from earlier dates only",
            "Method and duration labels are updated after the event date is featurised",
            "Hyperparameters selected on calibration period",
            "Probability calibration fitted on calibration period",
            "Final chronological test period remains untouched"
        ],
    }
    out = Path("artifacts/ufc_auxiliary_validation.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["training_rows"] or not report["fighter_profiles"]:
        raise SystemExit("No usable UFC training rows or fighter snapshots")
    for market in [metrics.get("method", {}), *metrics.get("rounds", {}).values()]:
        if market.get("available") and not (0 <= market.get("accuracy", -1) <= 1):
            raise SystemExit("Invalid accuracy metric")

if __name__ == "__main__":
    main()
