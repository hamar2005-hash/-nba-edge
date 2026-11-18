"""Smoke-test the dedicated 3PM model against the same feature schema used by daily projections."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
model_path = ROOT / "models" / "threes_model.joblib"
snapshot_path = ROOT / "models" / "latest_player_features.csv"
assert model_path.exists(), f"Missing trained model: {model_path}"
assert snapshot_path.exists(), f"Missing feature snapshot: {snapshot_path}"

pack = joblib.load(model_path)
snapshot = pd.read_csv(snapshot_path)
features = pack["feature_cols"]
missing = [c for c in features if c not in snapshot.columns]
assert not missing, f"Snapshot missing model features: {missing}"
assert pack.get("target") == "FG3M"
assert pack.get("market") == "threes"

sample = snapshot[features].tail(1)
prediction = float(pack["model"].predict(sample)[0])
assert np.isfinite(prediction), f"Non-finite prediction: {prediction}"
assert prediction >= 0, f"Negative 3PM prediction: {prediction}"

# Verify the production daily projection and UI/settlement paths expose this market.
daily = (ROOT / "auto_daily.py").read_text()
app = (ROOT / "app.py").read_text()
settle = (ROOT / "settle_results.py").read_text()
assert '"threes"' in daily and 'f"{market}_projection"' in daily
assert 'column.startswith("OPP_ALLOWED_")' in daily, "Daily inference must apply the team-keyed opponent-defense snapshot"
assert '"threes_projection"' in app and "Three-pointers made" in app
assert '"actual_threes"' in settle and '"threes": "FG3M"' in settle

print({
    "model": pack.get("model_version"),
    "feature_count": len(features),
    "snapshot_rows": len(snapshot),
    "sample_player": str(snapshot.iloc[-1].get("PLAYER_NAME", "unknown")),
    "sample_prediction": round(max(0.0, prediction), 2),
    "daily_projection_path": "present",
    "app_leaderboard_and_accuracy_path": "present",
    "settlement_path": "present",
})
