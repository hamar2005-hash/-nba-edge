"""Persistent, pre-fight UFC prediction logging and result settlement."""
from pathlib import Path
import unicodedata

import pandas as pd

HISTORY_COLUMNS = [
    "event_id", "event_name", "event_date", "bout", "fighter1", "fighter2",
    "predicted_winner", "confidence", "predicted_at", "actual_winner",
    "actual_method", "status",
]


def fighter_key(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = "".join(ch for ch in value if ch.isalnum() or ch.isspace())
    return " ".join(value.split())


def record_predictions(path, event, predictions, now_utc=None):
    """Append immutable predictions only when saved before the event begins."""
    path = Path(path)
    event_date = pd.to_datetime(event.get("date"), utc=True, errors="coerce")
    now_utc = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    if pd.isna(event_date) or now_utc.tz_convert("UTC") >= event_date:
        return 0
    if predictions is None or predictions.empty:
        return 0

    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=HISTORY_COLUMNS)
    for column in HISTORY_COLUMNS:
        if column not in existing.columns:
            existing[column] = pd.NA

    known = set(
        zip(existing["event_id"].astype(str), existing["fighter1"].map(fighter_key),
            existing["fighter2"].map(fighter_key))
    )
    additions = []
    for _, row in predictions.iterrows():
        parts = str(row.get("Bout", "")).split(" vs ", 1)
        if len(parts) != 2:
            continue
        fighter1, fighter2 = parts
        event_id = str(event.get("id", ""))
        pair = (event_id, fighter_key(fighter1), fighter_key(fighter2))
        reverse_pair = (event_id, fighter_key(fighter2), fighter_key(fighter1))
        if pair in known or reverse_pair in known:
            continue
        additions.append({
            "event_id": event_id,
            "event_name": str(event.get("name", "UFC event")),
            "event_date": event_date.isoformat(),
            "bout": f"{fighter1} vs {fighter2}",
            "fighter1": fighter1,
            "fighter2": fighter2,
            "predicted_winner": str(row.get("Predicted winner", "")),
            "confidence": pd.to_numeric(row.get("Confidence"), errors="coerce"),
            "predicted_at": now_utc.isoformat(),
            "actual_winner": "",
            "actual_method": "",
            "status": "Pending",
        })
        known.add(pair)
    if additions:
        updated = pd.concat([existing, pd.DataFrame(additions)], ignore_index=True)
        updated[HISTORY_COLUMNS].to_csv(path, index=False)
    elif not path.exists():
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(path, index=False)
    return len(additions)


def settle_predictions(path, results):
    """Grade saved picks against an unambiguous matching historical result."""
    path = Path(path)
    if not path.exists() or results is None or results.empty:
        return 0
    history = pd.read_csv(path)
    required = {"Date", "Fighter 1", "Fighter 2", "Result"}
    if not required.issubset(results.columns) or history.empty:
        return 0

    actual = results.copy()
    actual["_date"] = pd.to_datetime(actual["Date"], dayfirst=True, errors="coerce")
    actual["_a"] = actual["Fighter 1"].map(fighter_key)
    actual["_b"] = actual["Fighter 2"].map(fighter_key)
    actual["_result"] = pd.to_numeric(actual["Result"], errors="coerce")
    method_col = next((c for c in ("Method", "method") if c in actual.columns), None)

    changed = 0
    for idx, prediction in history.iterrows():
        if str(prediction.get("status", "")) != "Pending":
            continue
        event_date = pd.to_datetime(prediction.get("event_date"), utc=True, errors="coerce")
        if pd.isna(event_date):
            continue
        target_date = event_date.tz_convert("Europe/London").tz_localize(None).normalize()
        p1, p2 = fighter_key(prediction.get("fighter1")), fighter_key(prediction.get("fighter2"))
        dates = actual["_date"].dt.normalize()
        pair_match = (
            ((actual["_a"] == p1) & (actual["_b"] == p2)) |
            ((actual["_a"] == p2) & (actual["_b"] == p1))
        )
        candidates = actual[pair_match & ((dates - target_date).abs().dt.days <= 1)]
        if len(candidates) != 1:
            continue
        fight = candidates.iloc[0]
        result = fight["_result"]
        if result == 3:
            history.at[idx, "status"] = "Void"
            history.at[idx, "actual_winner"] = "Draw / No contest"
        elif result in (1, 2):
            winner = str(fight["Fighter 1"] if result == 1 else fight["Fighter 2"])
            history.at[idx, "actual_winner"] = winner
            history.at[idx, "status"] = (
                "Correct" if fighter_key(winner) == fighter_key(prediction["predicted_winner"])
                else "Incorrect"
            )
        else:
            continue
        if method_col:
            history.at[idx, "actual_method"] = str(fight[method_col])
        changed += 1
    if changed:
        history[HISTORY_COLUMNS].to_csv(path, index=False)
    return changed
