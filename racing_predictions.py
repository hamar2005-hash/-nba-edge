"""Live UK/Irish horse form scoring using The Racing API horse-result history.

This is an explicitly provisional, form-based baseline. Its within-race
probabilities are softmax-normalised scores, not calibrated probabilities.
"""
from __future__ import annotations

import math
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.theracingapi.com/v1"


def _records(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("results", "data", "horses", "runners"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            nested = _records(value)
            if nested:
                return nested
    return []


def _number(value):
    if value is None:
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def _finish(value):
    if value is None:
        return np.nan
    text = str(value).strip().lower()
    if text in {"pu", "pulled up", "fell", "ur", "unseated", "bd", "ref", "ro"}:
        return np.nan
    return _number(text)


def _history_stats(history: list[dict], course: str, going: str) -> dict:
    if not history:
        return {
            "Runs": 0, "Wins": 0, "Places": 0, "Win rate": np.nan,
            "Place rate": np.nan, "Recent form": "No history",
            "Last run": "Unknown", "Form score": 0.0,
        }
    rows = []
    for item in history:
        pos = _finish(item.get("position") or item.get("pos") or item.get("finish_position"))
        field = _number(item.get("runners") or item.get("field_size") or item.get("number_of_runners"))
        if pd.isna(pos):
            continue
        rows.append({
            "position": pos,
            "field": field if pd.notna(field) and field > 1 else np.nan,
            "course": str(item.get("course") or "").strip().casefold(),
            "going": str(item.get("going") or "").strip().casefold(),
            "date": pd.to_datetime(item.get("date") or item.get("race_date"), errors="coerce", dayfirst=True),
        })
    if not rows:
        return {
            "Runs": 0, "Wins": 0, "Places": 0, "Win rate": np.nan,
            "Place rate": np.nan, "Recent form": "No numeric results",
            "Last run": "Unknown", "Form score": 0.0,
        }
    df = pd.DataFrame(rows).sort_values("date", ascending=False, na_position="last")
    recent = df.head(5)
    runs = len(df)
    wins = int((df.position == 1).sum())
    places = int((df.position <= 3).sum())
    # Recent runs matter more, but old results still contribute to the baseline.
    recent_finish = []
    for row in recent.itertuples():
        if pd.notna(row.field) and row.field > 1:
            recent_finish.append(1 - min(max((row.position - 1) / (row.field - 1), 0), 1))
        else:
            recent_finish.append(1 / max(row.position, 1))
    recent_performance = float(np.mean(recent_finish)) if recent_finish else 0.0
    course_key = str(course or "").strip().casefold()
    going_key = str(going or "").strip().casefold()
    course_rows = df[df.course == course_key] if course_key else df.iloc[0:0]
    going_rows = df[df.going == going_key] if going_key else df.iloc[0:0]
    course_rate = float((course_rows.position == 1).mean()) if len(course_rows) else np.nan
    going_rate = float((going_rows.position == 1).mean()) if len(going_rows) else np.nan
    overall_win = wins / runs
    overall_place = places / runs
    # Shrink small samples toward a neutral prior to avoid one-run wonders.
    shrunk_win = (wins + 1.5) / (runs + 12)
    shrunk_place = (places + 3.0) / (runs + 12)
    course_component = course_rate if pd.notna(course_rate) else shrunk_win
    going_component = going_rate if pd.notna(going_rate) else shrunk_win
    score = (
        0.32 * recent_performance
        + 0.25 * shrunk_win
        + 0.18 * shrunk_place
        + 0.15 * course_component
        + 0.10 * going_component
    )
    last_date = df.date.dropna().max()
    last_run = f"{(pd.Timestamp.now().normalize() - last_date.normalize()).days} days ago" if pd.notna(last_date) else "Unknown"
    form = " ".join(str(int(x)) if x < 90 else "0" for x in recent.position.tolist())
    return {
        "Runs": runs, "Wins": wins, "Places": places,
        "Win rate": overall_win, "Place rate": overall_place,
        "Recent form": form or "Unavailable", "Last run": last_run,
        "Form score": float(score),
    }


def _card_form_stats(form: Any) -> dict:
    """Fallback using the racecard's published form string when history is not in plan."""
    text = str(form or "").strip()
    finishes = [int(x) for x in re.findall(r"(?<![A-Za-z])([1-9])", text)][-5:]
    if not finishes:
        return {
            "Runs": 0, "Wins": 0, "Places": 0, "Win rate": np.nan,
            "Place rate": np.nan, "Recent form": text or "No form supplied",
            "Last run": "Unknown", "Form score": 0.0,
            "History source": "No historical access",
        }
    # Recent form is a lightweight fallback, not a substitute for full runner history.
    weights = np.arange(1, len(finishes) + 1, dtype=float)
    performance = np.average([1 / x for x in finishes], weights=weights)
    return {
        "Runs": len(finishes), "Wins": sum(x == 1 for x in finishes),
        "Places": sum(x <= 3 for x in finishes),
        "Win rate": sum(x == 1 for x in finishes) / len(finishes),
        "Place rate": sum(x <= 3 for x in finishes) / len(finishes),
        "Recent form": text, "Last run": "Racecard form only",
        "Form score": float(performance),
        "History source": "Racecard form only",
    }


def _market_decimal(value):
    if value is None:
        return np.nan
    text = str(value).strip().lower()
    try:
        if "/" in text:
            a, b = text.split("/", 1)
            return 1 + float(a) / float(b)
        number = float(text.replace("evens", "1").replace("even", "1"))
        return number if number > 1 else np.nan
    except Exception:
        return np.nan


def fetch_form_predictions(cards: pd.DataFrame, username: str, password: str) -> pd.DataFrame:
    """Fetch up to 50 recent runs per declared runner and rank within each race.

    Requires The Racing API Basic plan for /racecards/{horse_id}/results.
    Calls are paced conservatively and responses are cached by Streamlit caller.
    """
    if cards.empty:
        return cards.copy()
    if "Horse ID" not in cards or cards["Horse ID"].astype(str).str.len().eq(0).all():
        raise ValueError("The racecard feed did not provide horse IDs, so form cannot be joined safely.")
    session = requests.Session()
    session.auth = (username, password)
    session.headers.update({"Accept": "application/json"})
    output = []
    histories = {}
    seen = set()
    history_access = True
    for row in cards.to_dict("records"):
        raw_id = row.get("Horse ID")
        horse_id = "" if pd.isna(raw_id) else str(raw_id).strip()
        if horse_id.casefold() in {"", "nan", "none"}:
            stats = _card_form_stats(row.get("Form"))
            output.append({**row, **stats})
            continue
        if horse_id not in seen:
            seen.add(horse_id)
            if history_access:
                response = session.get(
                    f"{BASE_URL}/racecards/{horse_id}/results",
                    params={"region": ["gb", "ire"], "limit": 50},
                    timeout=25,
                )
                if response.status_code in (401, 403):
                    history_access = False
                    histories[horse_id] = []
                else:
                    response.raise_for_status()
                    histories[horse_id] = _records(response.json())
                    time.sleep(0.22)
            else:
                histories[horse_id] = []
        history = histories.get(horse_id, [])
        if history:
            stats = _history_stats(history, row.get("Course"), row.get("Going"))
            stats["History source"] = "Historical runner results"
        else:
            stats = _card_form_stats(row.get("Form"))
        output.append({**row, **stats})
    result = pd.DataFrame(output)
    result["Model score"] = pd.to_numeric(result["Form score"], errors="coerce").fillna(0.0)
    race_keys = [c for c in ("Course", "Time", "Race ID") if c in result.columns]
    if race_keys:
        def normalise(group):
            values = group["Model score"].astype(float).to_numpy()
            # Softmax creates a race-relative ranking. This is not calibrated.
            z = (values - values.max()) / 0.12
            exp = np.exp(np.clip(z, -30, 0))
            group["Win estimate"] = exp / exp.sum() if exp.sum() else 1 / max(len(group), 1)
            return group
        result = result.groupby(race_keys, group_keys=False, dropna=False).apply(normalise)
    else:
        result["Win estimate"] = np.nan
    odds_col = next((c for c in ("Odds", "Price", "Decimal Odds", "Current Odds") if c in result.columns), None)
    if odds_col:
        result["Market odds"] = result[odds_col].map(_market_decimal)
        result["Market implied %"] = 100 / result["Market odds"]
        result["Model estimate %"] = result["Win estimate"] * 100
        result["Model vs market (pp)"] = result["Model estimate %"] - result["Market implied %"]
    else:
        result["Model estimate %"] = result["Win estimate"] * 100
    return result.sort_values(race_keys + ["Win estimate"], ascending=[True] * len(race_keys) + [False] if race_keys else ["Win estimate"], na_position="last")
