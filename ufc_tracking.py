"""Persistent, pre-fight UFC prediction logging and result settlement."""
from pathlib import Path
import unicodedata

import pandas as pd

HISTORY_COLUMNS = [
    "event_id", "event_name", "event_date", "bout", "fighter1", "fighter2",
    "predicted_winner", "confidence", "predicted_method",
    "predicted_over_1_5", "predicted_over_2_5",
    "predicted_odds", "odds_bookmaker",
    "predicted_over_1_5_side", "over_1_5_odds", "over_1_5_bookmaker", "over_1_5_profit",
    "predicted_over_2_5_side", "over_2_5_odds", "over_2_5_bookmaker", "over_2_5_profit",
    "predicted_at",
    "actual_winner", "actual_method", "actual_duration_seconds",
    "status", "method_status", "over_1_5_status", "over_2_5_status",
]


def fighter_key(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = "".join(ch for ch in value if ch.isalnum() or ch.isspace())
    return " ".join(value.split())


def _probability(row, label):
    try:
        value = float(row.get(label))
        return value if 0 <= value <= 100 else None
    except (TypeError, ValueError):
        return None


def record_predictions(path, event, predictions, auxiliary_predictions=None, now_utc=None):
    """Append immutable winner/method/round predictions before the event starts."""
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
    known = set(zip(
        existing["event_id"].astype(str),
        existing["fighter1"].map(fighter_key),
        existing["fighter2"].map(fighter_key),
    ))
    aux_lookup = {}
    if auxiliary_predictions is not None and not auxiliary_predictions.empty:
        for _, aux in auxiliary_predictions.iterrows():
            parts = str(aux.get("Bout", "")).split(" vs ", 1)
            if len(parts) == 2:
                aux_lookup[frozenset((fighter_key(parts[0]), fighter_key(parts[1])))] = aux

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
        aux = aux_lookup.get(frozenset((fighter_key(fighter1), fighter_key(fighter2))))
        aux = aux if aux is not None else {}
        additions.append({
            "event_id": event_id, "event_name": str(event.get("name", "UFC event")),
            "event_date": event_date.isoformat(), "bout": f"{fighter1} vs {fighter2}",
            "fighter1": fighter1, "fighter2": fighter2,
            "predicted_winner": str(row.get("Predicted winner", "")),
            "confidence": pd.to_numeric(row.get("Confidence"), errors="coerce"),
            "predicted_method": str(aux.get("Most likely method", "")),
            "predicted_over_1_5": _probability(aux, "Over 1.5 %"),
            "predicted_over_2_5": _probability(aux, "Over 2.5 %"),
            "predicted_over_1_5_side": ("Over" if _probability(aux, "Over 1.5 %") is not None and _probability(aux, "Over 1.5 %") >= 50 else "Under") if _probability(aux, "Over 1.5 %") is not None else "",
            "predicted_over_2_5_side": ("Over" if _probability(aux, "Over 2.5 %") is not None and _probability(aux, "Over 2.5 %") >= 50 else "Under") if _probability(aux, "Over 2.5 %") is not None else "",
            "predicted_at": now_utc.isoformat(), "actual_winner": "",
            "actual_method": "", "actual_duration_seconds": pd.NA,
            "status": "Pending", "method_status": "Pending" if aux.get("Most likely method") else "Not tracked",
            "over_1_5_status": "Pending" if _probability(aux, "Over 1.5 %") is not None else "Not tracked",
            "over_2_5_status": "Pending" if _probability(aux, "Over 2.5 %") is not None else "Not tracked",
        })
        known.add(pair)
    if additions:
        updated = pd.concat([existing, pd.DataFrame(additions)], ignore_index=True)
        updated[HISTORY_COLUMNS].to_csv(path, index=False)
    elif not path.exists():
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(path, index=False)
    return len(additions)


def _duration_lookup(fight_stats):
    required = {"Date", "Name", "Time"}
    if fight_stats is None or not required.issubset(fight_stats.columns):
        return {}
    stats = fight_stats.copy()
    stats["_date"] = pd.to_datetime(stats["Date"], format="mixed", dayfirst=True, errors="coerce").dt.normalize()
    stats["_fighter"] = stats["Name"].map(fighter_key)
    stats["_seconds"] = pd.to_numeric(stats["Time"], errors="coerce")
    stats = stats[stats["_date"].notna() & stats["_fighter"].ne("") & stats["_seconds"].between(1, 1500)]
    lookup = {}
    for (day, fighter), group in stats.groupby(["_date", "_fighter"]):
        # Ambiguous same-day repeat appearances are intentionally excluded.
        if len(group) == 1:
            lookup[(day, fighter)] = float(group.iloc[0]["_seconds"])
    return lookup


def settle_predictions(path, results, fight_stats=None):
    """Grade winner, method and round picks using unambiguous historical results."""
    path = Path(path)
    if not path.exists() or results is None or results.empty:
        return 0
    history = pd.read_csv(path, dtype={
        "event_id": "string", "event_name": "string", "bout": "string",
        "fighter1": "string", "fighter2": "string", "predicted_winner": "string",
        "predicted_method": "string", "actual_winner": "string",
        "actual_method": "string", "status": "string", "method_status": "string",
        "over_1_5_status": "string", "over_2_5_status": "string",
        "odds_bookmaker": "string",
    })
    for column in HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = pd.NA
    required = {"Date", "Fighter 1", "Fighter 2", "Result"}
    if not required.issubset(results.columns) or history.empty:
        return 0

    actual = results.copy()
    actual["_date"] = pd.to_datetime(actual["Date"], format="mixed", dayfirst=True, errors="coerce")
    actual["_a"] = actual["Fighter 1"].map(fighter_key)
    actual["_b"] = actual["Fighter 2"].map(fighter_key)
    actual["_result"] = pd.to_numeric(actual["Result"], errors="coerce")
    method_col = next((c for c in ("Method", "method") if c in actual.columns), None)
    durations = _duration_lookup(fight_stats)
    changed = 0

    for idx, prediction in history.iterrows():
        pending = [
            prediction.get("status") == "Pending",
            prediction.get("method_status") == "Pending",
            prediction.get("over_1_5_status") == "Pending",
            prediction.get("over_2_5_status") == "Pending",
        ]
        if not any(pending):
            continue
        event_date = pd.to_datetime(prediction.get("event_date"), utc=True, errors="coerce")
        if pd.isna(event_date):
            continue
        target_date = event_date.tz_convert("Europe/London").tz_localize(None).normalize()
        p1, p2 = fighter_key(prediction.get("fighter1")), fighter_key(prediction.get("fighter2"))
        dates = actual["_date"].dt.normalize()
        pair_match = (((actual["_a"] == p1) & (actual["_b"] == p2)) |
                      ((actual["_a"] == p2) & (actual["_b"] == p1)))
        candidates = actual[pair_match & ((dates - target_date).abs().dt.days <= 1)]
        if len(candidates) != 1:
            continue
        fight = candidates.iloc[0]
        result = fight["_result"]
        if result == 3:
            for col in ("status", "method_status", "over_1_5_status", "over_2_5_status"):
                if history.at[idx, col] == "Pending":
                    history.at[idx, col] = "Void"
            history.at[idx, "actual_winner"] = "Draw / No contest"
            changed += 1
            continue
        if result not in (1, 2):
            continue

        winner = str(fight["Fighter 1"] if result == 1 else fight["Fighter 2"])
        if history.at[idx, "status"] == "Pending":
            history.at[idx, "actual_winner"] = winner
            history.at[idx, "status"] = (
                "Correct" if fighter_key(winner) == fighter_key(prediction.get("predicted_winner"))
                else "Incorrect"
            )
        actual_method = str(fight[method_col]) if method_col else ""
        if method_col:
            history.at[idx, "actual_method"] = actual_method
            if history.at[idx, "method_status"] == "Pending":
                predicted = str(prediction.get("predicted_method", ""))
                # Group result labels to the same three classes as the model.
                label = actual_method.casefold()
                if "ko" in label or "tko" in label or "technical knockout" in label:
                    actual_group = "KO/TKO"
                elif "sub" in label:
                    actual_group = "Submission"
                elif "decision" in label:
                    actual_group = "Decision"
                else:
                    actual_group = ""
                if actual_group:
                    history.at[idx, "method_status"] = "Correct" if predicted == actual_group else "Incorrect"

        bout_day = pd.to_datetime(fight["Date"], dayfirst=True, errors="coerce")
        if pd.notna(bout_day):
            seconds = [durations.get((bout_day.normalize(), fighter)) for fighter in (p1, p2)]
            seconds = [value for value in seconds if value is not None]
            if seconds:
                duration = float(pd.Series(seconds).median())
                history.at[idx, "actual_duration_seconds"] = duration
                for column, threshold in (("over_1_5_status", 450), ("over_2_5_status", 750)):
                    if history.at[idx, column] == "Pending":
                        predicted_over = pd.to_numeric(prediction.get(
                            "predicted_" + column.replace("_status", "")
                        ), errors="coerce")
                        if pd.notna(predicted_over):
                            actual_over = duration > threshold
                            predicted_side = float(predicted_over) >= 50
                            settled = "Correct" if predicted_side == actual_over else "Incorrect"
                            history.at[idx, column] = settled
                            prefix = column.replace("_status", "")
                            history.at[idx, prefix + "_profit"] = one_unit_profit(
                                settled, prediction.get(prefix + "_odds")
                            )
        changed += 1

    if changed:
        history[HISTORY_COLUMNS].to_csv(path, index=False)
    return changed


def record_best_prices(path, odds_events):
    """Snapshot best available UK decimal prices for logged winner and totals picks."""
    path = Path(path)
    if not path.exists():
        return 0
    history = pd.read_csv(path)
    if history.empty:
        return 0
    for col in HISTORY_COLUMNS:
        if col not in history.columns:
            history[col] = pd.NA
    winner_prices, total_prices = {}, {}
    for event in odds_events or []:
        a, b = str(event.get("home_team", "")).strip(), str(event.get("away_team", "")).strip()
        if not a or not b:
            continue
        pair = frozenset((fighter_key(a), fighter_key(b)))
        winners, totals = {}, {}
        for book in event.get("bookmakers", []):
            title = book.get("title", book.get("key", "UK bookmaker"))
            for market in book.get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        name = fighter_key(outcome.get("name", ""))
                        try:
                            price = float(outcome.get("price", 0))
                        except (TypeError, ValueError):
                            continue
                        if price > winners.get(name, (0, ""))[0]:
                            winners[name] = (price, title)
                elif market.get("key") == "totals":
                    for outcome in market.get("outcomes", []):
                        try:
                            price, point = float(outcome.get("price", 0)), float(outcome.get("point"))
                        except (TypeError, ValueError):
                            continue
                        side = str(outcome.get("name", "")).strip().title()
                        if side not in ("Over", "Under") or price <= 1:
                            continue
                        slot = (str(point), side)
                        if price > totals.get(slot, (0, ""))[0]:
                            totals[slot] = (price, title)
        winner_prices[pair], total_prices[pair] = winners, totals
    changed = 0
    for idx, row in history.iterrows():
        pair = frozenset((fighter_key(row.get("fighter1")), fighter_key(row.get("fighter2"))))
        winner = fighter_key(row.get("predicted_winner"))
        old_price = row.get("predicted_odds")
        if pd.isna(old_price) or not str(old_price).strip():
            quote = winner_prices.get(pair, {}).get(winner)
            if quote:
                history.at[idx, "predicted_odds"], history.at[idx, "odds_bookmaker"] = quote
                changed += 1
        for line, prefix in (("1.5", "over_1_5"), ("2.5", "over_2_5")):
            side = str(row.get("predicted_" + prefix + "_side", "")).strip()
            if side not in ("Over", "Under"):
                continue
            current = row.get(prefix + "_odds")
            if pd.notna(current) and str(current).strip():
                continue
            quote = total_prices.get(pair, {}).get((line, side))
            if quote:
                history.at[idx, prefix + "_odds"] = quote[0]
                history.at[idx, prefix + "_bookmaker"] = quote[1]
                changed += 1
    if changed:
        history[HISTORY_COLUMNS].to_csv(path, index=False)
    return changed


def one_unit_profit(status, odds):
    """Return net profit for a settled one-unit selection at captured decimal odds."""
    try:
        price = float(odds)
    except (TypeError, ValueError):
        return None
    if price <= 1 or status not in ("Correct", "Incorrect"):
        return None
    return round(price - 1, 4) if status == "Correct" else -1.0
