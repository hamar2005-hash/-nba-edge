"""Build and email NBA Edge's daily accuracy report.

Reads settled_results.csv and prop_pick_history.csv, writes a dashboard-friendly
JSON snapshot, and optionally emails an HTML/plain-text summary over SMTP.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from email.message import EmailMessage
from html import escape
import json
import os
from pathlib import Path
import smtplib
import ssl
from typing import Any

import pandas as pd

DATA = Path("data")
SETTLED = DATA / "settled_results.csv"
PICKS = DATA / "prop_pick_history.csv"
REPORT = DATA / "accuracy_report.json"
MARKETS = {
    "Points": ("points_absolute_error", "points_projection", "actual_points"),
    "Rebounds": ("rebounds_absolute_error", "rebounds_projection", "actual_rebounds"),
    "Assists": ("assists_absolute_error", "assists_projection", "actual_assists"),
    "Three-pointers made": ("threes_absolute_error", "threes_projection", "actual_threes"),
}


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _date_frame(frame: pd.DataFrame, report_date: date) -> pd.DataFrame:
    if frame.empty or "game_date" not in frame.columns:
        return pd.DataFrame()
    result = frame.copy()
    result["_date"] = pd.to_datetime(result["game_date"], errors="coerce").dt.date
    return result[result["_date"].notna() & (result["_date"] <= report_date)]


def market_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for market, (error_col, projection_col, actual_col) in MARKETS.items():
        required = {projection_col, actual_col}
        if frame.empty or not required.issubset(frame.columns):
            output[market] = {"players": 0, "mae": None, "within_3": None, "bias": None}
            continue
        valid = frame.copy()
        valid[projection_col] = pd.to_numeric(valid[projection_col], errors="coerce")
        valid[actual_col] = pd.to_numeric(valid[actual_col], errors="coerce")
        valid = valid.dropna(subset=[projection_col, actual_col])
        if valid.empty:
            output[market] = {"players": 0, "mae": None, "within_3": None, "bias": None}
            continue
        errors = valid[projection_col] - valid[actual_col]
        absolute = errors.abs()
        output[market] = {
            "players": int(len(valid)),
            "mae": round(float(absolute.mean()), 2),
            "within_3": round(float((absolute <= 3).mean() * 100), 1),
            "bias": round(float(errors.mean()), 2),
        }
    return output


def pick_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "Actual result" not in frame.columns:
        return {"settled": 0, "wins": 0, "losses": 0, "pushes": 0, "hit_rate": None}
    results = frame["Actual result"].astype(str).str.strip().str.title()
    wins = int((results == "Win").sum())
    losses = int((results == "Loss").sum())
    pushes = int((results == "Push").sum())
    graded = wins + losses + pushes
    decided = wins + losses
    return {
        "settled": graded,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": round(wins / decided * 100, 1) if decided else None,
    }


def build_report(report_date: date) -> dict[str, Any]:
    settled = _date_frame(_load_csv(SETTLED), report_date)
    picks = _date_frame(_load_csv(PICKS), report_date)
    windows: dict[str, Any] = {}
    for label, days in (("daily", 1), ("7d", 7), ("30d", 30)):
        start = report_date - timedelta(days=days - 1)
        settled_window = settled[settled["_date"] >= start] if not settled.empty else settled
        picks_window = picks[picks["_date"] >= start] if not picks.empty else picks
        markets = market_metrics(settled_window)
        total_players = sum(item["players"] for item in markets.values())
        weighted_abs_error = sum(
            item["mae"] * item["players"]
            for item in markets.values()
            if item["mae"] is not None and item["players"]
        )
        total_market_players = sum(item["players"] for item in markets.values())
        windows[label] = {
            "start_date": start.isoformat(),
            "end_date": report_date.isoformat(),
            "settled_player_market_rows": int(total_market_players),
            "markets": markets,
            "overall_mae": round(weighted_abs_error / total_market_players, 2) if total_market_players else None,
            "prop_picks": pick_metrics(picks_window),
        }

    daily_players = windows["daily"]["settled_player_market_rows"]
    return {
        "report_date": report_date.isoformat(),
        "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "settlement_status": "settled" if daily_players else "no_settled_rows",
        "windows": windows,
    }


def render_email(report: dict[str, Any]) -> tuple[str, str]:
    report_date = report["report_date"]
    windows = report["windows"]
    daily = windows["daily"]
    rows = []
    text_rows = []
    for market, values in daily["markets"].items():
        mae = "n/a" if values["mae"] is None else str(values["mae"])
        within = "n/a" if values["within_3"] is None else f'{values["within_3"]}%'
        rows.append(
            "<tr>"
            f"<td style='padding:8px;border-bottom:1px solid #ddd'>{escape(market)}</td>"
            f"<td style='padding:8px;text-align:right;border-bottom:1px solid #ddd'>{values['players']}</td>"
            f"<td style='padding:8px;text-align:right;border-bottom:1px solid #ddd'>{mae}</td>"
            f"<td style='padding:8px;text-align:right;border-bottom:1px solid #ddd'>{within}</td>"
            "</tr>"
        )
        text_rows.append(f"{market}: {values['players']} player-markets | MAE {mae} | within 3: {within}")

    pick = daily["prop_picks"]
    hit = "n/a" if pick["hit_rate"] is None else f'{pick["hit_rate"]}%'
    summary = (
        f"Daily prop picks: {pick['wins']}-{pick['losses']}-{pick['pushes']} "
        f"(hit rate {hit}, pushes excluded)."
    )
    trend_lines = []
    for label, title in (("7d", "Last 7 days"), ("30d", "Last 30 days")):
        window = windows[label]
        mae = "n/a" if window["overall_mae"] is None else str(window["overall_mae"])
        p = window["prop_picks"]
        p_hit = "n/a" if p["hit_rate"] is None else f'{p["hit_rate"]}%'
        trend_lines.append(
            f"<li><strong>{title}:</strong> overall MAE {mae}; "
            f"prop picks {p['wins']}-{p['losses']}-{p['pushes']} (hit rate {p_hit})</li>"
        )

    status = (
        "Final box-score data was found and accuracy was updated."
        if report["settlement_status"] == "settled"
        else "No settled player rows were available for this report date. The report will update after settlement data arrives."
    )
    html_body = f"""<!doctype html><html><body style="font-family:Arial,sans-serif;color:#172033;line-height:1.5">
    <h2>NBA Edge daily accuracy report</h2><p><strong>{escape(report_date)}</strong></p>
    <p>{escape(status)}</p>
    <h3>Yesterday's model accuracy</h3>
    <table style="border-collapse:collapse;width:100%;max-width:680px">
    <thead><tr><th style="text-align:left;padding:8px">Market</th><th style="text-align:right;padding:8px">Players</th><th style="text-align:right;padding:8px">MAE</th><th style="text-align:right;padding:8px">Within 3</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table>
    <p>{escape(summary)}</p><h3>Trend check</h3><ul>{''.join(trend_lines)}</ul>
    <p style="color:#667085;font-size:12px">MAE is lower-is-better. Prop hit rate excludes pushes. Historical performance is not a guarantee of future results.</p>
    </body></html>"""
    plain = (
        f"NBA Edge daily accuracy report - {report_date}\n\n{status}\n\n"
        "Yesterday's model accuracy:\n" + "\n".join(text_rows) + "\n\n"
        + summary + "\n\nTrend check:\n" + "\n".join(
            line.replace("<strong>", "").replace("</strong>", "") for line in trend_lines
        )
        + "\n\nMAE is lower-is-better. Prop hit rate excludes pushes."
    )
    return html_body, plain


def send_email(report: dict[str, Any]) -> bool:
    recipient = os.getenv("NBA_EDGE_REPORT_EMAIL", "").strip()
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "")
    sender = os.getenv("NBA_EDGE_FROM_EMAIL", "").strip() or username
    if not all((recipient, username, password, sender)):
        print("Email not sent: configure NBA_EDGE_REPORT_EMAIL, SMTP_USERNAME, SMTP_PASSWORD, and NBA_EDGE_FROM_EMAIL GitHub Actions secrets.")
        return False

    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "465"))
    security = os.getenv("SMTP_SECURITY", "ssl").lower()
    html_body, plain_body = render_email(report)
    message = EmailMessage()
    message["Subject"] = f"NBA Edge accuracy report | {report['report_date']}"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(plain_body)
    message.add_alternative(html_body, subtype="html")

    if security == "starttls":
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            server.login(username, password)
            server.send_message(message)
    else:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as server:
            server.login(username, password)
            server.send_message(message)
    print(f"Accuracy report emailed to {recipient}.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Report date YYYY-MM-DD; defaults to yesterday in US Eastern time")
    parser.add_argument("--no-email", action="store_true", help="Build the JSON report without attempting email")
    args = parser.parse_args()
    if args.date:
        report_date = date.fromisoformat(args.date)
    else:
        from zoneinfo import ZoneInfo
        report_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
    report = build_report(report_date)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not args.no_email:
        send_email(report)


if __name__ == "__main__":
    main()
