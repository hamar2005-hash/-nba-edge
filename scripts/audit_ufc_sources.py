"""Download and audit the live UFC source feeds without changing model inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/Sam-Armstrong/UFC-Almanac/main/data"
SOURCES = {
    "fight_results": f"{BASE}/FightResults.csv",
    "fight_statistics": f"{BASE}/FightStats.csv",
    "fighter_profiles": f"{BASE}/FighterData.csv",
}
OUT = Path("artifacts/ufc_data_audit")
OUT.mkdir(parents=True, exist_ok=True)

PROFILE_CANDIDATES = {
    "age": ["Age", "age"],
    "height": ["Height", "height"],
    "reach": ["Reach", "reach"],
    "stance": ["Stance", "stance"],
    "weight_class": ["Weight Class", "weight_class", "Division"],
}
PERFORMANCE_CANDIDATES = {
    "knockdowns": ["Knockdowns"],
    "sig_strikes_landed": ["Sig Strikes Landed"],
    "sig_strikes_attempted": ["Sig Strikes Attempted"],
    "sig_strikes_absorbed": ["Sig Strikes Absorbed"],
    "takedowns": ["Takedowns"],
    "takedown_attempts": ["Takedown Attempts"],
    "submission_attempts": ["Submission Attempts"],
    "clinch_strikes": ["Clinch Strikes"],
    "ground_strikes": ["Ground Strikes"],
}
RESULT_REQUIRED = ["Date", "Fighter 1", "Fighter 2", "Result"]
STATS_REQUIRED = ["Date", "Name", "Time"]


def fetch_csv(name: str, url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=45, headers={"User-Agent": "UFC-Edge-data-audit/1.0"})
    response.raise_for_status()
    path = OUT / f"{name}.csv"
    path.write_bytes(response.content)
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"Source feed '{name}' downloaded successfully but contains no rows")
    return frame


def first_present(columns, candidates):
    return next((column for column in candidates if column in columns), None)


def describe_frame(name, frame):
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": [str(c) for c in frame.columns],
        "duplicate_full_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "missing_cell_rate": round(float(frame.isna().mean().mean()), 5) if len(frame.columns) else None,
    }


def main():
    frames = {name: fetch_csv(name, url) for name, url in SOURCES.items()}
    results = frames["fight_results"]
    stats = frames["fight_statistics"]
    profiles = frames["fighter_profiles"]

    result_missing = [column for column in RESULT_REQUIRED if column not in results.columns]
    stats_missing = [column for column in STATS_REQUIRED if column not in stats.columns]
    result_dates = pd.to_datetime(results["Date"], dayfirst=True, errors="coerce") if "Date" in results else pd.Series(dtype="datetime64[ns]")
    stat_dates = pd.to_datetime(stats["Date"], dayfirst=True, errors="coerce") if "Date" in stats else pd.Series(dtype="datetime64[ns]")

    profile_fields = {
        key: {
            "column": first_present(profiles.columns, candidates),
            "available": first_present(profiles.columns, candidates) is not None,
            "non_null_rate": (
                round(float(profiles[first_present(profiles.columns, candidates)].notna().mean()), 5)
                if first_present(profiles.columns, candidates) else None
            ),
        }
        for key, candidates in PROFILE_CANDIDATES.items()
    }
    performance_fields = {
        key: {
            "column": first_present(stats.columns, candidates),
            "available": first_present(stats.columns, candidates) is not None,
            "non_null_rate": (
                round(float(stats[first_present(stats.columns, candidates)].notna().mean()), 5)
                if first_present(stats.columns, candidates) else None
            ),
        }
        for key, candidates in PERFORMANCE_CANDIDATES.items()
    }

    stats_duplicate_fighter_dates = None
    if {"Date", "Name"}.issubset(stats.columns):
        normalized = stats.assign(
            _date=pd.to_datetime(stats["Date"], dayfirst=True, errors="coerce").dt.normalize(),
            _name=stats["Name"].fillna("").astype(str).str.casefold().str.strip(),
        )
        stats_duplicate_fighter_dates = int(
            normalized[normalized["_date"].notna()].duplicated(["_date", "_name"], keep=False).sum()
        )

    method_col = first_present(results.columns, ["Method", "method"])
    method_coverage = (
        round(float(results[method_col].notna().mean()), 5) if method_col else None
    )
    summary = {
        "sources": {
            name: {
                **describe_frame(name, frame),
                "date_min": str(dates.min().date()) if len(dates) and dates.notna().any() else None,
                "date_max": str(dates.max().date()) if len(dates) and dates.notna().any() else None,
            }
            for (name, frame), dates in zip(frames.items(), [result_dates, stat_dates, pd.Series(dtype="datetime64[ns]")])
        },
        "required_columns_missing": {
            "fight_results": result_missing,
            "fight_statistics": stats_missing,
        },
        "profile_fields": profile_fields,
        "performance_fields": performance_fields,
        "method_column": method_col,
        "method_label_coverage": method_coverage,
        "fight_statistics_duplicate_fighter_date_rows": stats_duplicate_fighter_dates,
        "result_invalid_dates": int(result_dates.isna().sum()) if len(result_dates) else None,
        "audit_status": "passed" if not result_missing and not stats_missing else "failed",
        "interpretation": (
            "This is a source/schema audit only. It does not claim a feature improves predictive accuracy; "
            "candidate features must pass chronological out-of-sample comparison before model promotion."
        ),
    }
    (OUT / "audit_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "audit_report.md").write_text(
        "# UFC live-source data audit\n\n"
        + "## Source summary\n"
        + "\n".join(
            f"- **{name}**: {item['rows']:,} rows, {item['columns']} columns, "
            f"{item['date_min'] or 'no date range'} to {item['date_max'] or 'no date range'}"
            for name, item in summary["sources"].items()
        )
        + "\n\n## Profile field availability\n"
        + "\n".join(f"- {key}: {value['column'] or 'not supplied'} (non-null rate: {value['non_null_rate']})" for key, value in profile_fields.items())
        + "\n\n## Fight-stat feature availability\n"
        + "\n".join(f"- {key}: {value['column'] or 'not supplied'} (non-null rate: {value['non_null_rate']})" for key, value in performance_fields.items())
        + f"\n\nMethod column: {method_col or 'not supplied'} (coverage: {method_coverage})\n"
        + f"\nAudit status: **{summary['audit_status']}**\n\n"
        + summary["interpretation"] + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if summary["audit_status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
