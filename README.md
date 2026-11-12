# NBA Edge

Automated NBA player projections built with Streamlit and the existing trained machine-learning models.

## What it does

- Automatically reads the NBA schedule and team rosters from ESPN's public data feed.
- Generates player projections for **points, rebounds and assists** using the saved trained models and historical player feature snapshot.
- Displays today's projections in the Streamlit dashboard, with game and player filters.
- Refreshes on a GitHub Actions schedule and can also be refreshed manually from the app.

## No bookmaker data

NBA Edge does not request, display, compare or calculate bookmaker odds, betting lines, prices or expected value.

## Setup

The repository must contain `NBA_Trained_Models.zip` with these files under the `models/` directory:

- `points_model.joblib`
- `rebounds_model.joblib`
- `assists_model.joblib`
- `latest_player_features.csv`

The GitHub Actions workflow requires repository Contents write permission so it can commit the daily CSV. Streamlit Community Cloud should be connected to the same repository and `app.py` as the entry point.

## Schedule

The GitHub Actions workflow runs at 16:00, 20:00 and 23:00 UTC every day, and can also be run manually from the Actions tab. GitHub scheduled runs can be delayed during periods of high load.

## Important limitations

- The current models cover points, rebounds and assists only; combo props and team totals are not included.
- Player projections use the latest historical feature snapshot included in the trained bundle. They are not retrained automatically each day.
- Recent historical minutes are shown as context, not as a forecast of tonight's playing time.
- ESPN roster data may not reflect late injury news or confirmed lineups. The app does not yet have a verified live injury/starting-lineup adjustment.
- Model estimates are not guarantees. Track performance and calibration over time.

## Run locally

```bash
pip install -r requirements.txt
python auto_daily.py
streamlit run app.py
```
