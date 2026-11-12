# NBA Edge — Automated NBA Projections

A Streamlit dashboard for automated player projections. The current models cover points, rebounds and assists.

## Run locally

```bash
pip install -r requirements.txt
python auto_daily.py
streamlit run app.py
```

## Deploy

1. Connect this GitHub repository to Streamlit Community Cloud.
2. Deploy `app.py`.
3. Ensure `NBA_Trained_Models.zip` contains the three trained model files and `latest_player_features.csv` under `models/`.
4. The GitHub Actions workflow generates projections three times daily and commits the latest CSV. It can also be run manually from GitHub Actions.

## What is displayed

- Today's scheduled games and player list
- Projected points, rebounds and assists
- Recent historical minutes for context
- Game and player filters

## No bookmaker data

This version does not request, display, compare or calculate bookmaker odds, lines, prices or expected value. No daily CSV upload is required.

## Limitations

The saved historical feature snapshot is not retrained each day. Recent historical minutes are not a confirmed minutes forecast. ESPN roster data may not contain complete or timely injury and starting-lineup information. The current models do not cover combo markets or team totals. Projections are estimates, not guarantees.
