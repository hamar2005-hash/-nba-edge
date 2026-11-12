# NBA Edge — NBA Player Props Dashboard

A Streamlit dashboard for the existing NBA player-stat ML workflow. It includes model training, daily prediction runs, all-bookmaker odds comparison (from uploaded CSV data), and results tracking.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free

1. Create a GitHub repository and upload the contents of this project folder.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), connect the repository and deploy `app.py`.
3. Upload historical player-game logs in the **Train model** tab and train the three models.
4. Keep the trained `models/` folder with the deployed app. The app can export the trained models as a ZIP.
5. Open the Streamlit URL on your phone and add it to your home screen.

## Daily odds CSV format

Required: `game_date,player,team,opponent,home,minutes,market,line`

Optional: `bookmaker,odds,side` (decimal odds; side is `over` or `under`). Add one row per bookmaker/line/side. Repeated player-market rows are expected and retained.

## Actual results CSV format

Required: `player,market,actual`; optional `game_date,team`.

## Model scope and limitations

The existing ML package trains points, rebounds and assists regressors using historical rolling features. It does not currently model PRA, team totals, game scores, injuries, lineup news, or calibrated probabilities. It does not scrape bookmaker websites or guarantee a free odds feed. Odds comparison works with all bookmaker rows supplied in the uploaded CSV.

Streamlit Community Cloud local storage can be ephemeral. Export/download results regularly; persistent cloud database integration is a future improvement.
