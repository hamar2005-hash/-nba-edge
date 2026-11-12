# NBA Edge — NBA Predictions & Bookmaker Comparison

A Streamlit web app built around the existing NBA player-prop ML workflow. It combines:

- Training models for **points, rebounds and assists** from historical player-game logs.
- Running projections from a daily props CSV.
- Comparing **every bookmaker/line/side supplied** in the uploaded CSV, with the best decimal price highlighted for identical player + market + line + side.
- Uploading actual player stats and tracking settled results.
- Downloading prediction, comparison and results CSVs.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free with Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of this folder.
2. Go to https://share.streamlit.io/ and connect that repository.
3. Choose `app.py` as the app entry point and deploy.
4. Open the resulting URL on your iPhone. In Safari, use **Share → Add to Home Screen**.

The app package is ready for deployment, but it is **not a live hosted website yet**. You must deploy it to a hosting account before it has a public URL.

## One-time model setup

1. Open **Train model**.
2. Upload historical player-game logs containing `PLAYER_ID, PLAYER_NAME, GAME_DATE, GAME_ID, MATCHUP, MIN, PTS, REB, AST`.
3. Train the three models.
4. Download and retain the trained model bundle. For Streamlit Cloud, commit the generated `models/` files to your private repository so they survive app restarts.

The data collector `collect_nba_data.py` attempts to download historical logs from NBA Stats. NBA Stats can throttle/block cloud requests; if that happens, use a permitted historical CSV export instead.

## Daily workflow

1. In **Run predictions**, download `nba_daily_odds_template.csv`.
2. Add one row for each player/market/line/bookmaker/side you have. Include every bookmaker you want compared. Decimal odds are expected (e.g. `1.91`).
3. Required columns: `game_date,player,team,opponent,home,minutes,market,line`.
4. Optional odds columns: `bookmaker,odds,side`. `side` should be `over` or `under`.
5. Upload the CSV and generate projections. The comparison tab retains all supplied rows.
6. After games finish, upload actual stats using `nba_actual_results_template.csv` in **Results tracker**.

## Current model scope — important

The existing trained model estimates **points, rebounds and assists** only. It does not currently model PRA/combo markets, team totals, game scores, injuries, confirmed lineups, or automatic player minutes. It does not scrape bookmaker websites. The app can compare all bookmakers whose odds you supply, and can accept manually assembled odds CSVs. Free odds feeds may be added later if a reliable provider and market coverage are available.

The current app requires a daily props CSV with expected minutes and lines. It therefore does **not yet produce a fully automatic daily slate on its own**. Daily scheduling, automatic slate/minutes ingestion, and durable cloud storage are future integration steps. Streamlit Community Cloud local files may be ephemeral; export your predictions/results regularly.

## Files

- `app.py` — dashboard
- `collect_nba_data.py` — historical data collector
- `train_models.py` — model training
- `predict_props.py` — projections and probability estimates
- `nba_daily_odds_template.csv` — daily props/odds template
- `nba_actual_results_template.csv` — actual-results template

This is a statistical research tool, not a guarantee of betting outcomes. Model probabilities need calibration and out-of-sample validation before being used to make staking decisions.
