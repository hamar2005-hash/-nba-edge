# NBA Edge

Automated NBA player projections built with Streamlit and the existing trained machine-learning models.

## Features

- Automatically reads the NBA schedule and team rosters from ESPN's public data feed.
- Generates points, rebounds and assists projections.
- Shows top 10 projected players for each market, with recent average minutes for context.
- Compares model projections with current NBA points, rebounds and assists prop lines from Stat Pick's public API.
- Displays projection gaps and model-estimated over/under probabilities; source bookmaker names and prices are discarded.
- Archives daily slates and compares them with completed ESPN box scores.
- Tracks mean absolute error by market and by date.
- Refreshes projections and settles completed games through GitHub Actions.
- Retrains points, rebounds and assists weekly from updated historical game logs, but only promotes a candidate model when it beats the current model on the same chronological holdout.

## No bookmaker data

NBA Edge does not display bookmaker odds, prices or expected value. It retrieves player prop line numbers only, then discards bookmaker prices and source bookmaker details.

## Model files

The repository must contain `NBA_Trained_Models.zip` with:

- `models/points_model.joblib`
- `models/rebounds_model.joblib`
- `models/assists_model.joblib`
- `models/latest_player_features.csv`

## Automation

GitHub Actions runs at 10:00, 16:00, 20:00 and 23:00 UTC. It archives the previous slate, settles yesterday's completed games from ESPN box scores, generates today's projections and commits the CSV history. Runs can be delayed by GitHub.

## Limitations

The trained models cover points, rebounds and assists only. Historical minutes are context, not a forecast of actual playing time. ESPN roster status and box-score feeds can be delayed or incomplete. Accuracy statistics only include players matched to final box scores; DNPs and unmatched names may be excluded. Model estimates are not guarantees.

## Run locally

```bash
pip install -r requirements.txt
python auto_daily.py
python settle_results.py
streamlit run app.py
```


## First basket scorer

NBA Edge includes a separate first made-field-goal scorer ranking. A weekly GitHub Actions job collects ESPN completed-game play-by-play and starting-lineup records, then builds smoothed player/team history. The daily workflow ranks players on the slate using historical starting frequency and first-basket rates. These are experimental model estimates, not calibrated probabilities; the section stays unavailable until the historical sample has been collected successfully. First basket means the first made field goal (2- or 3-point shot), not a free throw.
