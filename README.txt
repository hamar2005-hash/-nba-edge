NBA PLAYER PROPS MODEL — COMPLETE PACKAGE
========================================

WHAT IT DOES
- Trains separate player-game models for points, rebounds and assists.
- Uses previous-game rolling form (last 5 and 10), season-to-date form, rest days and home/away.
- Evaluates on a chronological holdout (later dates are kept out of training).
- Scores a supplied list of player prop lines and, when decimal odds are supplied, calculates implied probability, estimated edge and expected value per unit.

IMPORTANT LIMITATIONS
- This is a research prototype, not a proven profitable betting system.
- NBA Stats may time out or block Google Colab. The collector downloads one month at a time and caches successful months so it can resume. If NBA Stats is still unreachable, stop; do not repeatedly hammer the endpoint. We will need a permitted alternative data source/export.
- The model uses historical game logs. Listed minutes are an input adjustment; injuries, starting lineups, role changes, opponent defensive strength and late scratches are not fully modelled.
- Probability estimates use a normal approximation to historical residuals and need calibration testing. Do not stake based on an unvalidated edge.
- The current season may be incomplete; training on partial seasons is expected.

GOOGLE COLAB SETUP
1. Upload and extract this ZIP in Colab, or upload the files into the same working folder as the notebook.
2. Install dependencies: !pip -q install -r requirements.txt
3. Collect logs: !python collect_nba_data.py --start-season 2021-22 --end-season 2026-27
4. Train: !python train_models.py
5. Make a copy of todays_props_template.csv named todays_props.csv and replace the example row(s) with real props.
6. Predict: !python predict_props.py --props todays_props.csv

COLLECTOR RESUME BEHAVIOUR
- Each month is cached under data/monthly_cache.
- If a run stops, rerun the same collection command. Cached months are skipped and failed months are retried.
- A partial combined CSV can be produced when some months fail. Check data/download_failures.txt before training; for a complete historical dataset, rerun collection until all needed months are present.

INPUT FORMAT
Required columns: game_date, player, team, opponent, home, minutes, market, line
market must be points, rebounds or assists; home is 1 for home and 0 for away; odds are decimal.
Optional: bookmaker, odds, side (over or under).

OUTPUTS
- models/points_model.joblib, rebounds_model.joblib, assists_model.joblib
- models/holdout_metrics.json
- models/latest_player_features.csv
- predictions.csv
