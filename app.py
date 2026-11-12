from pathlib import Path
import io, json, subprocess, sys, zipfile
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA, MODELS = ROOT / 'data', ROOT / 'models'
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

MODEL_BUNDLE = ROOT / 'NBA_Trained_Models.zip'

if MODEL_BUNDLE.exists():
    required = [MODELS / f'{m}_model.joblib' for m in ['points', 'rebounds', 'assists']]
    if not all(p.exists() for p in required):
        with zipfile.ZipFile(MODEL_BUNDLE, 'r') as z:
            z.extractall(ROOT)
PREDICTIONS = DATA / 'predictions.csv'
RESULTS = DATA / 'settled_results.csv'

st.set_page_config(page_title='NBA Edge | Player Props', page_icon='🏀', layout='wide')
st.title('🏀 NBA Edge')
st.caption('NBA player projections • bookmaker comparison • results tracking')

MARKETS = ['points', 'rebounds', 'assists']
REQUIRED_PROPS = {'game_date','player','team','opponent','home','minutes','market','line'}
ODDS_COLUMNS = ['game_date','player','team','opponent','market','line','bookmaker','side','odds']


def read_csv_upload(upload, label):
    try:
        return pd.read_csv(upload)
    except Exception as exc:
        st.error(f'Could not read {label}: {exc}')
        return None


def get_predictions():
    if PREDICTIONS.exists():
        try: return pd.read_csv(PREDICTIONS)
        except Exception: return pd.DataFrame()
    return pd.DataFrame()


def save_predictions(df):
    df.to_csv(PREDICTIONS, index=False)


def decimal_odds(value):
    return pd.to_numeric(value, errors='coerce')


def style_value_bets(df):
    if df.empty: return df
    cols = [c for c in ['game_date','player','team','opponent','market','line','projection','bookmaker','side','odds','prob_over','prob_under','implied_probability','edge','expected_value_per_unit'] if c in df]
    out = df[cols].copy()
    for c in ['prob_over','prob_under','implied_probability','edge']:
        if c in out: out[c] = (pd.to_numeric(out[c], errors='coerce') * 100).round(1)
    if 'expected_value_per_unit' in out:
        out['expected_value_per_unit'] = pd.to_numeric(out['expected_value_per_unit'], errors='coerce').round(3)
    return out

with st.sidebar:
    st.header('Model status')
    model_files = [MODELS / f'{m}_model.joblib' for m in MARKETS]
    is_trained = all(p.exists() for p in model_files) and (MODELS / 'latest_player_features.csv').exists()
    if is_trained: st.success('Models ready')
    else: st.warning('Models need training')
    st.caption('Train the model once using historical NBA player-game logs. Daily prop lines can then be uploaded as a CSV.')
    if st.button('Refresh dashboard'):
        st.rerun()

# Summary metrics
pred = get_predictions()
if not pred.empty:
    st.subheader('Latest prediction run')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Prop rows', f'{len(pred):,}')
    c2.metric('Players', f'{pred.player.nunique():,}' if 'player' in pred else '—')
    c3.metric('Bookmakers', f'{pred.bookmaker.nunique():,}' if 'bookmaker' in pred else 'Manual / not supplied')
    if 'expected_value_per_unit' in pred:
        ev = pd.to_numeric(pred.expected_value_per_unit, errors='coerce')
        c4.metric('Positive EV rows', f'{int((ev > 0).sum()):,}')
    else: c4.metric('Positive EV rows','—')
else:
    st.info('No predictions saved yet. Train the model and upload today’s player prop lines to get started.')

home, run_tab, odds_tab, results_tab, train_tab, guide_tab = st.tabs([
    'Dashboard', 'Run predictions', 'Bookmaker comparison', 'Results tracker', 'Train model', 'How to use'
])

with home:
    st.subheader('Current predictions')
    if pred.empty:
        st.write('Your latest predictions will appear here after a run.')
    else:
        show = style_value_bets(pred)
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button('Download latest predictions CSV', pred.to_csv(index=False), 'nba_predictions.csv', 'text/csv')
    st.divider()
    st.subheader('What the dashboard covers')
    a,b,c = st.columns(3)
    a.markdown('**Player projections**\n\nPoints, rebounds and assists from the existing machine-learning models.')
    b.markdown('**All bookmaker prices**\n\nEvery bookmaker and line supplied in your odds CSV stays visible. Best available prices are highlighted in the comparison tab.')
    c.markdown('**Performance tracking**\n\nUpload actual player stats to settle predictions and track hit rate by market and bookmaker.')

with run_tab:
    st.subheader('Run today’s predictions')
    st.write('Upload one row per bookmaker, player, market, line and side. If a player has prices at five bookmakers, include all five rows (and both sides when available).')
    template = pd.DataFrame([
        {'game_date':'2026-11-12','player':'Example Player','team':'BOS','opponent':'NYK','home':1,'minutes':30,'market':'points','line':20.5,'bookmaker':'Sky Bet','odds':1.91,'side':'over'},
        {'game_date':'2026-11-12','player':'Example Player','team':'BOS','opponent':'NYK','home':1,'minutes':30,'market':'points','line':21.5,'bookmaker':'bet365','odds':2.05,'side':'under'},
    ])
    st.download_button('Download daily odds CSV template', template.to_csv(index=False), 'nba_daily_odds_template.csv', 'text/csv')
    props_file = st.file_uploader('Upload today’s player props and bookmaker odds (.csv)', type=['csv'], key='daily_props')
    if props_file:
        props = read_csv_upload(props_file, 'daily props')
        if props is not None:
            missing = REQUIRED_PROPS - set(props.columns)
            if missing: st.error(f'Missing required columns: {sorted(missing)}')
            else:
                props['market'] = props.market.astype(str).str.lower().str.strip()
                invalid = sorted(set(props.market.dropna()) - set(MARKETS))
                if invalid: st.warning(f'Unsupported markets will be skipped: {invalid}')
                st.write('Upload preview')
                st.dataframe(props.head(20), use_container_width=True, hide_index=True)
                if st.button('Generate NBA predictions', type='primary', disabled=not is_trained or bool(missing)):
                    input_path = DATA / 'todays_props.csv'
                    output_path = PREDICTIONS
                    props.to_csv(input_path, index=False)
                    command = [sys.executable, str(ROOT/'predict_props.py'), '--props', str(input_path), '--models', str(MODELS), '--features', str(MODELS/'latest_player_features.csv'), '--out', str(output_path)]
                    with st.spinner('Generating projections and pricing every uploaded line…'):
                        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
                    if result.returncode:
                        st.error('Prediction run failed. Check player names, market values and model files.')
                        st.code(result.stderr or result.stdout)
                    else:
                        new_pred = pd.read_csv(output_path)
                        st.success(f'Generated {len(new_pred):,} prediction rows.')
                        st.dataframe(style_value_bets(new_pred), use_container_width=True, hide_index=True)
                        st.download_button('Download predictions', new_pred.to_csv(index=False), 'nba_predictions.csv', 'text/csv')
                        st.caption(result.stdout)
    if not is_trained:
        st.warning('Train the three models in the Train model tab before generating predictions.')

with odds_tab:
    st.subheader('Compare every bookmaker')
    if pred.empty:
        st.info('Run predictions first. Every uploaded bookmaker row will be retained here.')
    else:
        odds = pred.copy()
        for col in ['bookmaker','side','odds']:
            if col not in odds: odds[col] = np.nan
        odds['odds'] = decimal_odds(odds.odds)
        if 'edge' not in odds: odds['edge'] = np.nan
        if 'expected_value_per_unit' not in odds: odds['expected_value_per_unit'] = np.nan
        filters1, filters2, filters3 = st.columns(3)
        players = sorted(odds.player.dropna().astype(str).unique()) if 'player' in odds else []
        books = sorted(odds.bookmaker.dropna().astype(str).unique())
        markets = sorted(odds.market.dropna().astype(str).unique()) if 'market' in odds else []
        selected_players = filters1.multiselect('Players', players, default=players)
        selected_books = filters2.multiselect('Bookmakers', books, default=books)
        selected_markets = filters3.multiselect('Markets', markets, default=markets)
        if 'player' in odds: odds = odds[odds.player.astype(str).isin(selected_players)]
        odds = odds[odds.bookmaker.astype(str).isin(selected_books)]
        if 'market' in odds: odds = odds[odds.market.astype(str).isin(selected_markets)]
        if 'odds' in odds: odds = odds[odds.odds.notna()]
        if not odds.empty:
            odds['best_price'] = odds.groupby(['player','market','line','side'], dropna=False).odds.transform('max')
            odds['is_best'] = odds.odds.eq(odds.best_price)
            view_cols = [c for c in ['player','team','opponent','market','line','side','bookmaker','odds','best_price','projection','prob_over','prob_under','edge','expected_value_per_unit','is_best'] if c in odds]
            view = odds[view_cols].copy()
            for c in ['prob_over','prob_under','edge']:
                if c in view: view[c] = (pd.to_numeric(view[c], errors='coerce')*100).round(1)
            st.dataframe(view.sort_values(['player','market','line','side','odds'], ascending=[True,True,True,True,False]), use_container_width=True, hide_index=True)
            st.caption('Best price is compared only for the same player, market, line and side. Different lines remain separate.')
            st.download_button('Download bookmaker comparison', view.to_csv(index=False), 'nba_bookmaker_comparison.csv', 'text/csv')
        else: st.info('No odds rows match these filters.')

with results_tab:
    st.subheader('Settle predictions and track accuracy')
    st.write('Upload actual player stats after games finish. Required columns: player, market, actual. Optional: game_date, team.')
    result_template = pd.DataFrame([{'game_date':'2026-11-12','player':'Example Player','market':'points','actual':24}])
    st.download_button('Download actual results template', result_template.to_csv(index=False), 'nba_actual_results_template.csv', 'text/csv')
    actual_file = st.file_uploader('Upload actual player results (.csv)', type=['csv'], key='actual_results')
    if actual_file:
        actual = read_csv_upload(actual_file, 'actual results')
        if actual is not None:
            needed = {'player','market','actual'}
            if needed - set(actual.columns): st.error(f'Missing columns: {sorted(needed-set(actual.columns))}')
            elif pred.empty: st.warning('Generate or load predictions first.')
            else:
                actual['market'] = actual.market.astype(str).str.lower().str.strip()
                settled = pred.merge(actual[['player','market','actual']], on=['player','market'], how='inner')
                if 'side' not in settled: settled['side'] = 'over'
                if 'line' in settled:
                    side = settled.side.astype(str).str.lower()
                    settled['outcome'] = np.where(settled.actual == settled.line, 'push', np.where(((settled.actual > settled.line) & (side == 'over')) | ((settled.actual < settled.line) & (side == 'under')), 'win', 'loss'))
                    settled['hit'] = settled.outcome.eq('win')
                    st.dataframe(settled[[c for c in ['player','market','line','side','bookmaker','odds','projection','actual','outcome'] if c in settled]], use_container_width=True, hide_index=True)
                    wins = int(settled.outcome.eq('win').sum()); losses = int(settled.outcome.eq('loss').sum()); pushes = int(settled.outcome.eq('push').sum())
                    x,y,z = st.columns(3); x.metric('Wins',wins); y.metric('Losses',losses); z.metric('Pushes',pushes)
                    old = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()
                    combined = pd.concat([old, settled], ignore_index=True).drop_duplicates(keep='last')
                    combined.to_csv(RESULTS,index=False)
                    st.download_button('Download settled results history', combined.to_csv(index=False), 'nba_settled_results.csv', 'text/csv')
                else: st.error('Predictions file has no line column; regenerate predictions from the odds template.')
    if RESULTS.exists():
        history = pd.read_csv(RESULTS)
        if not history.empty:
            st.divider(); st.subheader('Saved results history')
            decided = history[history.outcome.isin(['win','loss'])] if 'outcome' in history else pd.DataFrame()
            if not decided.empty:
                hit_rate = decided.hit.mean()*100 if 'hit' in decided else (decided.outcome=='win').mean()*100
                st.metric('Historical hit rate (excluding pushes)', f'{hit_rate:.1f}%')
                if 'bookmaker' in decided:
                    book_stats = decided.groupby('bookmaker').outcome.apply(lambda s: (s=='win').sum() / max(1,(s.isin(['win','loss'])).sum()) * 100).rename('hit_rate_pct').reset_index()
                    st.dataframe(book_stats.round(1), use_container_width=True, hide_index=True)

with train_tab:
    st.subheader('Train the player-stat models')
    st.write('Upload historical NBA player-game logs. The required columns are PLAYER_ID, PLAYER_NAME, GAME_DATE, GAME_ID, MATCHUP, MIN, PTS, REB and AST.')
    hist_template = st.download_button('Download training columns guide', 'PLAYER_ID,PLAYER_NAME,GAME_DATE,GAME_ID,MATCHUP,MIN,PTS,REB,AST\n', 'historical_data_columns.csv', 'text/csv')
    hist_file = st.file_uploader('Historical player-game logs (.csv)', type=['csv'], key='history_upload')
    if hist_file:
        st.caption(f'{hist_file.name} • {hist_file.size:,} bytes')
    st.markdown('#### Automatic historical data download')
    st.caption('Try the free NBA Stats data collector first. NBA Stats may block cloud-hosted requests; if that happens, use the CSV upload option below.')
    c1, c2 = st.columns(2)
    start_season = c1.selectbox('Start season', ['2021-22','2022-23','2023-24','2024-25','2025-26'], index=0)
    end_season = c2.selectbox('End season', ['2025-26','2026-27'], index=1)
    if st.button('Download historical logs and train models', type='primary', disabled=is_trained):
        hist_path = DATA/'player_game_logs.csv'
        with st.spinner('Downloading historical NBA player-game logs. This may take several minutes…'):
            collect = subprocess.run([sys.executable, str(ROOT/'collect_nba_data.py'), '--start-season', start_season, '--end-season', end_season, '--out', str(hist_path)], cwd=ROOT, capture_output=True, text=True, timeout=2400)
        if collect.returncode:
            st.error('The free NBA Stats feed could not be reached from the hosting server. No paid service is required; you can still upload a historical CSV manually.')
            st.code((collect.stderr or collect.stdout)[-12000:])
        else:
            st.success(f'Downloaded historical logs: {hist_path.stat().st_size:,} bytes')
            with st.spinner('Training points, rebounds and assists models…'):
                train = subprocess.run([sys.executable, str(ROOT/'train_models.py'), '--input', str(hist_path), '--outdir', str(MODELS)], cwd=ROOT, capture_output=True, text=True, timeout=1800)
            if train.returncode:
                st.error('Data downloaded, but model training failed.')
                st.code(train.stderr or train.stdout)
            else:
                st.success('All three models trained successfully.')
                st.code(train.stdout)
                st.info('Download the trained model bundle below, then add its models/ files to GitHub so they survive app restarts.')
                st.rerun()
    if st.button('Train points, rebounds and assists', type='primary', disabled=hist_file is None):
        try:
            hist_path = DATA/'player_game_logs.csv'; hist_path.write_bytes(hist_file.getvalue())
            with st.spinner('Training models — this can take several minutes…'):
                run = subprocess.run([sys.executable,str(ROOT/'train_models.py'),'--input',str(hist_path),'--outdir',str(MODELS)],cwd=ROOT,capture_output=True,text=True,timeout=1800)
            if run.returncode:
                st.error('Training failed.'); st.code(run.stderr or run.stdout)
            else:
                st.success('Training completed.'); st.code(run.stdout)
                metric_path = MODELS/'holdout_metrics.json'
                if metric_path.exists(): st.dataframe(pd.DataFrame(json.loads(metric_path.read_text())).T.reset_index(names='market'),use_container_width=True,hide_index=True)
                st.info('Download the trained model bundle and add its models/ folder to your deployed app repository so the model is available after restarts.')
        except subprocess.TimeoutExpired: st.error('Training timed out. Try a smaller or cleaner historical CSV.')
        except Exception as exc: st.exception(exc)
    if is_trained:
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
            for f in MODELS.glob('*'):
                if f.is_file(): z.write(f,arcname=f'models/{f.name}')
        st.download_button('Download trained model files',bundle.getvalue(),'nba_trained_models.zip','application/zip')
        metrics_path = MODELS/'holdout_metrics.json'
        if metrics_path.exists():
            st.subheader('Chronological holdout metrics')
            st.dataframe(pd.DataFrame(json.loads(metrics_path.read_text())).T.reset_index(names='market'),use_container_width=True,hide_index=True)

with guide_tab:
    st.subheader('How this app works')
    st.markdown('''
**One-time setup**
1. Upload historical player-game logs in **Train model**.
2. Train the points, rebounds and assists models.
3. Keep the trained `models/` folder with the deployed app.

**Each NBA day**
1. Download the daily odds template.
2. Add one row for every bookmaker/player/market/line/side you can access. Odds are decimal (for example, 1.91).
3. Upload the CSV in **Run predictions** and generate predictions.
4. Use **Bookmaker comparison** to compare all submitted prices. Best price is flagged for matching lines only.
5. After games, upload actual player stats in **Results tracker**.

**Important limitations**
- The model currently predicts points, rebounds and assists only. PRA, team totals, game scores, injury adjustments and lineup news are not included in this trained model.
- It does not scrape bookmaker websites. Odds are only as complete and current as the CSV you provide or a future legitimate free feed.
- This is a statistical research tool, not betting advice. Probabilities are model estimates and must be validated; never stake money you cannot afford to lose.
- Streamlit Community Cloud has ephemeral local storage. Download your predictions/results regularly; for durable automated history, a persistent database or connected storage will need to be added.
''')
    st.subheader('Putting it online')
    st.markdown('''
1. Create a **private GitHub repository** for this project and upload the files from this ZIP.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and connect that repository.
3. Set the main file to `app.py` and deploy.
4. Open the generated website on your iPhone and use **Share → Add to Home Screen**.

The app package is ready to deploy, but it is **not yet a live website**. Your GitHub account currently shows no repository available to this project, and hosting requires you to create/select a repository and authorize deployment. No paid odds service is required for the manual workflow.
''')
    st.markdown('[Streamlit deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)')
