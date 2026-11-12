"""Train separate points, rebounds and assists models with chronological holdout."""
from pathlib import Path
import argparse, json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

TARGETS = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST', 'threes': 'FG3M', 'steals': 'STL', 'blocks': 'BLK', 'turnovers': 'TOV'}
STATS = ['PTS','REB','AST','MIN','FGA','FG3A','FG3M','FTA','TOV','STL','BLK']


DEFENSE_STATS = ['PTS','REB','AST','FG3M','FG3A','FGA','FTA','TOV','STL','BLK']

def add_opponent_defense(df):
    """Build opponent allowed-stat trends from completed games only."""
    df = df.copy()
    df['OPPONENT'] = df['MATCHUP'].astype(str).str.extract(r'(?:vs\.|@)\s*([A-Z]{2,3})', expand=False)
    team_games = df.groupby(['GAME_ID','GAME_DATE','TEAM_ABBREVIATION'], as_index=False)[DEFENSE_STATS].sum(min_count=1)
    # For each defending team, the other team's box score is what it allowed.
    allowed = team_games.merge(team_games, on='GAME_ID', suffixes=('_def','_opp'))
    allowed = allowed[allowed['TEAM_ABBREVIATION_def'] != allowed['TEAM_ABBREVIATION_opp']].copy()
    allowed['DEF_TEAM'] = allowed['TEAM_ABBREVIATION_def']
    for stat in DEFENSE_STATS:
        allowed[f'ALLOW_{stat}'] = allowed[f'{stat}_opp']
    allowed = allowed[['GAME_ID','GAME_DATE_def','DEF_TEAM'] + [f'ALLOW_{x}' for x in DEFENSE_STATS]]
    allowed = allowed.rename(columns={'GAME_DATE_def':'GAME_DATE'})
    allowed = allowed.sort_values(['DEF_TEAM','GAME_DATE','GAME_ID']).reset_index(drop=True)
    features = []
    for stat in DEFENSE_STATS:
        col = f'ALLOW_{stat}'
        for n in (5,10):
            name = f'OPP_{col}_L{n}'
            allowed[name] = allowed.groupby('DEF_TEAM')[col].transform(
                lambda values: values.shift(1).rolling(n, min_periods=2).mean())
            features.append(name)
        name = f'OPP_{col}_SEASON_PRE'
        allowed[name] = allowed.groupby('DEF_TEAM')[col].transform(
            lambda values: values.shift(1).expanding(min_periods=2).mean())
        features.append(name)
    latest = allowed.sort_values(['GAME_DATE','GAME_ID']).groupby('DEF_TEAM',as_index=False).tail(1)
    latest = latest[['DEF_TEAM'] + features].rename(columns={'DEF_TEAM':'TEAM_ABBREVIATION'})
    df = df.merge(allowed[['GAME_ID','DEF_TEAM'] + features],
                  left_on=['GAME_ID','OPPONENT'], right_on=['GAME_ID','DEF_TEAM'], how='left')
    return df.drop(columns=['DEF_TEAM']), latest


def prepare(path):
    df = pd.read_csv(path)
    needed = {'PLAYER_ID','PLAYER_NAME','GAME_DATE','MATCHUP','MIN','PTS','REB','AST'}
    missing = needed - set(df.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
    df = df.dropna(subset=['GAME_DATE','PLAYER_ID']).copy()
    for c in STATS:
        if c not in df: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values(['PLAYER_ID','GAME_DATE','GAME_ID']).reset_index(drop=True)
    df['HOME'] = df['MATCHUP'].astype(str).str.contains(' vs. ', regex=False).astype(int)
    df['REST_DAYS'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days.clip(0,14)
    df['GAMES_PLAYED_BEFORE'] = df.groupby('PLAYER_ID').cumcount()
    for stat in STATS:
        grouped = df.groupby('PLAYER_ID')[stat]
        for n in (5,10):
            df[f'{stat}_L{n}'] = grouped.transform(lambda s: s.shift(1).rolling(n, min_periods=2).mean())
        df[f'{stat}_SEASON_PRE'] = grouped.transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
    df['OPPONENT'] = df['MATCHUP'].astype(str).str.extract(r'(?:vs\.|@)\s*([A-Z]{2,3})', expand=False)
    return df

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='data/player_game_logs.csv'); ap.add_argument('--outdir',default='models'); args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    df=prepare(args.input)
    features=[c for c in df.columns if c.endswith(('_L5','_L10','_SEASON_PRE'))] + ['REST_DAYS','HOME','GAMES_PLAYED_BEFORE']
    features=list(dict.fromkeys(features))
    dates=np.sort(df.GAME_DATE.dropna().unique())
    if len(dates)<20: raise SystemExit('Not enough distinct game dates to create a reliable chronological holdout.')
    cutoff=dates[max(1,int(len(dates)*0.8))]
    train=df[df.GAME_DATE<cutoff]; test=df[df.GAME_DATE>=cutoff]
    metrics={}
    for market,target in TARGETS.items():
        tr=train.dropna(subset=[target]); te=test.dropna(subset=[target])
        tr=tr[tr.GAMES_PLAYED_BEFORE>=2]; te=te[te.GAMES_PLAYED_BEFORE>=2]
        if len(tr)<1000 or len(te)<100: raise SystemExit(f'Insufficient rows for {market}: train={len(tr)}, test={len(te)}')
        model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('regressor',HistGradientBoostingRegressor(max_iter=250,learning_rate=.055,max_leaf_nodes=15,l2_regularization=2.0,random_state=42))])
        model.fit(tr[features],tr[target]); pred=model.predict(te[features]); residual=te[target].to_numpy()-pred
        metrics[market]={'train_rows':int(len(tr)),'test_rows':int(len(te)),'holdout_start':str(pd.Timestamp(cutoff).date()),'mae':float(mean_absolute_error(te[target],pred)),'rmse':float(np.sqrt(mean_squared_error(te[target],pred))),'residual_std':float(np.std(residual,ddof=1))}
        joblib.dump({'model':model,'feature_cols':features,'residual_std':metrics[market]['residual_std'],'target':target,'market':market},out/f'{market}_model.joblib')
        print(f"{market}: MAE {metrics[market]['mae']:.3f} | RMSE {metrics[market]['rmse']:.3f} | holdout rows {len(te):,}")
    (out/'holdout_metrics.json').write_text(json.dumps(metrics,indent=2))
    # The most recent row is a pregame feature snapshot because all rolling features are shifted.
    latest=df.sort_values('GAME_DATE').groupby('PLAYER_ID',as_index=False).tail(1)
    cols=['PLAYER_ID','PLAYER_NAME'] + (['TEAM_ABBREVIATION'] if 'TEAM_ABBREVIATION' in latest else []) + ['GAME_DATE'] + features
    latest[cols].to_csv(out/'latest_player_features.csv',index=False)
    df.attrs['team_defense'].to_csv(out/'latest_team_defense.csv',index=False)
    print(f'Models, metrics and player feature snapshots saved to {out}/')

if __name__=='__main__': main()
