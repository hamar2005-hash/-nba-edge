"""Train NBA Edge player prop models with an enhanced points model (V2)."""
from pathlib import Path
import argparse, json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

TARGETS = {'points':'PTS', 'rebounds':'REB', 'assists':'AST'}
STATS = ['PTS','REB','AST','MIN','FGA','FG3A','FTA','TOV','STL','BLK']
BASE_FEATURES = [f'{s}_L{n}' for s in STATS for n in (5,10)] + [f'{s}_SEASON_PRE' for s in STATS] + ['REST_DAYS','HOME','GAMES_PLAYED_BEFORE']
POINT_EXTRA = ['PTS_PER_MIN_L5','PTS_PER_MIN_L10','SHOT_ATTEMPTS_L5','SHOT_ATTEMPTS_L10','USAGE_PROXY_L5','USAGE_PROXY_L10','TS_PROXY_L5','TS_PROXY_L10','PTS_STD_L5','PTS_STD_L10','PTS_TREND_5V10']

def prepare(path):
    df=pd.read_csv(path)
    needed={'PLAYER_ID','PLAYER_NAME','GAME_DATE','MATCHUP','MIN','PTS','REB','AST'}
    missing=needed-set(df.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    if 'GAME_ID' not in df: df['GAME_ID']=np.arange(len(df)).astype(str)
    df['GAME_DATE']=pd.to_datetime(df['GAME_DATE'],errors='coerce')
    df=df.dropna(subset=['GAME_DATE','PLAYER_ID']).copy()
    for c in STATS:
        if c not in df: df[c]=np.nan
        df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.sort_values(['PLAYER_ID','GAME_DATE','GAME_ID']).reset_index(drop=True)
    df['HOME']=df['MATCHUP'].astype(str).str.contains(' vs. ',regex=False).astype(int)
    df['REST_DAYS']=df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days.clip(0,14)
    df['GAMES_PLAYED_BEFORE']=df.groupby('PLAYER_ID').cumcount()
    for stat in STATS:
        g=df.groupby('PLAYER_ID',sort=False)[stat]
        for n in (5,10): df[f'{stat}_L{n}']=g.transform(lambda s:s.shift(1).rolling(n,min_periods=2).mean())
        df[f'{stat}_SEASON_PRE']=g.transform(lambda s:s.shift(1).expanding(min_periods=2).mean())
    df['PTS_PER_MIN']=df['PTS']/df['MIN'].replace(0,np.nan)
    df['SHOT_ATTEMPTS']=df['FGA']+0.44*df['FTA']
    df['USAGE_PROXY']=(df['FGA']+0.44*df['FTA']+df['TOV'])/df['MIN'].replace(0,np.nan)
    df['TS_PROXY']=df['PTS']/(2*(df['FGA']+0.44*df['FTA']).replace(0,np.nan))
    for stat in ['PTS_PER_MIN','SHOT_ATTEMPTS','USAGE_PROXY','TS_PROXY']:
        g=df.groupby('PLAYER_ID',sort=False)[stat]
        for n in (5,10): df[f'{stat}_L{n}']=g.transform(lambda s:s.shift(1).rolling(n,min_periods=2).mean())
    g=df.groupby('PLAYER_ID',sort=False)['PTS']
    for n in (5,10): df[f'PTS_STD_L{n}']=g.transform(lambda s:s.shift(1).rolling(n,min_periods=3).std())
    df['PTS_TREND_5V10']=df['PTS_L5']-df['PTS_L10']
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='data/player_game_logs.csv')
    ap.add_argument('--outdir',default='models')
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    df=prepare(args.input)
    base=[c for c in BASE_FEATURES if c in df.columns]
    point=list(dict.fromkeys(base+[c for c in POINT_EXTRA if c in df.columns]))
    dates=np.sort(df.GAME_DATE.dropna().unique())
    if len(dates)<20: raise SystemExit('Not enough distinct game dates for chronological holdout.')
    cutoff=dates[max(1,int(len(dates)*.8))]
    train=df[df.GAME_DATE<cutoff]; test=df[df.GAME_DATE>=cutoff]
    metrics={}
    for market,target in TARGETS.items():
        features=point if market=='points' else base
        tr=train.dropna(subset=[target]); te=test.dropna(subset=[target])
        tr=tr[tr.GAMES_PLAYED_BEFORE>=2]; te=te[te.GAMES_PLAYED_BEFORE>=2]
        if len(tr)<1000 or len(te)<100: raise SystemExit(f'Insufficient rows for {market}: train={len(tr)}, test={len(te)}')
        if market=='points':
            estimator=VotingRegressor([
                ('hgb',HistGradientBoostingRegressor(max_iter=300,learning_rate=.045,max_leaf_nodes=15,l2_regularization=2.5,random_state=42)),
                ('extra',ExtraTreesRegressor(n_estimators=350,min_samples_leaf=4,max_features=.85,n_jobs=-1,random_state=42))
            ],weights=[.55,.45])
        else:
            estimator=HistGradientBoostingRegressor(max_iter=250,learning_rate=.055,max_leaf_nodes=15,l2_regularization=2.0,random_state=42)
        model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('regressor',estimator)])
        model.fit(tr[features],tr[target]); pred=model.predict(te[features]); residual=te[target].to_numpy()-pred
        metrics[market]={'train_rows':int(len(tr)),'test_rows':int(len(te)),'holdout_start':str(pd.Timestamp(cutoff).date()),'mae':float(mean_absolute_error(te[target],pred)),'rmse':float(np.sqrt(mean_squared_error(te[target],pred))),'residual_std':float(np.std(residual,ddof=1)),'feature_count':len(features),'model_version':'points_v2' if market=='points' else 'baseline'}
        joblib.dump({'model':model,'feature_cols':features,'residual_std':metrics[market]['residual_std'],'target':target,'market':market,'model_version':metrics[market]['model_version']},out/f'{market}_model.joblib')
        print(f"{market}: MAE {metrics[market]['mae']:.3f} | RMSE {metrics[market]['rmse']:.3f} | features {len(features)}")
    (out/'holdout_metrics.json').write_text(json.dumps(metrics,indent=2))
    latest=df.sort_values('GAME_DATE').groupby('PLAYER_ID',as_index=False).tail(1)
    identity=['PLAYER_ID','PLAYER_NAME']+(['TEAM_ABBREVIATION'] if 'TEAM_ABBREVIATION' in latest else [])+['GAME_DATE']
    latest[list(dict.fromkeys(identity+base+point))].to_csv(out/'latest_player_features.csv',index=False)
    pd.DataFrame(columns=['TEAM_ABBREVIATION']).to_csv(out/'latest_team_defense.csv',index=False)
    print(f'Models, metrics and feature snapshots saved to {out}/')

if __name__=='__main__': main()
