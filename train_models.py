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
POINT_SHOOTING_STATS = ['FGM','FG3M','FTM']
BASE_FEATURES = [f'{s}_L{n}' for s in STATS for n in (5,10)] + [f'{s}_SEASON_PRE' for s in STATS] + ['REST_DAYS','HOME','GAMES_PLAYED_BEFORE']
# Points-only features. Keep BASE_FEATURES untouched so rebounds and assists do not change.
POINT_EXTRA = [
    'PTS_PER_MIN_L5','PTS_PER_MIN_L10','PTS_PER_MIN_SEASON_PRE',
    'SHOT_ATTEMPTS_L5','SHOT_ATTEMPTS_L10',
    'USAGE_PROXY_L5','USAGE_PROXY_L10','USAGE_PROXY_SEASON_PRE',
    'TS_PROXY_L5','TS_PROXY_L10','TS_PROXY_SEASON_PRE',
    'PTS_STD_L5','PTS_STD_L10','PTS_TREND_5V10',
    'MIN_LAST','MIN_L3','MIN_STD_L5','MIN_STD_L10',
    'MIN_TREND_5V10','MIN_SEASON_PRE',
    'PTS_PER_MIN_L3','PTS_PER_MIN_STD_L5',
    'EFG_PCT_L5','EFG_PCT_L10','EFG_PCT_SEASON_PRE',
    'THREE_PA_RATE_L5','THREE_PA_RATE_L10','THREE_PA_RATE_SEASON_PRE',
    'FT_RATE_L5','FT_RATE_L10','FT_RATE_SEASON_PRE',
    'THREE_PCT_L5','THREE_PCT_L10','THREE_PCT_SEASON_PRE',
    'FT_PCT_L5','FT_PCT_L10','FT_PCT_SEASON_PRE',
    'POINTS_PER_FGA_L5','POINTS_PER_FGA_L10','POINTS_PER_FGA_SEASON_PRE',
    'OPP_ALLOWED_3PA_L5','OPP_ALLOWED_3PA_L10','OPP_ALLOWED_3PA_SEASON_PRE',
    'OPP_ALLOWED_3P_PCT_L5','OPP_ALLOWED_3P_PCT_L10','OPP_ALLOWED_3P_PCT_SEASON_PRE',
    'OPP_ALLOWED_2P_PCT_L5','OPP_ALLOWED_2P_PCT_L10','OPP_ALLOWED_2P_PCT_SEASON_PRE',
    'OPP_ALLOWED_EFG_L5','OPP_ALLOWED_EFG_L10','OPP_ALLOWED_EFG_SEASON_PRE',
    'OPP_ALLOWED_FT_RATE_L5','OPP_ALLOWED_FT_RATE_L10','OPP_ALLOWED_FT_RATE_SEASON_PRE',
]

def prepare(path):
    df=pd.read_csv(path)
    needed={'PLAYER_ID','PLAYER_NAME','GAME_DATE','MATCHUP','MIN','PTS','REB','AST'}
    missing=needed-set(df.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    if 'GAME_ID' not in df: df['GAME_ID']=np.arange(len(df)).astype(str)
    df['GAME_DATE']=pd.to_datetime(df['GAME_DATE'],errors='coerce')
    df=df.dropna(subset=['GAME_DATE','PLAYER_ID']).copy()
    for c in STATS + POINT_SHOOTING_STATS:
        if c not in df: df[c]=np.nan
        df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.sort_values(['PLAYER_ID','GAME_DATE','GAME_ID']).reset_index(drop=True)
    # Build pre-game opponent defensive shot-profile features from player game logs.
    # A team's box-score shooting totals in a game are the shots its opponent allowed.
    matchup = df['MATCHUP'].astype(str)
    df['TEAM_ABBREVIATION'] = df.get('TEAM_ABBREVIATION', matchup.str.split().str[0]).astype(str)
    df['OPPONENT_ABBREVIATION'] = np.where(
        matchup.str.contains(' vs. ', regex=False),
        matchup.str.split(' vs. ').str[-1],
        matchup.str.split(' @ ').str[-1]
    )
    df['HOME']=matchup.str.contains(' vs. ',regex=False).astype(int)
    df['REST_DAYS']=df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days.clip(0,14)
    df['GAMES_PLAYED_BEFORE']=df.groupby('PLAYER_ID').cumcount()
    for stat in STATS:
        g=df.groupby('PLAYER_ID',sort=False)[stat]
        for n in (5,10): df[f'{stat}_L{n}']=g.transform(lambda s:s.shift(1).rolling(n,min_periods=2).mean())
        df[f'{stat}_SEASON_PRE']=g.transform(lambda s:s.shift(1).expanding(min_periods=2).mean())
    # Three-pointers made history for the dedicated 3PM research model.
    g3m=df.groupby('PLAYER_ID',sort=False)['FG3M']
    for n in (5,10):
        df[f'FG3M_L{n}']=g3m.transform(lambda s:s.shift(1).rolling(n,min_periods=2).mean())
    df['FG3M_SEASON_PRE']=g3m.transform(lambda s:s.shift(1).expanding(min_periods=2).mean())
    df['MIN_LAST']=df.groupby('PLAYER_ID')['MIN'].shift(1)
    gmin=df.groupby('PLAYER_ID',sort=False)['MIN']
    df['MIN_L3']=gmin.transform(lambda s:s.shift(1).rolling(3,min_periods=2).mean())
    for n in (5,10):
        df[f'MIN_STD_L{n}']=gmin.transform(lambda s:s.shift(1).rolling(n,min_periods=3).std())
    df['MIN_TREND_5V10']=df['MIN_L5']-df['MIN_L10']
    df['PTS_PER_MIN']=df['PTS']/df['MIN'].replace(0,np.nan)
    df['SHOT_ATTEMPTS']=df['FGA']+0.44*df['FTA']
    df['USAGE_PROXY']=(df['FGA']+0.44*df['FTA']+df['TOV'])/df['MIN'].replace(0,np.nan)
    df['TS_PROXY']=df['PTS']/(2*(df['FGA']+0.44*df['FTA']).replace(0,np.nan))
    for stat in ['PTS_PER_MIN','SHOT_ATTEMPTS','USAGE_PROXY','TS_PROXY']:
        g=df.groupby('PLAYER_ID',sort=False)[stat]
        for n in (5,10): df[f'{stat}_L{n}']=g.transform(lambda s:s.shift(1).rolling(n,min_periods=2).mean())
        df[f'{stat}_SEASON_PRE']=g.transform(lambda s:s.shift(1).expanding(min_periods=2).mean())
    gppm=df.groupby('PLAYER_ID',sort=False)['PTS_PER_MIN']
    df['PTS_PER_MIN_L3']=gppm.transform(lambda s:s.shift(1).rolling(3,min_periods=2).mean())
    df['PTS_PER_MIN_STD_L5']=gppm.transform(lambda s:s.shift(1).rolling(5,min_periods=3).std())
    g=df.groupby('PLAYER_ID',sort=False)['PTS']
    for n in (5,10): df[f'PTS_STD_L{n}']=g.transform(lambda s:s.shift(1).rolling(n,min_periods=3).std())
    df['PTS_TREND_5V10']=df['PTS_L5']-df['PTS_L10']
    # Shooting-profile features are points-only. Every rolling/season feature is
    # shifted one game to avoid using the game being predicted.
    fga=df['FGA'].replace(0,np.nan)
    fg3a=df['FG3A'].replace(0,np.nan)
    fta=df['FTA'].replace(0,np.nan)
    df['EFG_PCT']=(df['FGM']+0.5*df['FG3M'])/fga
    df['THREE_PA_RATE']=df['FG3A']/fga
    df['FT_RATE']=df['FTA']/fga
    df['THREE_PCT']=df['FG3M']/fg3a
    df['FT_PCT']=df['FTM']/fta
    df['POINTS_PER_FGA']=df['PTS']/fga
    for stat in ['EFG_PCT','THREE_PA_RATE','FT_RATE','THREE_PCT','FT_PCT','POINTS_PER_FGA']:
        g=df.groupby('PLAYER_ID',sort=False)[stat]
        for n in (5,10):
            df[f'{stat}_L{n}']=g.transform(lambda s:s.shift(1).rolling(n,min_periods=2).mean())
        df[f'{stat}_SEASON_PRE']=g.transform(lambda s:s.shift(1).expanding(min_periods=2).mean())

    # Aggregate each offense by game, then attribute those shots to the opponent
    # as shots allowed. Rolling features are shifted, so today's game is excluded.
    team_game = df.groupby(['GAME_ID','GAME_DATE','TEAM_ABBREVIATION'], as_index=False).agg(
        TEAM_FGA=('FGA','sum'), TEAM_FGM=('FGM','sum'),
        TEAM_FG3A=('FG3A','sum'), TEAM_FG3M=('FG3M','sum'),
        TEAM_FTA=('FTA','sum'), TEAM_FTM=('FTM','sum')
    )
    matchup_map = df[['GAME_ID','TEAM_ABBREVIATION','OPPONENT_ABBREVIATION']].drop_duplicates()
    team_game = team_game.merge(matchup_map, on=['GAME_ID','TEAM_ABBREVIATION'], how='left')
    team_game['OPP_ALLOWED_3PA'] = team_game['TEAM_FG3A']
    team_game['OPP_ALLOWED_3P_PCT'] = team_game['TEAM_FG3M'] / team_game['TEAM_FG3A'].replace(0,np.nan)
    team_game['OPP_ALLOWED_2P_PCT'] = (
        (team_game['TEAM_FGM']-team_game['TEAM_FG3M']) /
        (team_game['TEAM_FGA']-team_game['TEAM_FG3A']).replace(0,np.nan)
    )
    team_game['OPP_ALLOWED_EFG'] = (
        team_game['TEAM_FGM']+0.5*team_game['TEAM_FG3M']
    ) / team_game['TEAM_FGA'].replace(0,np.nan)
    team_game['OPP_ALLOWED_FT_RATE'] = team_game['TEAM_FTA'] / team_game['TEAM_FGA'].replace(0,np.nan)
    defense = team_game.rename(columns={
        'OPPONENT_ABBREVIATION':'DEF_TEAM',
        'TEAM_ABBREVIATION':'OFF_TEAM'
    })
    defense = defense.dropna(subset=['DEF_TEAM']).sort_values(['DEF_TEAM','GAME_DATE','GAME_ID'])
    defense_features = ['OPP_ALLOWED_3PA','OPP_ALLOWED_3P_PCT','OPP_ALLOWED_2P_PCT','OPP_ALLOWED_EFG','OPP_ALLOWED_FT_RATE']
    for stat in defense_features:
        g=defense.groupby('DEF_TEAM',sort=False)[stat]
        for n in (5,10):
            defense[f'{stat}_L{n}']=g.transform(lambda v:v.shift(1).rolling(n,min_periods=3).mean())
        defense[f'{stat}_SEASON_PRE']=g.transform(lambda v:v.shift(1).expanding(min_periods=3).mean())
    keep=['GAME_ID','DEF_TEAM']+[f'{stat}_{period}' for stat in defense_features for period in ('L5','L10','SEASON_PRE')]
    defense=defense[keep].drop_duplicates(['GAME_ID','DEF_TEAM'])
    df=df.merge(defense,left_on=['GAME_ID','OPPONENT_ABBREVIATION'],right_on=['GAME_ID','DEF_TEAM'],how='left')
    df=df.drop(columns=['DEF_TEAM'])
    return df


def latest_team_defense_snapshot(df, feature_cols):
    """Return each opponent's latest pre-game defensive profile for upcoming fixtures."""
    defense_cols = [c for c in feature_cols if c.startswith("OPP_ALLOWED_")]
    required = {"OPPONENT_ABBREVIATION", "GAME_DATE", "GAME_ID"}
    if not required.issubset(df.columns) or not defense_cols:
        return pd.DataFrame(columns=["TEAM_ABBREVIATION"] + defense_cols)
    snapshot = (
        df.dropna(subset=["OPPONENT_ABBREVIATION"])
          .sort_values(["GAME_DATE", "GAME_ID"])
          .groupby("OPPONENT_ABBREVIATION", sort=False)
          .tail(1)
    )
    snapshot = snapshot[["OPPONENT_ABBREVIATION"] + defense_cols].copy()
    snapshot = snapshot.rename(columns={"OPPONENT_ABBREVIATION": "TEAM_ABBREVIATION"})
    return snapshot.drop_duplicates("TEAM_ABBREVIATION", keep="last").reset_index(drop=True)

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
    latest_team_defense_snapshot(df, point).to_csv(out/'latest_team_defense.csv',index=False)
    print(f'Models, metrics and feature snapshots saved to {out}/')

if __name__=='__main__': main()
