"""Weekly candidate retraining; promote a model only when holdout MAE improves."""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from train_models import prepare, TARGETS, BASE_FEATURES, POINT_EXTRA

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; MODELS=ROOT/'models'

def fit_candidate(train,test,features,market,target):
    tr=train.dropna(subset=[target]); te=test.dropna(subset=[target])
    tr=tr[tr.GAMES_PLAYED_BEFORE>=2]; te=te[te.GAMES_PLAYED_BEFORE>=2]
    if len(tr)<1000 or len(te)<100: raise RuntimeError(f'Insufficient {market} rows: train={len(tr)}, holdout={len(te)}')
    if market=='points':
        estimator=VotingRegressor([
            ('hgb',HistGradientBoostingRegressor(max_iter=300,learning_rate=.045,max_leaf_nodes=15,l2_regularization=2.5,random_state=42)),
            ('extra',ExtraTreesRegressor(n_estimators=350,min_samples_leaf=4,max_features=.85,n_jobs=-1,random_state=42))
        ],weights=[.55,.45])
    else:
        estimator=HistGradientBoostingRegressor(max_iter=250,learning_rate=.055,max_leaf_nodes=15,l2_regularization=2.0,random_state=42)
    model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('regressor',estimator)])
    model.fit(tr[features],tr[target]); pred=model.predict(te[features]); residual=te[target].to_numpy()-pred
    pack={'model':model,'feature_cols':features,'target':target,'market':market,'residual_std':float(np.std(residual,ddof=1)),'model_version':'points_v2' if market=='points' else 'baseline'}
    score={'train_rows':len(tr),'test_rows':len(te),'mae':float(mean_absolute_error(te[target],pred)),'rmse':float(np.sqrt(mean_squared_error(te[target],pred)))}
    return pack,score,te

def main():
    source=DATA/'player_game_logs.csv'
    if not source.exists(): raise SystemExit('Historical player game logs are missing.')
    df=prepare(source); dates=np.sort(df.GAME_DATE.dropna().unique())
    if len(dates)<30: raise SystemExit('Not enough distinct dates for safe weekly retraining.')
    cutoff=dates[max(1,int(len(dates)*.8))]
    train=df[df.GAME_DATE<cutoff]; test=df[df.GAME_DATE>=cutoff]
    base=[c for c in BASE_FEATURES if c in df.columns]
    points=list(dict.fromkeys(base+[c for c in POINT_EXTRA if c in df.columns]))
    MODELS.mkdir(exist_ok=True)
    report={'trained_at_utc':pd.Timestamp.utcnow().isoformat(),'holdout_start':str(pd.Timestamp(cutoff).date()),'markets':{}}
    accepted_any=False
    for market,target in TARGETS.items():
        features=points if market=='points' else base
        candidate,score,holdout=fit_candidate(train,test,features,market,target)
        incumbent_path=MODELS/f'{market}_model.joblib'; incumbent_mae=None
        if incumbent_path.exists():
            incumbent=joblib.load(incumbent_path)
            old_features=incumbent.get('feature_cols',features)
            common=holdout.dropna(subset=[target]); common=common[common.GAMES_PLAYED_BEFORE>=2]
            try:
                incumbent_pred=incumbent['model'].predict(common[old_features])
                incumbent_mae=float(mean_absolute_error(common[target],incumbent_pred))
            except Exception as exc:
                print(f'{market}: incumbent comparison failed: {exc}')
        accept=(not incumbent_path.exists()) or (incumbent_mae is not None and score['mae']<incumbent_mae)
        if accept:
            joblib.dump(candidate,incumbent_path); accepted_any=True
        report['markets'][market]={'incumbent_holdout_mae':incumbent_mae,'candidate_holdout_mae':score['mae'],'candidate_holdout_rmse':score['rmse'],'candidate_accepted':bool(accept),'train_rows':score['train_rows'],'holdout_rows':score['test_rows'],'model_version':candidate['model_version']}
        print(f"{market}: incumbent MAE={incumbent_mae}; candidate MAE={score['mae']:.4f}; accepted={accept}")
    latest=df.sort_values('GAME_DATE').groupby('PLAYER_ID',as_index=False).tail(1)
    identity=['PLAYER_ID','PLAYER_NAME']+(['TEAM_ABBREVIATION'] if 'TEAM_ABBREVIATION' in latest else [])+['GAME_DATE']
    latest[list(dict.fromkeys(identity+base+points))].to_csv(MODELS/'latest_player_features.csv',index=False)
    pd.DataFrame(columns=['TEAM_ABBREVIATION']).to_csv(MODELS/'latest_team_defense.csv',index=False)
    report['any_model_updated']=accepted_any
    (MODELS/'weekly_retrain_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
