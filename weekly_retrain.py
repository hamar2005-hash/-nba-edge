"""Weekly candidate retraining; promote only models that beat incumbents on a chronological holdout."""
from pathlib import Path
import argparse, json, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from train_models import prepare, TARGETS, BASE_FEATURES, POINT_EXTRA
from points_v2 import PointsV2Predictor

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; MODELS=ROOT/'models'


def make_regressor(seed=42):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('regressor', HistGradientBoostingRegressor(
            max_iter=300, learning_rate=.045, max_leaf_nodes=15,
            l2_regularization=2.5, random_state=seed
        ))
    ])


def fit_candidate(train, test, features, market, target):
    tr=train.dropna(subset=[target]); te=test.dropna(subset=[target])
    tr=tr[tr.GAMES_PLAYED_BEFORE>=2]; te=te[te.GAMES_PLAYED_BEFORE>=2]
    if len(tr)<1000 or len(te)<100:
        raise RuntimeError(f'Insufficient {market} rows: train={len(tr)}, holdout={len(te)}')
    model=make_regressor()
    model.fit(tr[features],tr[target])
    pred=model.predict(te[features])
    residual=te[target].to_numpy()-pred
    pack={'model':model,'feature_cols':features,'target':target,'market':market,
          'residual_std':float(np.std(residual,ddof=1)),'model_version':'baseline'}
    score={'train_rows':len(tr),'test_rows':len(te),
           'mae':float(mean_absolute_error(te[target],pred)),
           'rmse':float(np.sqrt(mean_squared_error(te[target],pred)))}
    return pack,score,te


def fit_points_v2(train, test, features):
    """Fit minutes and points-per-minute models, then combine them on holdout."""
    train = train[train.GAMES_PLAYED_BEFORE >= 2].copy()
    test = test[test.GAMES_PLAYED_BEFORE >= 2].copy()
    train = train[(train.MIN > 0) & train.PTS.notna()].copy()
    test = test[(test.MIN > 0) & test.PTS.notna()].copy()
    if len(train) < 1000 or len(test) < 100:
        raise RuntimeError(f'Insufficient points-v2 rows: train={len(train)}, holdout={len(test)}')

    minutes_model = make_regressor(seed=43)
    rate_model = make_regressor(seed=44)
    rate_train = train['PTS'] / train['MIN'].clip(lower=1)
    minutes_model.fit(train[features], train['MIN'])
    rate_model.fit(train[features], rate_train)

    predictor = PointsV2Predictor(minutes_model, rate_model, features)
    details = predictor.predict_with_details(test[features])
    pred = details['points']
    residual = test['PTS'].to_numpy() - pred
    minute_residual = test['MIN'].to_numpy() - details['expected_minutes']

    pack = {
        'model': predictor,
        'feature_cols': features,
        'target': 'PTS',
        'market': 'points',
        'model_version': 'points_v2_minutes_rate',
        'residual_std': float(np.std(residual, ddof=1)),
        'minutes_residual_std': float(np.std(minute_residual, ddof=1)),
        'points_per_minute_residual_std': float(np.std(
            (test['PTS'] / test['MIN'].clip(lower=1)).to_numpy() - details['points_per_minute'],
            ddof=1
        )),
    }
    score = {
        'train_rows': int(len(train)),
        'test_rows': int(len(test)),
        'mae': float(mean_absolute_error(test['PTS'], pred)),
        'rmse': float(np.sqrt(mean_squared_error(test['PTS'], pred))),
        'minutes_mae': float(mean_absolute_error(test['MIN'], details['expected_minutes'])),
        'minutes_rmse': float(np.sqrt(mean_squared_error(test['MIN'], details['expected_minutes']))),
    }
    return pack, score, test


def main():
    parser=argparse.ArgumentParser(description='Retrain and validate NBA Edge models.')
    parser.add_argument('--market', choices=['all','points','rebounds','assists'], default='all')
    args=parser.parse_args()
    source=DATA/'player_game_logs.csv'
    if not source.exists(): raise SystemExit('Historical player game logs are missing.')
    df=prepare(source); dates=np.sort(df.GAME_DATE.dropna().unique())
    if len(dates)<30: raise SystemExit('Not enough distinct dates for safe weekly retraining.')
    cutoff=dates[max(1,int(len(dates)*.8))]
    train=df[df.GAME_DATE<cutoff]; test=df[df.GAME_DATE>=cutoff]
    base=[c for c in BASE_FEATURES if c in df.columns]
    points_features=list(dict.fromkeys(base+[c for c in POINT_EXTRA if c in df.columns]))
    MODELS.mkdir(exist_ok=True)
    report={'trained_at_utc':pd.Timestamp.utcnow().isoformat(),
            'holdout_start':str(pd.Timestamp(cutoff).date()),'markets':{}}
    accepted_any=False
    selected_targets = TARGETS if args.market == 'all' else {args.market: TARGETS[args.market]}

    for market,target in selected_targets.items():
        incumbent_path=MODELS/f'{market}_model.joblib'
        if market == 'points':
            features=points_features
            candidate,score,holdout=fit_points_v2(train,test,features)
            incumbent_mae=None
            if incumbent_path.exists():
                incumbent=joblib.load(incumbent_path)
                old_features=incumbent.get('feature_cols',features)
                common=holdout.dropna(subset=[target])
                common=common[common.GAMES_PLAYED_BEFORE>=2]
                try:
                    incumbent_pred=incumbent['model'].predict(common[old_features])
                    incumbent_mae=float(mean_absolute_error(common[target],incumbent_pred))
                except Exception as exc:
                    print(f'points: incumbent comparison failed: {exc}')
            accept=(not incumbent_path.exists()) or (
                incumbent_mae is not None and score['mae'] < incumbent_mae
            )
            if accept:
                joblib.dump(candidate,incumbent_path)
                accepted_any=True
            report['markets'][market]={
                'incumbent_holdout_mae':incumbent_mae,
                'candidate_holdout_mae':score['mae'],
                'candidate_holdout_rmse':score['rmse'],
                'candidate_minutes_mae':score['minutes_mae'],
                'candidate_minutes_rmse':score['minutes_rmse'],
                'candidate_accepted':bool(accept),
                'train_rows':score['train_rows'],
                'holdout_rows':score['test_rows'],
                'model_version':candidate['model_version'],
                'confidence_tier_note':'Heuristic tiers use holdout point/minute residual spread and forecast-vs-recent-minutes deviation; not calibrated probabilities.'
            }
            print(f"points-v2: incumbent MAE={incumbent_mae}; candidate MAE={score['mae']:.4f}; minutes MAE={score['minutes_mae']:.3f}; accepted={accept}")
            continue

        features=base
        candidate,score,holdout=fit_candidate(train,test,features,market,target)
        incumbent_mae=None
        if incumbent_path.exists():
            incumbent=joblib.load(incumbent_path)
            old_features=incumbent.get('feature_cols',features)
            common=holdout.dropna(subset=[target])
            common=common[common.GAMES_PLAYED_BEFORE>=2]
            try:
                incumbent_pred=incumbent['model'].predict(common[old_features])
                incumbent_mae=float(mean_absolute_error(common[target],incumbent_pred))
            except Exception as exc:
                print(f'{market}: incumbent comparison failed: {exc}')
        accept=(not incumbent_path.exists()) or (
            incumbent_mae is not None and score['mae'] < incumbent_mae
        )
        if accept:
            joblib.dump(candidate,incumbent_path)
            accepted_any=True
        report['markets'][market]={
            'incumbent_holdout_mae':incumbent_mae,
            'candidate_holdout_mae':score['mae'],
            'candidate_holdout_rmse':score['rmse'],
            'candidate_accepted':bool(accept),
            'train_rows':score['train_rows'],
            'holdout_rows':score['test_rows'],
            'model_version':candidate['model_version']
        }
        print(f"{market}: incumbent MAE={incumbent_mae}; candidate MAE={score['mae']:.4f}; accepted={accept}")

    latest=df.sort_values('GAME_DATE').groupby('PLAYER_ID',as_index=False).tail(1)
    identity=['PLAYER_ID','PLAYER_NAME']+(['TEAM_ABBREVIATION'] if 'TEAM_ABBREVIATION' in latest else [])+['GAME_DATE']
    latest[list(dict.fromkeys(identity+base+points_features))].to_csv(MODELS/'latest_player_features.csv',index=False)
    pd.DataFrame(columns=['TEAM_ABBREVIATION']).to_csv(MODELS/'latest_team_defense.csv',index=False)
    report['any_model_updated']=accepted_any
    (MODELS/'weekly_retrain_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
