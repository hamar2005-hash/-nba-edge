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
from rate_minutes_predictor import RateMinutesPredictor
from team_defense_snapshot import save_latest_team_defense_snapshot

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
    """Tune the points blend on an inner chronological validation split, then score untouched holdout."""
    train = train[train.GAMES_PLAYED_BEFORE >= 2].copy()
    test = test[test.GAMES_PLAYED_BEFORE >= 2].copy()
    train = train[(train.MIN > 0) & train.PTS.notna()].copy()
    test = test[(test.MIN > 0) & test.PTS.notna()].copy()
    if len(train) < 2000 or len(test) < 100:
        raise RuntimeError(f'Insufficient points-v2 rows: train={len(train)}, holdout={len(test)}')

    train_dates = np.sort(train.GAME_DATE.dropna().unique())
    inner_cutoff = train_dates[max(1, int(len(train_dates) * 0.82))]
    inner_fit = train[train.GAME_DATE < inner_cutoff]
    inner_val = train[train.GAME_DATE >= inner_cutoff]
    if len(inner_fit) < 1000 or len(inner_val) < 200:
        raise RuntimeError(f'Insufficient inner validation rows: fit={len(inner_fit)}, validation={len(inner_val)}')

    def fit_components(fit):
        minutes = Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('regressor', ExtraTreesRegressor(
                n_estimators=350, min_samples_leaf=5, max_features=0.85,
                n_jobs=-1, random_state=43
            ))
        ])
        rate = make_regressor(seed=44)
        direct = make_regressor(seed=45)
        rate_target = fit['PTS'] / fit['MIN'].clip(lower=1)
        minutes.fit(fit[features], fit['MIN'])
        rate.fit(fit[features], rate_target)
        direct.fit(fit[features], fit['PTS'])
        return minutes, rate, direct

    # Select blend weight only on data earlier than the final holdout.
    val_minutes, val_rate, val_direct = fit_components(inner_fit)
    val_min = np.clip(val_minutes.predict(inner_val[features]), 0, 48)
    val_ppm = np.clip(val_rate.predict(inner_val[features]), 0, 1.5)
    val_rate_points = np.maximum(0, val_min * val_ppm)
    val_direct_points = np.maximum(0, val_direct.predict(inner_val[features]))
    weights = np.round(np.arange(0.0, 0.751, 0.05), 2)
    validation_maes = {
        str(float(w)): float(mean_absolute_error(
            inner_val['PTS'], (1.0 - w) * val_rate_points + w * val_direct_points
        )) for w in weights
    }
    # Prefer the lower direct-model weight when validation MAE is tied.
    best_weight = min(weights, key=lambda w: (validation_maes[str(float(w))], w))

    # Refit components on all permitted training data, keeping the final holdout untouched.
    minutes_model, rate_model, direct_model = fit_components(train)
    predictor = PointsV2Predictor(
        minutes_model, rate_model, features,
        direct_model=direct_model, direct_weight=float(best_weight)
    )
    details = predictor.predict_with_details(test[features])
    pred = details['points']
    direct_pred = details['direct_points']
    rate_pred = details['rate_points']
    residual = test['PTS'].to_numpy() - pred
    minute_residual = test['MIN'].to_numpy() - details['expected_minutes']
    baseline_minutes = test['MIN_L5'].fillna(test['MIN_SEASON_PRE']).fillna(train['MIN'].median()).to_numpy()
    baseline_points = test['PTS_L5'].fillna(test['PTS_SEASON_PRE']).fillna(train['PTS'].median()).to_numpy()
    rate_actual = test['PTS'].to_numpy() / test['MIN'].clip(lower=1).to_numpy()
    score = {
        'train_rows': int(len(train)), 'test_rows': int(len(test)),
        'inner_validation_rows': int(len(inner_val)),
        'inner_validation_start': str(pd.Timestamp(inner_cutoff).date()),
        'selected_direct_weight': float(best_weight),
        'inner_validation_mae_by_weight': validation_maes,
        'mae': float(mean_absolute_error(test['PTS'], pred)),
        'rmse': float(np.sqrt(mean_squared_error(test['PTS'], pred))),
        'direct_points_mae': float(mean_absolute_error(test['PTS'], direct_pred)),
        'rate_product_mae': float(mean_absolute_error(test['PTS'], rate_pred)),
        'minutes_mae': float(mean_absolute_error(test['MIN'], details['expected_minutes'])),
        'minutes_rmse': float(np.sqrt(mean_squared_error(test['MIN'], details['expected_minutes']))),
        'minutes_l5_baseline_mae': float(mean_absolute_error(test['MIN'], baseline_minutes)),
        'points_l5_baseline_mae': float(mean_absolute_error(test['PTS'], baseline_points)),
        'rate_mae': float(mean_absolute_error(rate_actual, details['points_per_minute'])),
    }
    pack = {
        'model': predictor, 'feature_cols': features, 'target': 'PTS', 'market': 'points',
        'model_version': 'points_v2_shooting_features_tuned_blend',
        'residual_std': float(np.std(residual, ddof=1)),
        'minutes_residual_std': float(np.std(minute_residual, ddof=1)),
        'points_per_minute_residual_std': float(np.std(
            rate_actual - details['points_per_minute'], ddof=1
        )),
    }
    return pack, score, test


def fit_rate_minutes_candidate(train, test, features, market, target):
    """Train minutes/rate/direct components; tune blend on inner dates only."""
    train = train[train.GAMES_PLAYED_BEFORE >= 2].copy()
    test = test[test.GAMES_PLAYED_BEFORE >= 2].copy()
    train = train[(train.MIN > 0) & train[target].notna()].copy()
    test = test[(test.MIN > 0) & test[target].notna()].copy()
    if len(train) < 2000 or len(test) < 100:
        raise RuntimeError(f'Insufficient {market} rate-model rows: train={len(train)}, holdout={len(test)}')
    dates = np.sort(train.GAME_DATE.dropna().unique())
    cutoff = dates[max(1, int(len(dates) * 0.82))]
    inner_fit = train[train.GAME_DATE < cutoff]
    inner_val = train[train.GAME_DATE >= cutoff]
    if len(inner_fit) < 1000 or len(inner_val) < 200:
        raise RuntimeError(f'Insufficient {market} inner validation rows: fit={len(inner_fit)}, validation={len(inner_val)}')

    rate_cap = 0.85 if market == 'rebounds' else 0.65

    def fit_components(frame):
        minutes = Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('regressor', ExtraTreesRegressor(
                n_estimators=350, min_samples_leaf=5, max_features=0.85,
                n_jobs=-1, random_state=53
            ))
        ])
        rate = make_regressor(seed=54)
        direct = make_regressor(seed=55)
        minutes.fit(frame[features], frame['MIN'])
        rate.fit(frame[features], frame[target] / frame['MIN'].clip(lower=1))
        direct.fit(frame[features], frame[target])
        return minutes, rate, direct

    val_minutes, val_rate, val_direct = fit_components(inner_fit)
    val_min = np.clip(val_minutes.predict(inner_val[features]), 0, 48)
    val_rate_per_min = np.clip(val_rate.predict(inner_val[features]), 0, rate_cap)
    val_rate_projection = np.maximum(0, val_min * val_rate_per_min)
    val_direct_projection = np.maximum(0, val_direct.predict(inner_val[features]))
    weights = np.round(np.arange(0.0, 0.751, 0.05), 2)
    validation_maes = {
        str(float(w)): float(mean_absolute_error(
            inner_val[target],
            (1.0 - w) * val_rate_projection + w * val_direct_projection
        )) for w in weights
    }
    best_weight = min(weights, key=lambda w: (validation_maes[str(float(w))], w))

    minutes_model, rate_model, direct_model = fit_components(train)
    predictor = RateMinutesPredictor(
        minutes_model, rate_model, features, market,
        direct_model=direct_model, direct_weight=float(best_weight)
    )
    details = predictor.predict_with_details(test[features])
    pred = details['projection']
    residual = test[target].to_numpy() - pred
    minute_residual = test['MIN'].to_numpy() - details['expected_minutes']
    baseline = test[f'{target}_L5'].fillna(
        test[f'{target}_SEASON_PRE']
    ).fillna(train[target].median()).to_numpy()
    rate_actual = test[target].to_numpy() / test['MIN'].clip(lower=1).to_numpy()
    score = {
        'train_rows': int(len(train)), 'test_rows': int(len(test)),
        'inner_validation_rows': int(len(inner_val)),
        'inner_validation_start': str(pd.Timestamp(cutoff).date()),
        'selected_direct_weight': float(best_weight),
        'inner_validation_mae_by_weight': validation_maes,
        'mae': float(mean_absolute_error(test[target], pred)),
        'rmse': float(np.sqrt(mean_squared_error(test[target], pred))),
        'direct_mae': float(mean_absolute_error(test[target], details['direct_projection'])),
        'rate_product_mae': float(mean_absolute_error(test[target], details['rate_projection'])),
        'minutes_mae': float(mean_absolute_error(test['MIN'], details['expected_minutes'])),
        'minutes_baseline_mae': float(mean_absolute_error(test['MIN'], test['MIN_L5'].fillna(test['MIN_SEASON_PRE']).fillna(train['MIN'].median()))),
        'last5_baseline_mae': float(mean_absolute_error(test[target], baseline)),
        'rate_mae': float(mean_absolute_error(rate_actual, details['per_minute_rate'])),
    }
    pack = {
        'model': predictor, 'feature_cols': features, 'target': target,
        'market': market, 'model_version': f'{market}_minutes_rate_blend_v1',
        'residual_std': float(np.std(residual, ddof=1)),
        'minutes_residual_std': float(np.std(minute_residual, ddof=1)),
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

            # Controlled ablation: same chronological split, same training rows,
            # same inner-validation blend tuning and same untouched holdout.
            defense_features=[c for c in features if c.startswith('OPP_ALLOWED_')]
            no_defense_features=[c for c in features if c not in defense_features]
            _,ablation_score,_=fit_points_v2(train,test,no_defense_features)
            defense_ablation={
                'full_feature_mae':float(score['mae']),
                'without_defense_features_mae':float(ablation_score['mae']),
                'full_feature_rmse':float(score['rmse']),
                'without_defense_features_rmse':float(ablation_score['rmse']),
                'mae_change_from_defense_features':float(ablation_score['mae']-score['mae']),
                'rmse_change_from_defense_features':float(ablation_score['rmse']-score['rmse']),
                'defense_feature_count':len(defense_features),
                'same_holdout_rows':int(score['test_rows'])==int(ablation_score['test_rows']),
                'interpretation':'Positive MAE/RMSE change means removing defensive features made error worse; negative means removing them improved error.'
            }
            incumbent_mae=None
            incumbent_rmse=None
            if incumbent_path.exists():
                incumbent=joblib.load(incumbent_path)
                old_features=incumbent.get('feature_cols',features)
                common=holdout.dropna(subset=[target])
                common=common[common.GAMES_PLAYED_BEFORE>=2]
                try:
                    incumbent_pred=incumbent['model'].predict(common[old_features])
                    incumbent_mae=float(mean_absolute_error(common[target],incumbent_pred))
                    incumbent_rmse=float(np.sqrt(mean_squared_error(common[target],incumbent_pred)))
                except Exception as exc:
                    print(f'points: incumbent comparison failed: {exc}')
            accept=(not incumbent_path.exists()) or (
                incumbent_mae is not None
                and score['mae'] <= incumbent_mae * 0.99
                and (incumbent_rmse is None or score['rmse'] <= incumbent_rmse)
            )
            if accept:
                joblib.dump(candidate,incumbent_path)
                accepted_any=True
            report['markets'][market]={
                'incumbent_holdout_mae':incumbent_mae,
                'incumbent_holdout_rmse':incumbent_rmse,
                'candidate_holdout_mae':score['mae'],
                'candidate_holdout_rmse':score['rmse'],
                'holdout_rows':int(score['test_rows']),
                'same_holdout_for_both_models':bool(incumbent_mae is not None),
                'mae_improvement_points':(float(incumbent_mae-score['mae']) if incumbent_mae is not None else None),
                'mae_improvement_pct':(float((incumbent_mae-score['mae'])/incumbent_mae*100) if incumbent_mae not in (None,0) else None),
                'rmse_improvement_points':(float(incumbent_rmse-score['rmse']) if incumbent_rmse is not None else None),
                'rmse_improvement_pct':(float((incumbent_rmse-score['rmse'])/incumbent_rmse*100) if incumbent_rmse not in (None,0) else None),
                'candidate_minutes_mae':score['minutes_mae'],
                'candidate_minutes_rmse':score['minutes_rmse'],
                'candidate_direct_points_mae':score['direct_points_mae'],
                'candidate_rate_product_mae':score['rate_product_mae'],
                'candidate_minutes_l5_baseline_mae':score['minutes_l5_baseline_mae'],
                'candidate_points_l5_baseline_mae':score['points_l5_baseline_mae'],
                'candidate_rate_mae':score['rate_mae'],
                'defense_feature_ablation':defense_ablation,
                'ablation_without_defense_selected_direct_weight':float(ablation_score['selected_direct_weight']),
                'selected_direct_weight':score['selected_direct_weight'],
                'inner_validation_start':score['inner_validation_start'],
                'inner_validation_mae_by_weight':score['inner_validation_mae_by_weight'],
                'promotion_threshold_mae_improvement_pct':1.0,
                'candidate_accepted':bool(accept),
                'train_rows':score['train_rows'],
                'holdout_rows':score['test_rows'],
                'model_version':candidate['model_version'],
                'confidence_tier_note':'Heuristic tiers use holdout point/minute residual spread and forecast-vs-recent-minutes deviation; not calibrated probabilities.'
            }
            print(f"points-v2: incumbent MAE={incumbent_mae}; candidate MAE={score['mae']:.4f}; tuned direct weight={score['selected_direct_weight']:.2f}; direct MAE={score['direct_points_mae']:.4f}; minutes ML/baseline MAE={score['minutes_mae']:.3f}/{score['minutes_l5_baseline_mae']:.3f}; accepted={accept}")
            continue

        features=base
        if market in ('rebounds', 'assists'):
            candidate,score,holdout=fit_rate_minutes_candidate(train,test,features,market,target)
            incumbent_mae=None
            incumbent_rmse=None
            if incumbent_path.exists():
                incumbent=joblib.load(incumbent_path)
                old_features=incumbent.get('feature_cols',features)
                common=holdout.dropna(subset=[target])
                common=common[common.GAMES_PLAYED_BEFORE>=2]
                try:
                    incumbent_pred=incumbent['model'].predict(common[old_features])
                    incumbent_mae=float(mean_absolute_error(common[target],incumbent_pred))
                    incumbent_rmse=float(np.sqrt(mean_squared_error(common[target],incumbent_pred)))
                except Exception as exc:
                    print(f'{market}: incumbent comparison failed: {exc}')
            accept=(not incumbent_path.exists()) or (
                incumbent_mae is not None
                and score['mae'] <= incumbent_mae * 0.99
                and (incumbent_rmse is None or score['rmse'] <= incumbent_rmse)
            )
            if accept:
                joblib.dump(candidate,incumbent_path)
                accepted_any=True
            report['markets'][market]={
                'incumbent_holdout_mae':incumbent_mae,
                'incumbent_holdout_rmse':incumbent_rmse,
                'candidate_holdout_mae':score['mae'],
                'candidate_holdout_rmse':score['rmse'],
                'candidate_direct_mae':score['direct_mae'],
                'candidate_rate_product_mae':score['rate_product_mae'],
                'candidate_minutes_mae':score['minutes_mae'],
                'candidate_minutes_baseline_mae':score['minutes_baseline_mae'],
                'candidate_last5_baseline_mae':score['last5_baseline_mae'],
                'candidate_rate_mae':score['rate_mae'],
                'selected_direct_weight':score['selected_direct_weight'],
                'inner_validation_start':score['inner_validation_start'],
                'inner_validation_mae_by_weight':score['inner_validation_mae_by_weight'],
                'promotion_threshold_mae_improvement_pct':1.0,
                'candidate_accepted':bool(accept),
                'train_rows':score['train_rows'],
                'holdout_rows':score['test_rows'],
                'model_version':candidate['model_version'],
            }
            print(f"{market}-minutes-rate: incumbent MAE={incumbent_mae}; candidate MAE={score['mae']:.4f}; minutes MAE/baseline={score['minutes_mae']:.3f}/{score['minutes_baseline_mae']:.3f}; accepted={accept}")
            continue

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
    defense_snapshot = save_latest_team_defense_snapshot(df, MODELS/'latest_team_defense.csv')
    print(f'Updated opponent-defense snapshot for {len(defense_snapshot)} teams.')
    report['any_model_updated']=accepted_any
    (MODELS/'weekly_retrain_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
