"""Generate PTS/REB/AST prop probabilities and optional bookmaker value estimates.

Input CSV columns: game_date,player,team,opponent,home,minutes,market,line
Optional: bookmaker,odds,side (over/under). Odds must be decimal.
"""
from pathlib import Path
import argparse, math
import joblib
import numpy as np
import pandas as pd

MARKETS={'points','rebounds','assists'}
def cdf(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--props',default='todays_props.csv'); ap.add_argument('--models',default='models'); ap.add_argument('--features',default='models/latest_player_features.csv'); ap.add_argument('--out',default='predictions.csv'); args=ap.parse_args()
    props=pd.read_csv(args.props)
    required={'game_date','player','team','opponent','home','minutes','market','line'}
    missing=required-set(props.columns)
    if missing: raise SystemExit(f'Missing columns in props file: {sorted(missing)}')
    features=pd.read_csv(args.features)
    key=lambda s:s.astype(str).str.lower().str.replace(r'[^a-z0-9]','',regex=True)
    props['player_key']=key(props.player); features['player_key']=key(features.PLAYER_NAME)
    merged=props.merge(features,on='player_key',how='left',suffixes=('','_hist'))
    unknown=merged.loc[merged.PLAYER_ID.isna(),'player'].drop_duplicates().tolist()
    if unknown: print('WARNING — no trained player match (excluded): '+', '.join(unknown))
    rows=[]; packs={}
    for _,r in merged.iterrows():
        market=str(r.market).strip().lower()
        if market not in MARKETS or pd.isna(r.get('PLAYER_ID')): continue
        if market not in packs: packs[market]=joblib.load(Path(args.models)/f'{market}_model.joblib')
        pack=packs[market]; X=pd.DataFrame([{c:r.get(c,np.nan) for c in pack['feature_cols']}])
        projection=float(pack['model'].predict(X)[0])
        avg_min=pd.to_numeric(r.get('MIN_L5',np.nan),errors='coerce'); listed=pd.to_numeric(r.minutes,errors='coerce')
        if pd.notna(avg_min) and avg_min>0 and pd.notna(listed): projection*=max(.5,min(1.5,float(listed)/float(avg_min)))
        line=float(r.line); sd=max(float(pack.get('residual_std',1.0)),.5)
        # Continuity correction for integer stat outcomes against a half-point/whole-number line.
        p_over=float(1-cdf((line+0.5-projection)/sd)); p_over=min(max(p_over,0.0001),.9999); p_under=1-p_over
        out=r.to_dict(); out.update({'projection':projection,'prob_over':p_over,'prob_under':p_under,'fair_odds_over':1/p_over,'fair_odds_under':1/p_under,'implied_probability':np.nan,'selected_side':str(r.get('side','over')).lower(),'edge':np.nan,'expected_value_per_unit':np.nan})
        odds=pd.to_numeric(r.get('odds',np.nan),errors='coerce')
        if pd.notna(odds) and odds>1:
            side=out['selected_side']; prob=p_under if side=='under' else p_over
            out['implied_probability']=1/float(odds); out['edge']=prob-1/float(odds); out['expected_value_per_unit']=prob*(float(odds)-1)-(1-prob)
        rows.append(out)
    result=pd.DataFrame(rows)
    if result.empty: raise SystemExit('No predictions produced. Check player names, markets and trained feature file.')
    result.to_csv(args.out,index=False)
    show=[c for c in ['player','market','line','projection','prob_over','prob_under','selected_side','edge','expected_value_per_unit'] if c in result]
    print(result[show].to_string(index=False)); print(f'\nSaved {len(result)} predictions to {args.out}')
    print('Reminder: probabilities are model estimates, not guarantees. Validate calibration and track results before staking.')
if __name__=='__main__': main()
