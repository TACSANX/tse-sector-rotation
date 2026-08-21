#!/usr/bin/env python3
"""Fast, survivorship-light backtest using only the 17 TOPIX-17 ETFs + CASH.

All 17 ETFs share the same 2008-03-25 listing start. This deliberately excludes
the 33-industry proxy layer and macro/fundamental inputs, so it tests whether the
tradable ETF rotation + absolute-trend + CASH core has merit on its own.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from screener import BENCHMARK, TOPIX17_ETFS


def cs_z(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).clip(-2.5, 2.5).fillna(0)


def z100(z: pd.DataFrame) -> pd.DataFrame:
    return 100.0 / (1.0 + np.exp(-1.15 * z))


def dirscore(x: pd.DataFrame | pd.Series, scale: float):
    return (50.0 + 50.0 * np.tanh(x * scale)).clip(0, 100)


def maxdd_rolling(close: pd.DataFrame, n=126):
    # rolling maximum drawdown approximation: worst drawdown from rolling peak
    roll_peak = close.rolling(n, min_periods=n//2).max()
    dd = close / roll_peak - 1
    return dd.rolling(n, min_periods=n//2).min()


def performance(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1+r).cumprod()
    years = (r.index[-1]-r.index[0]).days/365.25
    cagr = eq.iloc[-1]**(1/years)-1
    vol = r.std(ddof=0)*np.sqrt(252)
    sharpe = r.mean()/r.std(ddof=0)*np.sqrt(252) if r.std(ddof=0)>0 else np.nan
    downside = r[r<0].std(ddof=0)
    sortino = r.mean()/downside*np.sqrt(252) if pd.notna(downside) and downside>0 else np.nan
    dd = eq/eq.cummax()-1
    mdd = dd.min()
    return {
        "total_return": float(eq.iloc[-1]-1), "cagr": float(cagr), "volatility": float(vol),
        "sharpe": float(sharpe), "sortino": float(sortino), "max_drawdown": float(mdd),
        "calmar": float(cagr/abs(mdd)) if mdd<0 else np.nan,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--start', default='2008-03-25')
    ap.add_argument('--signal-start', default='2009-04-01')
    ap.add_argument('--rebalance', choices=['weekly','monthly'], default='monthly')
    ap.add_argument('--cost-bps', type=float, default=10.0)
    ap.add_argument('--out-dir', type=Path, default=Path('backtest_etf_only'))
    args=ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    tickers=list(TOPIX17_ETFS)+[BENCHMARK]
    raw=yf.download(tickers,start=args.start,auto_adjust=True,group_by='column',progress=False,threads=True,timeout=45)
    close=raw['Close'].copy(); vol=raw['Volume'].copy()
    close=close.dropna(how='all').ffill()
    etfs=list(TOPIX17_ETFS)
    close=close[[*etfs,BENCHMARK]]
    vol=vol.reindex(close.index)[etfs]
    b=close[BENCHMARK]
    e=close[etfs]

    ret1=e.pct_change(21); ret3=e.pct_change(63); ret6=e.pct_change(126)
    br1=b.pct_change(21); br3=b.pct_change(63); br6=b.pct_change(126)
    rs1=ret1.sub(br1,axis=0); rs3=ret3.sub(br3,axis=0); rs6=ret6.sub(br6,axis=0)
    ma50=e/e.rolling(50).mean()-1; ma200=e/e.rolling(200).mean()-1
    daily=e.pct_change(); vol20=daily.rolling(20).std(ddof=0)*np.sqrt(252)
    mdd=maxdd_rolling(e,126)
    turnover=(e*vol).rolling(20).mean()

    technical=z100(.18*cs_z(ret1)+.27*cs_z(ret3)+.15*cs_z(ret6)+.15*cs_z(rs3)+.10*cs_z(ma50)+.15*cs_z(ma200))
    rotation=z100(.25*cs_z(rs1)+.40*cs_z(rs3)+.20*cs_z(rs6)+.15*cs_z(rs1-rs3/3.0))
    risk=z100(-.65*cs_z(vol20)+.35*cs_z(mdd))
    liquidity=z100(cs_z(np.log1p(turnover.clip(lower=0))))
    absolute=.25*dirscore(ret3,5)+.20*dirscore(ret6,3)+.20*dirscore(ma50,9)+.35*dirscore(ma200,7)

    # No 33-industry/fundamental/macro layer in this clean execution-universe test.
    relative=.50*technical+.28*rotation+.12*risk+.10*liquidity
    score=.60*relative+.40*absolute
    score=score-(ma200<-.05)*8-(turnover<20_000_000)*5

    # market/CASH rule matched to production structure, minus VIX term.
    bm50=b/b.rolling(50).mean()-1; bm200=b/b.rolling(200).mean()-1
    market_abs=.25*dirscore(br3,5)+.20*dirscore(br6,3)+.20*dirscore(bm50,9)+.35*dirscore(bm200,7)
    breadth50=(ma50>0).mean(axis=1); breadth200=(ma200>0).mean(axis=1); breadth3=(ret3>0).mean(axis=1)
    breadth=100*(.30*breadth50+.45*breadth200+.25*breadth3)
    risk_on=(.68*market_abs+.32*breadth).clip(0,100)
    top_score=score.max(axis=1)
    cash=(100-risk_on + (62-top_score).clip(lower=0)*.80).clip(0,100)

    chosen=score.idxmax(axis=1).astype(object)
    chosen.loc[cash>top_score]='CASH'

    valid=chosen.index[chosen.index>=pd.Timestamp(args.signal_start)]
    if args.rebalance=='monthly':
        reb=pd.Series(valid,index=valid).groupby(valid.to_period('M')).last()
    else:
        reb=pd.Series(valid,index=valid).groupby(valid.to_period('W-FRI')).last()
    signal=chosen.reindex(pd.DatetimeIndex(reb.values)).dropna()

    # Signal at close, trade next trading day; returns accrue from the day after execution.
    dates=close.index
    execs={}
    for d,a in signal.items():
        f=dates[dates>d]
        if len(f): execs[f[0]]=a
    pos=pd.Series(index=dates[dates>=min(execs)],dtype=object)
    for d,a in execs.items():
        if d in pos.index: pos.loc[d]=a
    pos=pos.ffill().fillna('CASH')
    held=pos.shift(1).fillna('CASH')
    strat=pd.Series(0.0,index=pos.index)
    eret=e.pct_change().reindex(pos.index).fillna(0)
    for t in etfs:
        mask=held==t; strat.loc[mask]=eret.loc[mask,t]
    switch=pos.ne(pos.shift(1)); prev=pos.shift(1).fillna('CASH'); cost=args.cost_bps/10000
    costs=pd.Series(0.0,index=pos.index)
    for d in pos.index[switch]:
        sides=int(prev.loc[d]!='CASH')+int(pos.loc[d]!='CASH'); costs.loc[d]=sides*cost
    strat-=costs
    bret=b.pct_change().reindex(pos.index).fillna(0)

    # Always-invested top ETF comparison using identical ranking but no CASH.
    signal_nc=score.idxmax(axis=1).reindex(pd.DatetimeIndex(reb.values)).dropna()
    exec_nc={}
    for d,a in signal_nc.items():
        f=dates[dates>d]
        if len(f): exec_nc[f[0]]=a
    pos_nc=pd.Series(index=pos.index,dtype=object)
    for d,a in exec_nc.items():
        if d in pos_nc.index: pos_nc.loc[d]=a
    pos_nc=pos_nc.ffill().bfill()
    held_nc=pos_nc.shift(1).bfill()
    r_nc=pd.Series(0.0,index=pos.index)
    for t in etfs:
        mask=held_nc==t; r_nc.loc[mask]=eret.loc[mask,t]
    sw_nc=pos_nc.ne(pos_nc.shift(1)); prv_nc=pos_nc.shift(1).bfill(); c_nc=pd.Series(0.0,index=pos.index)
    for d in pos.index[sw_nc]: c_nc.loc[d]=(int(prv_nc.loc[d]!=pos_nc.loc[d])*2)*cost
    r_nc-=c_nc

    rows=[]
    for name,r,p in [('TOPIX-17 + CASH',strat,pos),('TOPIX-17 always invested',r_nc,pos_nc),('TOPIX 1306 buy&hold',bret,None)]:
        d={'strategy':name,**performance(r)}
        d['cash_exposure']=float((p=='CASH').mean()) if p is not None else 0.0
        d['switches']=int(p.ne(p.shift(1)).sum()) if p is not None else 0
        rows.append(d)
    summary=pd.DataFrame(rows)
    equity=pd.DataFrame({'with_cash':(1+strat).cumprod(),'always_invested':(1+r_nc).cumprod(),'topix':(1+bret).cumprod(),'position':pos})
    annual=pd.DataFrame({'with_cash':(1+strat).groupby(strat.index.year).prod()-1,'always_invested':(1+r_nc).groupby(r_nc.index.year).prod()-1,'topix':(1+bret).groupby(bret.index.year).prod()-1})
    summary.to_csv(args.out_dir/'summary.csv',index=False); equity.to_csv(args.out_dir/'equity.csv'); annual.to_csv(args.out_dir/'annual_returns.csv')
    meta={'period_start':str(pos.index[0].date()),'period_end':str(pos.index[-1].date()),'rebalance':args.rebalance,'cost_bps_per_side':args.cost_bps,'cash_signal_share':float((signal=='CASH').mean()),'signals':int(len(signal))}
    (args.out_dir/'results.json').write_text(json.dumps({'meta':meta,'summary':summary.to_dict(orient='records')},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(summary.to_string(index=False)); print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
