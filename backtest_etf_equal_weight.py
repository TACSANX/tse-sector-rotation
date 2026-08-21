#!/usr/bin/env python3
"""TOPIX-17 ETF rotation backtest against an equal-weight 17-ETF benchmark.

This is the clean execution-layer validation:
- universe: 1617.T ... 1633.T only
- synthetic benchmark: daily rebalanced equal-weight return of all 17 ETFs
- no 33-industry constituent history, macro, or fundamentals
- monthly/weekly rebalance
- signal at close, effective next trading day
- 10 bps per side default transaction cost
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from screener import TOPIX17_ETFS


def cs_z(x):
    mu=x.mean(axis=1); sd=x.std(axis=1,ddof=0).replace(0,np.nan)
    return x.sub(mu,axis=0).div(sd,axis=0).clip(-2.5,2.5).fillna(0)

def z100(z): return 100/(1+np.exp(-1.15*z))
def ds(x,s): return (50+50*np.tanh(x*s)).clip(0,100)

def perf(r):
    r=r.dropna(); eq=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.25
    cagr=eq.iloc[-1]**(1/yrs)-1; vol=r.std(ddof=0)*np.sqrt(252)
    sh=r.mean()/r.std(ddof=0)*np.sqrt(252) if r.std(ddof=0)>0 else np.nan
    dn=r[r<0].std(ddof=0); so=r.mean()/dn*np.sqrt(252) if pd.notna(dn) and dn>0 else np.nan
    dd=eq/eq.cummax()-1; mdd=dd.min()
    return dict(total_return=float(eq.iloc[-1]-1),cagr=float(cagr),volatility=float(vol),sharpe=float(sh),sortino=float(so),max_drawdown=float(mdd),calmar=float(cagr/abs(mdd)) if mdd<0 else np.nan)

def run(rebalance,cost_bps,outdir):
    outdir.mkdir(parents=True,exist_ok=True); tickers=list(TOPIX17_ETFS)
    raw=yf.download(tickers,start='2008-03-25',auto_adjust=True,group_by='column',progress=False,threads=False,timeout=45)
    e=raw['Close'][tickers].dropna(how='all').ffill(); volume=raw['Volume'][tickers].reindex(e.index)
    dr=e.pct_change(fill_method=None)
    # Broad sector ETFs should not exhibit >50% one-day adjusted returns; fail rather than hide split/data artifacts.
    mx=dr.abs().max(); bad=mx[mx>0.50]
    if len(bad):
        raise RuntimeError('unadjusted/suspect ETF price jumps: '+bad.to_json())
    bench_ret=dr.mean(axis=1,skipna=True).fillna(0); bench=(1+bench_ret).cumprod()
    r1=e.pct_change(21); r3=e.pct_change(63); r6=e.pct_change(126)
    b1=bench.pct_change(21); b3=bench.pct_change(63); b6=bench.pct_change(126)
    rs1=r1.sub(b1,axis=0); rs3=r3.sub(b3,axis=0); rs6=r6.sub(b6,axis=0)
    ma50=e/e.rolling(50).mean()-1; ma200=e/e.rolling(200).mean()-1
    vol20=dr.rolling(20).std(ddof=0)*np.sqrt(252)
    peak=e.rolling(126,min_periods=63).max(); dd=e/peak-1; mdd=dd.rolling(126,min_periods=63).min()
    turnover=(e*volume).rolling(20).mean()
    technical=z100(.18*cs_z(r1)+.27*cs_z(r3)+.15*cs_z(r6)+.15*cs_z(rs3)+.10*cs_z(ma50)+.15*cs_z(ma200))
    rotation=z100(.25*cs_z(rs1)+.40*cs_z(rs3)+.20*cs_z(rs6)+.15*cs_z(rs1-rs3/3))
    risk=z100(-.65*cs_z(vol20)+.35*cs_z(mdd)); liq=z100(cs_z(np.log1p(turnover.clip(lower=0))))
    absolute=.25*ds(r3,5)+.20*ds(r6,3)+.20*ds(ma50,9)+.35*ds(ma200,7)
    relative=.50*technical+.28*rotation+.12*risk+.10*liq
    score=.60*relative+.40*absolute-(ma200<-.05)*8-(turnover<20_000_000)*5
    # CASH score from equal-weight market trend and breadth.
    bm50=bench/bench.rolling(50).mean()-1; bm200=bench/bench.rolling(200).mean()-1
    market_abs=.25*ds(b3,5)+.20*ds(b6,3)+.20*ds(bm50,9)+.35*ds(bm200,7)
    breadth=100*(.30*(ma50>0).mean(axis=1)+.45*(ma200>0).mean(axis=1)+.25*(r3>0).mean(axis=1))
    risk_on=(.68*market_abs+.32*breadth).clip(0,100); top=score.max(axis=1)
    cash=(100-risk_on+(62-top).clip(lower=0)*.80).clip(0,100)
    valid=score.notna().any(axis=1); chosen=pd.Series(index=score.index,dtype=object)
    chosen.loc[valid]=score.loc[valid].idxmax(axis=1); chosen.loc[valid & (cash>top)]='CASH'
    idx=chosen.index[(chosen.index>=pd.Timestamp('2009-04-01')) & chosen.notna()]
    p=pd.Series(idx,index=idx)
    dates=pd.DatetimeIndex(p.groupby(idx.to_period('M' if rebalance=='monthly' else 'W-FRI')).last().values)
    sig=chosen.reindex(dates).dropna(); sig_nc=score.loc[dates].idxmax(axis=1).dropna()
    all_dates=e.index
    def simulate(sig,allow_cash):
        ex={}
        for d,a in sig.items():
            f=all_dates[all_dates>d]
            if len(f): ex[f[0]]=a
        pos=pd.Series(index=all_dates[all_dates>=min(ex)],dtype=object)
        for d,a in ex.items(): pos.loc[d]=a
        pos=pos.ffill().fillna('CASH')
        held=pos.shift(1).fillna('CASH'); ret=pd.Series(0.0,index=pos.index); drr=dr.reindex(pos.index).fillna(0)
        for t in tickers: ret.loc[held==t]=drr.loc[held==t,t]
        prev=pos.shift(1).fillna('CASH'); switches=pos.ne(prev); c=cost_bps/10000
        for d in pos.index[switches]: ret.loc[d]-=(int(prev.loc[d]!='CASH')+int(pos.loc[d]!='CASH'))*c
        return ret,pos
    sr,pos=simulate(sig,True); nr,npos=simulate(sig_nc,False); br=bench_ret.reindex(sr.index).fillna(0)
    rows=[]
    for name,r,p0 in [('rotation + CASH',sr,pos),('rotation always invested',nr,npos),('17 ETF equal-weight',br,None)]:
        d={'strategy':name,**perf(r),'cash_exposure':float((p0=='CASH').mean()) if p0 is not None else 0.0,'switches':int(p0.ne(p0.shift(1)).sum()) if p0 is not None else 0}; rows.append(d)
    summ=pd.DataFrame(rows); annual=pd.DataFrame({'rotation_cash':(1+sr).groupby(sr.index.year).prod()-1,'rotation_no_cash':(1+nr).groupby(nr.index.year).prod()-1,'equal_weight':(1+br).groupby(br.index.year).prod()-1})
    meta={'period_start':str(sr.index[0].date()),'period_end':str(sr.index[-1].date()),'rebalance':rebalance,'cost_bps_per_side':cost_bps,'signals':len(sig),'cash_signal_share':float((sig=='CASH').mean()),'max_abs_daily_return':float(mx.max())}
    summ.to_csv(outdir/'summary.csv',index=False); annual.to_csv(outdir/'annual_returns.csv'); (outdir/'results.json').write_text(json.dumps({'meta':meta,'summary':summ.to_dict('records')},ensure_ascii=False,indent=2)+'\n')
    print(summ.to_string(index=False)); print(json.dumps(meta,ensure_ascii=False,indent=2))

def main():
    a=argparse.ArgumentParser(); a.add_argument('--rebalance',choices=['monthly','weekly'],default='monthly'); a.add_argument('--cost-bps',type=float,default=10); a.add_argument('--out-dir',type=Path,required=True); x=a.parse_args(); run(x.rebalance,x.cost_bps,x.out_dir)
if __name__=='__main__': main()
