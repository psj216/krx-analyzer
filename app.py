
# KRX Auto Analyzer v0.4

import os, json, warnings, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import requests, pytz

from pathlib import Path
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from pykrx import stock

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*use_container_width.*")

# ==================== 전역 ====================
FAV_PATH       = Path("favorites.json")
PORTFOLIO_PATH = Path("portfolio.json")
KR_TZ          = pytz.timezone("Asia/Seoul")
FETCH_TIMEOUT  = 5
LOOKBACK       = 180
ATR_MULT       = 1.0
MAX_PCT        = 0.04
MIN_GAP        = 0.01

st.set_page_config(page_title="KRX Auto Analyzer v0.4", layout="wide", page_icon="📈")

# ==================== 유틸 ====================
def _load_json(path, default):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except: return default

def _save_json(path, data):
    try: path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except: pass

def load_favorites():  return _load_json(FAV_PATH, [])
def save_favorites(f): _save_json(FAV_PATH, f)
def load_portfolio():  return _load_json(PORTFOLIO_PATH, {})
def save_portfolio(p): _save_json(PORTFOLIO_PATH, p)

def is_spac(name: str) -> bool:
    n = str(name)
    return "스팩" in n or "SPAC" in n.upper() or bool(re.search(r'제\d+호', n))

# ==================== KRX 심볼 ====================
@st.cache_data(ttl=3600*6)
def load_krx_symbols():
    cache = Path("krx_symbols_cache.csv")
    def from_cache():
        try:
            df = pd.read_csv(cache, encoding="utf-8-sig")
            if {"Code","Name","Market"}.issubset(df.columns) and len(df)>0:
                df["Code"] = df["Code"].astype(str).str.zfill(6)
                return df
        except: pass
        return None

    def from_pykrx():
        try:
            rows=[]
            for mkt in ["KOSPI","KOSDAQ"]:
                for c in (stock.get_market_ticker_list(market=mkt) or []):
                    rows.append({"Code":str(c).zfill(6),
                                 "Name":stock.get_market_ticker_name(c),
                                 "Market":mkt})
            df = pd.DataFrame(rows).dropna().drop_duplicates("Code").reset_index(drop=True)
            return df if len(df)>0 else None
        except: return None

    df = from_pykrx()
    if df is None: df = from_cache()
    if df is None: return pd.DataFrame(columns=["Code","Name","Market"])

    # 스팩 제거
    df = df[~df["Name"].apply(is_spac)].reset_index(drop=True)
    try: df.to_csv(cache, index=False, encoding="utf-8-sig")
    except: pass
    return df

# ==================== 지수 (pykrx + yfinance fallback) ====================
@st.cache_data(ttl=300)
def get_index_info():
    result = {}
    today = datetime.now(KR_TZ).strftime("%Y%m%d")
    start = (datetime.now(KR_TZ)-timedelta(days=10)).strftime("%Y%m%d")

    for name, krx_code, yf_ticker in [("KOSPI","1001","^KS11"),("KOSDAQ","2001","^KQ11")]:
        # 1) pykrx
        try:
            df = stock.get_index_ohlcv_by_date(start, today, krx_code)
            if df is not None and len(df) >= 2:
                last = float(df["종가"].iloc[-1])
                prev = float(df["종가"].iloc[-2])
                result[name] = {"price":last, "chg":(last-prev)/prev*100}
                continue
        except: pass

        # 2) yfinance fallback
        try:
            df = yf.download(yf_ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = df.dropna(subset=["Close"])
            if len(df) >= 2:
                last = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                result[name] = {"price":last, "chg":(last-prev)/prev*100}
                continue
        except: pass

        # 3) 네이버 폴링 fallback
        try:
            nv_code = "KOSPI" if name=="KOSPI" else "KOSDAQ"
            url = f"https://finance.naver.com/sise/sise_index.naver?code={nv_code}"
            res = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=5)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(res.text, "html.parser")
            now_tag  = soup.select_one("#now_value")
            prev_tag = soup.select_one("#prev_value")
            if now_tag and prev_tag:
                last = float(now_tag.text.replace(",",""))
                prev = float(prev_tag.text.replace(",",""))
                result[name] = {"price":last, "chg":(last-prev)/prev*100}
                continue
        except: pass

        result[name] = None
    return result

# ==================== Fetch ====================
def market_suffix(market):
    m = (market or "").upper()
    if "KOSPI"  in m: return ".KS"
    if "KOSDAQ" in m: return ".KQ"
    return ".KS"

def to_yf_symbol(code, market):
    return f"{str(code).zfill(6)}{market_suffix(market)}"

def _fetch_pykrx(code, start_krx, end_krx):
    df = stock.get_market_ohlcv_by_date(start_krx, end_krx, code)
    if df is not None and not df.empty:
        df = df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
        df.index = pd.to_datetime(df.index)
        return df[["Open","High","Low","Close","Volume"]].dropna()
    return pd.DataFrame()

def _fetch_yf(code, market):
    sym = to_yf_symbol(code, market)
    df  = yf.download(sym, period="2y", interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    df.index = pd.to_datetime(df.index)
    return df[[c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]].dropna()

def fetch_prices(code: str, market: str) -> pd.DataFrame:
    s = (datetime.now(KR_TZ)-timedelta(days=730)).strftime("%Y%m%d")
    e = (datetime.now(KR_TZ)+timedelta(days=1)).strftime("%Y%m%d")
    for fn, args in [(_fetch_yf,(code,market)), (_fetch_pykrx,(code,s,e))]:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn, *args)
            try:
                df = fut.result(timeout=FETCH_TIMEOUT)
                if df is not None and not df.empty: return df
            except: pass
    return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_prices_cached(code, market): return fetch_prices(code, market)

def get_realtime_price(code):
    try:
        from bs4 import BeautifulSoup
        res = requests.get(f"https://finance.naver.com/item/main.naver?code={code}",
                           headers={"User-Agent":"Mozilla/5.0"}, timeout=3)
        tag = BeautifulSoup(res.text,"html.parser").select_one("p.no_today .blind")
        if tag: return float(tag.text.replace(",",""))
    except: pass
    return None

# ==================== 지표 ====================
def _ema(s,n): return s.ewm(span=n, adjust=False).mean()
def _sma(s,n): return s.rolling(n).mean()

def _rsi(close, n=14):
    d=close.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    ag=g.ewm(alpha=1/n,adjust=False).mean()
    al=l.ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+ag/al.replace(0,np.nan))

def _macd(close,fast=12,slow=26,signal=9):
    ml=_ema(close,fast)-_ema(close,slow)
    sl=_ema(ml,signal); return ml,sl,ml-sl

def _bbands(close,n=20,k=2.0):
    ma=close.rolling(n).mean(); sd=close.rolling(n).std(ddof=0)
    return ma-k*sd, ma, ma+k*sd

def _atr(df,n=14):
    h,l,c=df["High"],df["Low"],df["Close"]; pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def _obv(df):
    return (np.sign(df["Close"].diff().fillna(0))*df["Volume"]).cumsum()

def _adx(df,n=14):
    h,l,c=df["High"],df["Low"],df["Close"]; pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    dmp=(h-h.shift(1)).clip(lower=0); dmm=(l.shift(1)-l).clip(lower=0)
    dmp=dmp.where(dmp>dmm,0); dmm=dmm.where(dmm>dmp,0)
    atrs=tr.ewm(alpha=1/n,adjust=False).mean()
    dip=100*dmp.ewm(alpha=1/n,adjust=False).mean()/atrs.replace(0,np.nan)
    dim=100*dmm.ewm(alpha=1/n,adjust=False).mean()/atrs.replace(0,np.nan)
    dx=100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean()

def _ichimoku_spans(df):
    h9=df["High"].rolling(9).max(); l9=df["Low"].rolling(9).min()
    h26=df["High"].rolling(26).max(); l26=df["Low"].rolling(26).min()
    h52=df["High"].rolling(52).max(); l52=df["Low"].rolling(52).min()
    tenkan=(h9+l9)/2; kijun=(h26+l26)/2
    return ((tenkan+kijun)/2).shift(26), ((h52+l52)/2).shift(26)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    if df is None or df.empty or not {"Open","High","Low","Close","Volume"}.issubset(df.columns):
        return df
    for n in [5,20,60,120]: df[f"SMA{n}"] = _sma(df["Close"],n)
    bbl,bbm,bbu = _bbands(df["Close"])
    df["BBL_20_2.0"]=bbl; df["BBM_20_2.0"]=bbm; df["BBU_20_2.0"]=bbu
    df["BBP_20_2.0"] = (df["Close"]-bbl)/(bbu-bbl).replace(0,np.nan)
    df["RSI14"] = _rsi(df["Close"])
    ml,sl,hist = _macd(df["Close"])
    df["MACD_12_26_9"]=ml; df["MACDs_12_26_9"]=sl; df["MACDh_12_26_9"]=hist
    df["ATR14"] = _atr(df)
    df["OBV"]   = _obv(df)
    df["ADX14"] = _adx(df)
    spanA,spanB = _ichimoku_spans(df)
    df["IchiSpanA"]=spanA; df["IchiSpanB"]=spanB
    df["TurnoverAmt"] = df["Close"]*df["Volume"]
    return df

# ==================== 시장 국면 ====================
def market_regime(df_kospi) -> str:
    if df_kospi is None or len(df_kospi)<60: return "sideways"
    c=df_kospi["Close"]
    s20=c.rolling(20).mean().iloc[-1]; s60=c.rolling(60).mean().iloc[-1]
    r20=c.iloc[-1]/c.iloc[-20]-1
    if s20>s60 and r20>0.03: return "bull"
    if s20<s60 and r20<-0.03: return "bear"
    return "sideways"

def regime_weights(regime) -> dict:
    if regime=="bull":    return {"trend":0.30,"momentum":0.40,"flow":0.20,"volume":0.05,"ichi":0.05}
    elif regime=="bear":  return {"trend":0.45,"momentum":0.10,"flow":0.30,"volume":0.05,"ichi":0.10}
    else:                 return {"trend":0.35,"momentum":0.25,"flow":0.25,"volume":0.10,"ichi":0.05}

# ==================== 시그널 ====================
def basic_signals(df: pd.DataFrame) -> dict:
    if df is None or df.empty or len(df)<20: return {}
    n=len(df); last=df.iloc[-1]; prev=df.iloc[-2] if n>=2 else last

    def xo(an,bn,ap,bp):
        if any(pd.isna(x) for x in [an,bn,ap,bp]): return False
        return ap<=bp and an>bn
    def xu(an,bn,ap,bp):
        if any(pd.isna(x) for x in [an,bn,ap,bp]): return False
        return ap>=bp and an<bn

    sig={}
    rsi=last.get("RSI14",np.nan); close=last.get("Close",np.nan)
    bb_u=last.get("BBU_20_2.0",np.nan); bb_l=last.get("BBL_20_2.0",np.nan)

    sig["rsi_overbought"] = (not pd.isna(rsi)) and rsi>=70
    sig["rsi_oversold"]   = (not pd.isna(rsi)) and rsi<=30
    sig["bb_breakout_up"] = (not pd.isna(bb_u)) and close>bb_u
    sig["bb_breakout_dn"] = (not pd.isna(bb_l)) and close<bb_l

    if n>=60:
        m=last.get("MACD_12_26_9",np.nan); ms=last.get("MACDs_12_26_9",np.nan)
        sig["macd_bull_cross"] = xo(m,ms,prev.get("MACD_12_26_9"),prev.get("MACDs_12_26_9"))
        sig["macd_bear_cross"] = xu(m,ms,prev.get("MACD_12_26_9"),prev.get("MACDs_12_26_9"))

    sma20=last.get("SMA20",np.nan); sma60=last.get("SMA60",np.nan)
    if n>=120 and not pd.isna(sma20) and not pd.isna(sma60):
        sig["trend_ma20_gt_ma60"] = "UP" if sma20>sma60 else ("DOWN" if sma20<sma60 else "FLAT")
        sig["golden_cross"] = xo(sma20,sma60,prev.get("SMA20"),prev.get("SMA60"))
        sig["death_cross"]  = xu(sma20,sma60,prev.get("SMA20"),prev.get("SMA60"))
    else:
        sig["trend_ma20_gt_ma60"]="N/A"; sig["golden_cross"]=False; sig["death_cross"]=False

    if n>=250:
        h52=df["High"].tail(252).max(); l52=df["Low"].tail(252).min()
        sig["near_52w_high"]=close>=h52*0.97; sig["near_52w_low"]=close<=l52*1.03
        sig["52w_high"]=h52; sig["52w_low"]=l52
    else:
        sig["near_52w_high"]=False; sig["near_52w_low"]=False

    if "TurnoverAmt" in df.columns and n>=20:
        ta=float(df["TurnoverAmt"].iloc[-1])
        ta_avg=float(df["TurnoverAmt"].tail(20).mean())
        sig["turnover_surge"]=ta>ta_avg*3.0
        sig["turnover_ratio"]=ta/ta_avg if ta_avg>0 else 0

    sug=[]
    if sig.get("golden_cross"):   sug.append("골든크로스")
    if sig.get("death_cross"):    sug.append("데드크로스")
    if sig.get("near_52w_high"):  sug.append("52주 신고가 근접")
    if sig.get("near_52w_low"):   sug.append("52주 신저가 근접")
    if sig.get("turnover_surge"): sug.append(f"거래대금 급등 {sig.get('turnover_ratio',0):.1f}배")
    sig["suggestions"]=sug
    return sig

# ==================== 캔들 패턴 ====================
def _candle_bull_engulfing(df):
    if len(df)<2: return False
    p,c=df.iloc[-2],df.iloc[-1]
    if not (p["Close"]<p["Open"] and c["Close"]>c["Open"]): return False
    bp=abs(p["Close"]-p["Open"]); bc=abs(c["Close"]-c["Open"])
    return bc>bp and c["Open"]<=p["Close"] and c["Close"]>=p["Open"]

def _candle_breakout_long(df):
    if len(df)<21: return False
    last=df.iloc[-1]
    if last["Close"]<=last["Open"]: return False
    bodies=(df["Close"]-df["Open"]).tail(40).abs()
    med=bodies.median()
    if pd.isna(med) or med==0: return False
    return abs(last["Close"]-last["Open"])>1.5*med and last["Close"]>df["High"].iloc[-21:-1].max()

# ==================== 점수 ====================
def score_technical(df: pd.DataFrame, signals: dict, fib: dict) -> dict:
    empty={"score":0,"score_raw":0.0,"trend_score":0,"trend_raw":0.0,
           "momentum_score":0,"momentum_raw":0.0,"breakout_score":0,
           "overheat_score":0,"liquidity_risk":0,"grade":"데이터 부족",
           "reasons":[],"reasons_trend":[],"reasons_momentum":[]}
    if df is None or df.empty or not signals: return empty

    last=df.iloc[-1]; prev=df.iloc[-2] if len(df)>=2 else last
    close=float(last["Close"]); open_=float(last.get("Open",close))
    high=float(last.get("High",close)); low=float(last.get("Low",close))
    n=len(df)
    rt=[]; rm=[]

    def pctrank(series, val, window=120):
        sub=series.dropna().tail(window)
        return float((sub<val).mean()*100) if len(sub)>=10 else 50.0

    rsi_now=float(last.get("RSI14",np.nan)) if not pd.isna(last.get("RSI14",np.nan)) else None
    sma5=float(last.get("SMA5",np.nan)); sma20=float(last.get("SMA20",np.nan))
    sma60=float(last.get("SMA60",np.nan)); sma120=float(last.get("SMA120",np.nan))
    adx=float(last.get("ADX14",np.nan)); atr_val=float(last.get("ATR14",np.nan))
    vol_now=float(last["Volume"])

    vol_pct=pctrank(df["Volume"],vol_now,60)
    gap5_series=(df["Close"]/df["SMA5"]-1).dropna() if "SMA5" in df.columns else pd.Series()
    gap5_now=(close/sma5-1) if not pd.isna(sma5) and sma5 else 0
    gap5_pct=pctrank(gap5_series,gap5_now,120) if len(gap5_series)>=10 else 50.0
    day_ret=(close/float(df["Close"].iloc[-2])-1) if n>=2 else 0

    # ===== TREND =====
    tr=50.0
    if not pd.isna(sma120):
        c=4.0 if close>sma120 else -5.0; tr+=c; rt.append(f"SMA120 {'상방' if c>0 else '하방'} → {c:+.1f}")
    if n>=2 and not pd.isna(sma20):
        sp=float(df["SMA20"].iloc[-2])
        if sp:
            s=(sma20-sp)/sp
            c=3.0 if 0<s<0.003 else (1.0 if s<0.01 else (-4.0 if s>=0.01 else -1.0))
            tr+=c; rt.append(f"SMA20 기울기 {s*100:.2f}% → {c:+.1f}")
    if n>=2 and not pd.isna(sma60):
        sp=float(df["SMA60"].iloc[-2])
        if sp:
            s=(sma60-sp)/sp
            c=2.0 if 0<s<0.003 else (1.0 if s<0.007 else (-3.0 if s>=0.007 else -1.0))
            tr+=c; rt.append(f"SMA60 기울기 {s*100:.2f}% → {c:+.1f}")
    if not pd.isna(sma20) and sma20:
        g=close/sma20-1
        c=3.0 if abs(g)<=0.05 else (0.0 if g<=0.10 else (-6.0 if g>0.10 else -2.0))
        tr+=c; rt.append(f"20일선 이격 {g*100:.1f}% → {c:+.1f}")
    if not pd.isna(sma20) and not pd.isna(sma60) and sma20 and sma60:
        c=3.0 if sma20>sma60 else -5.0; tr+=c
        rt.append(f"20/60 {'정배열' if c>0 else '역배열'} → {c:+.1f}")
    if "BBM_20_2.0" in df.columns:
        mid=float(last["BBM_20_2.0"]); c=2.0 if close>mid else -3.0
        tr+=c; rt.append(f"볼밴 중단 {'상방' if c>0 else '하방'} → {c:+.1f}")
    if "OBV" in df.columns and n>=2:
        c=2.0 if float(last["OBV"])>float(df["OBV"].iloc[-2]) else -2.0
        tr+=c; rt.append(f"OBV {'상승' if c>0 else '하락'} → {c:+.1f}")
    if not pd.isna(adx):
        c=-3.0 if adx<20 else (3.0 if adx<30 else (6.0 if adx<40 else -2.0))
        tr+=c; rt.append(f"ADX {adx:.1f} → {c:+.1f}")
    if "IchiSpanA" in df.columns and "IchiSpanB" in df.columns:
        spA=float(last["IchiSpanA"]); spB=float(last["IchiSpanB"])
        if not pd.isna(spA) and not pd.isna(spB):
            cloud=abs(spA-spB)/close
            c=-3.0 if cloud<0.02 else (1.0 if cloud<0.05 else (3.0 if cloud<0.10 else -2.0))
            tr+=c; rt.append(f"일목 구름 {cloud*100:.1f}% → {c:+.1f}")
    if fib and "levels" in fib and fib.get("direction")=="up":
        lv=fib["levels"]
        lo,hi=min(lv["61.8%"],lv["38.2%"]),max(lv["61.8%"],lv["38.2%"])
        if lo<=close<=hi: tr+=2.0; rt.append("피보 38.2~61.8% 구간 → +2")

    # ===== MOMENTUM =====
    mo=50.0
    is_breakout=_candle_breakout_long(df)

    # 5일선
    if not pd.isna(sma5) and sma5:
        above5 = close>sma5
        if above5: mo+=4; rm.append("5일선 상방 → +4")
        if n>=2:
            sp=float(df["SMA5"].iloc[-2])
            sl5=(sma5-sp)/sp if sp else 0
            rising5 = sl5>0
            if rising5: mo+=3; rm.append("5일선 우상향 → +3")
            # 둘 다 만족 콤보
            if above5 and rising5: mo+=2; rm.append("5일선 상방+우상향 콤보 → +2")
            # 급경사 감점 완화
            if sl5>0.015: mo-=2; rm.append("5일선 급경사 → -2")
        # 이격 상대화
        if gap5_pct>=95:   mo-=4; rm.append(f"5일선 이격 자기분포 상위5% → -4")
        elif gap5_pct>=90: mo-=2; rm.append(f"5일선 이격 자기분포 상위10% → -2")
        elif 0<gap5_now<=0.03: mo+=3; rm.append("5일선 이격 0~3% → +3")

    # RSI (완화된 기준)
    if rsi_now is not None:
        if 50<rsi_now<=55:  mo+=4; rm.append("RSI 50~55 → +4")
        elif 55<rsi_now<=65: mo+=3; rm.append("RSI 55~65 → +3")
        if 70<=rsi_now<75:  mo-=1; rm.append("RSI 70~75 → -1")
        elif 75<=rsi_now<80: mo-=3; rm.append("RSI 75~80 → -3")
        elif rsi_now>=80:   mo-=5; rm.append("RSI 80↑ → -5")

    # RSI slope
    if n>=3 and "RSI14" in df.columns and rsi_now is not None:
        rs1=float(df["RSI14"].iloc[-1])-float(df["RSI14"].iloc[-2])
        rs2=float(df["RSI14"].iloc[-2])-float(df["RSI14"].iloc[-3])
        if rs1>0 and rs2>0 and 30<rsi_now<65:
            mo+=4; rm.append("RSI 연속 상승 가속 → +4")
        elif rs1<0 and rs2<0 and rsi_now>50:
            mo-=3; rm.append("RSI 연속 하락 → -3")

    # MACD
    if "MACD_12_26_9" in df.columns and n>=3:
        mv=float(last["MACD_12_26_9"]); sv=float(last["MACDs_12_26_9"])
        hist=mv-sv
        hp=float(df["MACD_12_26_9"].iloc[-2])-float(df["MACDs_12_26_9"].iloc[-2])
        ms1=float(df["MACD_12_26_9"].iloc[-1])-float(df["MACD_12_26_9"].iloc[-2])
        ms2=float(df["MACD_12_26_9"].iloc[-2])-float(df["MACD_12_26_9"].iloc[-3])
        if hist>0 and hp<=0: mo+=5; rm.append("MACD 양전 초입 → +5")
        if hist>0 and hp>0 and hist>hp:
            if ms1>0 and ms1>ms2: mo+=5; rm.append("MACD 증가+가속 → +5")
            else: mo+=3; rm.append("MACD 증가 → +3")
        if hp>0 and hist<hp: mo-=4; rm.append("MACD 약화 → -4")

    # 볼린저
    if all(c in df.columns for c in ["BBL_20_2.0","BBM_20_2.0","BBU_20_2.0"]) and n>=2:
        bbu_v=float(last["BBU_20_2.0"]); bbl_v=float(last["BBL_20_2.0"]); bbm_v=float(last["BBM_20_2.0"])
        w_now=(bbu_v-bbl_v)/(bbm_v or 1)
        bp=float(df["BBU_20_2.0"].iloc[-2]); blp=float(df["BBL_20_2.0"].iloc[-2]); bmp=float(df["BBM_20_2.0"].iloc[-2])
        w_prev=(bp-blp)/(bmp or 1)
        if w_prev<0.05 and w_now>w_prev*1.1: mo+=3; rm.append("볼밴 squeeze→확장 → +3")
        dist=(close-bbu_v)/(bbu_v or 1)
        if 0.03<dist<=0.05: mo-=3; rm.append("볼밴 과열 3~5% → -3")
        if dist>0.05: mo-=6; rm.append("볼밴 심과열 → -6")

    # 거래량 + 맥락
    if "Volume" in df.columns:
        vol20=float(df["Volume"].tail(20).mean())
        r=vol_now/vol20 if vol20>0 else 1.0
        if vol_now<10000:  mo-=8; rm.append("거래량 1만↓ → -8")
        elif vol_now<30000: mo-=4; rm.append("거래량 1~3만 → -4")
        if vol_pct<=20: mo-=6; rm.append("거래량 자기분포 하위20% → -6")
        elif 1.2<=r<=2.5:
            mo+=(5 if is_breakout else 3)
            rm.append(f"거래량 1.2~2.5배{'+ 돌파양봉' if is_breakout else ''} → +{5 if is_breakout else 3}")
        elif 2.5<r<=5:
            mo+=(4 if is_breakout else 1)
            rm.append(f"거래량 2.5~5배{'+ 돌파양봉' if is_breakout else ''} → +{4 if is_breakout else 1}")
        elif 5<r<=8: mo-=2; rm.append("거래량 5~8배 → -2")
        elif r>8:    mo-=5; rm.append("거래량 8배↑ → -5")

    # 거래대금 급등 점수 연결
    if signals.get("turnover_surge"):
        body=abs(close-open_); rng_c=max(high-low,1e-9)
        tail_up=(high-max(open_,close))/rng_c
        if tail_up<0.3: mo+=3; rm.append("거래대금 급등+윗꼬리 작음 → +3")
        else:           mo+=1; rm.append("거래대금 급등(윗꼬리 있음) → +1")

    # 캔들 구조
    body=abs(close-open_); rng_c=max(high-low,1e-9)
    tail_up=(high-max(open_,close))/rng_c
    tail_dn=(min(open_,close)-low)/rng_c
    if close>open_ and body/rng_c>=0.6 and tail_up<0.2:
        mo+=(5 if vol_pct>=60 else 2); rm.append(f"장대양봉{'+ 거래량' if vol_pct>=60 else ''} → +{5 if vol_pct>=60 else 2}")
    if tail_dn>=0.4 and close>=open_: mo+=3; rm.append("아래꼬리 반등형 → +3")
    if tail_up>=0.4: mo-=4; rm.append("윗꼬리 40%↑ → -4")
    if _candle_bull_engulfing(df):
        mo+=(4 if vol_pct>=60 else 2); rm.append(f"상승 장악형{'+ 거래량' if vol_pct>=60 else ''} → +{4 if vol_pct>=60 else 2}")

    # 전일 대비 (ATR 상대화)
    if not pd.isna(atr_val) and atr_val>0 and close>0:
        ap=atr_val/close
        if abs(day_ret)>ap*2.5: mo-=5; rm.append(f"ATR 대비 일변동 과도 → -5")
        elif 0.01<=day_ret<=ap*1.5: mo+=2; rm.append(f"전일 +{day_ret*100:.1f}% 적정 → +2")
        elif day_ret>ap*1.5: mo-=3; rm.append(f"전일 급등 ATR초과 → -3")
    else:
        if 0.01<=day_ret<=0.03: mo+=2
        elif 0.03<day_ret<=0.06: mo+=3
        elif day_ret>0.06: mo-=6

    # ===== 파생 점수 =====
    breakout=0
    if is_breakout: breakout+=40
    if _candle_bull_engulfing(df): breakout+=20
    if signals.get("bb_breakout_up"): breakout+=20
    if signals.get("macd_bull_cross"): breakout+=20
    breakout=min(100,breakout)

    overheat=0
    if rsi_now and rsi_now>=70: overheat+=30
    if rsi_now and rsi_now>=80: overheat+=20
    if gap5_pct>=95: overheat+=25
    if signals.get("bb_breakout_up"): overheat+=25
    overheat=min(100,overheat)

    liq_risk=0
    if vol_now<10000: liq_risk+=50
    elif vol_now<30000: liq_risk+=25
    if vol_pct<=20: liq_risk+=30
    liq_risk=min(100,liq_risk)

    tr=float(np.clip(tr,0,100)); mo=float(np.clip(mo,0,100))
    sr=(tr+mo)/2; score=int(round(sr))

    grade=("매수 유망" if score>=72 else
           "매수 관심" if score>=60 else
           "관망/보유" if score>=48 else "주의")

    return {"score":score,"score_raw":sr,"trend_score":int(round(tr)),"trend_raw":tr,
            "momentum_score":int(round(mo)),"momentum_raw":mo,
            "breakout_score":breakout,"overheat_score":overheat,"liquidity_risk":liq_risk,
            "grade":grade,"reasons":rt+rm,"reasons_trend":rt,"reasons_momentum":rm}

# ==================== 수급 ====================
def _investor_flow_score(code):
    def fetch(days):
        try:
            e=datetime.now().strftime("%Y%m%d")
            s=(datetime.now()-timedelta(days=days)).strftime("%Y%m%d")
            df=stock.get_market_trading_value_by_investor(s,e,code)
            if df is None or df.empty: return None
            col=("순매수" if "순매수" in df.columns else df.columns[-1])
            gn=lambda nm: float(df.loc[nm,col]) if nm in df.index else 0.0
            return {"외국인":gn("외국인"),"기관":gn("기관합계"),"연기금":gn("연기금 등"),
                    "base":float(df[col].abs().sum()) or 1.0}
        except: return None

    r5=fetch(5); r20=fetch(20)
    def sc(r):
        if not r: return 50
        w=0.30*r["외국인"]+0.40*r["기관"]+0.30*r["연기금"]
        return int(max(0,min(100,50+35*np.tanh(w/r["base"]))))
    ss=sc(r5); sm=sc(r20)
    if r5 and r20:
        n5=r5["외국인"]+r5["기관"]; n20=r20["외국인"]+r20["기관"]
        sp=70 if (n5>0 and n20>0) else (30 if (n5<0 and n20<0) else 50)
    else: sp=50
    final=int(0.40*ss+0.35*sm+0.25*sp)
    return final, r5 or {}, {"short":ss,"mid":sm,"persist":sp}

def _volume_score(df,days=20):
    try:
        lv=float(df["Volume"].iloc[-1]); avg=float(df["Volume"].tail(days).mean())
        return int(max(0,min(100,50+40*np.tanh(lv/avg-1)))) if avg>0 else 50
    except: return 50

# ==================== 피보나치 / 구간 ====================
def find_fib_levels(df, lookback=180):
    if df is None or df.empty: return {}
    sub=df.tail(lookback); high=sub["High"].max(); low=sub["Low"].min()
    if pd.isna(high) or high==low: return {}
    d=high-low
    lv={"0.0%":high,"23.6%":high-0.236*d,"38.2%":high-0.382*d,
        "50.0%":high-0.5*d,"61.8%":high-0.618*d,"78.6%":high-0.786*d,"100%":low}
    sma20=df["SMA20"].iloc[-1] if "SMA20" in df.columns else np.nan
    direction="up" if not pd.isna(sma20) and df["Close"].iloc[-1]>=sma20 else "down"
    return {"high":high,"low":low,"levels":lv,"direction":direction}

def _atr_half(df,fib):
    last=df.iloc[-1]; close=float(last["Close"]); cands=[]
    atr=last.get("ATR14",np.nan)
    if not pd.isna(atr): cands.append(float(atr)*ATR_MULT)
    cands.append(close*MAX_PCT)
    if fib and "levels" in fib:
        lv=fib["levels"]; cands.append(abs(lv["38.2%"]-lv["61.8%"])/2)
    return min(cands) if cands else close*0.02

def suggest_trade_zones(df,fib):
    if df is None or df.empty: return {}
    cur=float(df.iloc[-1]["Close"]); half=_atr_half(df,fib)
    gu=lambda x: max(x,cur*(1+MIN_GAP)); gd=lambda x: min(x,cur*(1-MIN_GAP))
    return {"buy_zone":(gd(max(cur-2*half,0)),gd(cur-half)),
            "sell_zone":(gu(cur+half),gu(cur+2*half)),
            "stop_loss":gd(cur-3*half),
            "take_profit":(gu(cur+2*half),gu(cur+3*half))}

def multi_tf_trend(df,fib):
    res={"short":{},"mid":{},"long":{}}
    if df is None or df.empty: return res
    last=df.iloc[-1]; close=float(last["Close"])
    for key,ma_col,tail_n in [("short","SMA20",20),("mid","SMA60",60),("long","SMA120",120)]:
        ma=float(last.get(ma_col,np.nan))
        trend="UP" if not pd.isna(ma) and close>=ma else ("DOWN" if not pd.isna(ma) else "N/A")
        half=_atr_half(df,fib); center=ma if not pd.isna(ma) else close
        lo_sw=float(df["Low"].tail(tail_n).min()); hi_sw=float(df["High"].tail(tail_n).max())
        buy_z=(max(center-half,lo_sw),center); sell_z=(center,min(center+half,hi_sw))
        res[key]={"trend":trend,
                  "buy_zone":(round(buy_z[0],-1),round(buy_z[1],-1)),
                  "sell_zone":(round(sell_z[0],-1),round(sell_z[1],-1))}
    return res

# ==================== 일목 ====================
def ichimoku_signals(df):
    try:
        h9=df["High"].rolling(9).max(); l9=df["Low"].rolling(9).min()
        h26=df["High"].rolling(26).max(); l26=df["Low"].rolling(26).min()
        h52=df["High"].rolling(52).max(); l52=df["Low"].rolling(52).min()
        tk=(h9+l9)/2; kj=(h26+l26)/2
        spA=((tk+kj)/2).shift(26); spB=((h52+l52)/2).shift(26)
        chikou=df["Close"].shift(-26); lc=df["Close"].iloc[-1]
        sig={}
        sig["전환-기준"]="강세" if tk.iloc[-1]>kj.iloc[-1] else "약세"
        if lc>max(spA.iloc[-1],spB.iloc[-1]): sig["구름"]="상승구간"
        elif lc<min(spA.iloc[-1],spB.iloc[-1]): sig["구름"]="하락구간"
        else: sig["구름"]="중립"
        sig["후행스팬"]="상승 확인" if (len(df)>=27 and chikou.iloc[-1]>df["Close"].iloc[-26]) else "하락 확인"
        return sig
    except: return {}

# ==================== 점수 전체 ====================
def compute_score(code,name,market,df,lookback=400,regime="sideways"):
    try:
        if df is None or df.empty or len(df)<60: return None
        df=df.tail(lookback); df=add_indicators(df)
        if len(df)<60: return None
        fib=find_fib_levels(df); sig=basic_signals(df); sc=score_technical(df,sig,fib)
        ichi=ichimoku_signals(df)
        ichi_sc=(10 if ichi.get("전환-기준")=="강세" else 0)+ \
                (10 if ichi.get("구름")=="상승구간" else 0)+ \
                (5  if ichi.get("후행스팬")=="상승 확인" else 0)
        flow,_,_ = _investor_flow_score(code)
        vol=_volume_score(df)
        w=regime_weights(regime)
        fr=w["trend"]*sc["trend_raw"]+w["momentum"]*sc["momentum_raw"]+ \
           w["flow"]*flow+w["volume"]*vol+w["ichi"]*ichi_sc
        return {"Code":code,"Name":name,"Market":market,
                "Score":int(round(fr)),"ScoreRaw":float(fr),
                "Trend":sc["trend_score"],"Momentum":sc["momentum_score"],
                "Breakout":sc["breakout_score"],"Overheat":sc["overheat_score"],
                "LiqRisk":sc["liquidity_risk"],"Flow":flow,"Vol":vol,
                "Close":float(df["Close"].iloc[-1])}
    except: return None

# ==================== 랭킹 ====================
def rank_top_scores(all_syms,universe="ALL",limit=1000,lookback=400,regime="sideways"):
    syms=all_syms.copy()
    if universe=="KOSPI":  syms=syms[syms["Market"]=="KOSPI"]
    elif universe=="KOSDAQ": syms=syms[syms["Market"]=="KOSDAQ"]
    syms=syms[syms["Market"].isin(["KOSPI","KOSDAQ"])].head(limit).reset_index(drop=True)

    rows=[]; total=len(syms)
    rej={"no_data":0,"too_short":0,"score_none":0,"exc":0,"ok":0}

    def process_one(row):
        code=str(row["Code"]).zfill(6); name=str(row["Name"]); market=str(row.get("Market","KOSPI"))
        try:
            df=fetch_prices(code,market)
            if df is None or df.empty: return "no_data",None
            if len(df)<max(60,int(lookback*0.8)): return "too_short",None
            res=compute_score(code,name,market,df,lookback,regime)
            return ("ok",res) if res else ("score_none",None)
        except: return "exc",None

    prog=st.progress(0.0,text="점수 계산 중…")
    completed=[0]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures={ex.submit(process_one,row):i for i,(_,row) in enumerate(syms.iterrows())}
        try:
            for fut in as_completed(futures, timeout=total*5+120):
                try:
                    status,res=fut.result(timeout=FETCH_TIMEOUT+3)
                    rej[status]=rej.get(status,0)+1
                    if res: rows.append(res)
                except: rej["exc"]=rej.get("exc",0)+1
                completed[0]+=1
                prog.progress(completed[0]/max(total,1),text=f"점수 계산 중… {completed[0]}/{total}")
        except FuturesTimeout:
            st.warning(f"⏱ 시간 초과 — {completed[0]}/{total} 완료")

    prog.empty()
    st.caption(f"완료 {rej.get('ok',0)} / 데이터없음 {rej.get('no_data',0)} / 짧음 {rej.get('too_short',0)} / 오류 {rej.get('exc',0)}")
    if not rows: return pd.DataFrame(columns=["Code","Name","Market","Score","Close"])
    return pd.DataFrame(rows).sort_values("ScoreRaw",ascending=False).reset_index(drop=True)

# ==================== 거래대금 스캐너 ====================
def scan_turnover_surge(all_syms,limit=200,threshold=3.0):
    results=[]; syms=all_syms[all_syms["Market"].isin(["KOSPI","KOSDAQ"])].head(limit)

    def check_one(row):
        code=str(row["Code"]).zfill(6); name=str(row["Name"]); market=str(row.get("Market","KOSPI"))
        try:
            df=fetch_prices(code,market)
            if df is None or len(df)<21: return None
            close=float(df["Close"].iloc[-1]); vol=float(df["Volume"].iloc[-1])
            ta=close*vol; ta_avg=(df["Close"]*df["Volume"]).tail(20).mean()
            ratio=ta/ta_avg if ta_avg>0 else 0
            if ratio<threshold: return None
            ret1d=(df["Close"].iloc[-1]/df["Close"].iloc[-2]-1)*100 if len(df)>=2 else 0
            return {"Code":code,"Name":name,"Market":market,
                    "거래대금(억)":round(ta/1e8,1),"평균대비":round(ratio,1),
                    "전일대비%":round(float(ret1d),2),"현재가":round(close,0)}
        except: return None

    prog=st.progress(0.0,text="거래대금 스캔 중…")
    total=len(syms); done=[0]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures={ex.submit(check_one,row):i for i,(_,row) in enumerate(syms.iterrows())}
        try:
            for fut in as_completed(futures, timeout=total*4+60):
                try:
                    r=fut.result(timeout=FETCH_TIMEOUT+3)
                    if r: results.append(r)
                except: pass
                done[0]+=1
                prog.progress(done[0]/max(total,1),text=f"스캔 중… {done[0]}/{total}")
        except FuturesTimeout:
            st.warning(f"⏱ 시간 초과 — {done[0]}/{total} 완료")

    prog.empty()
    if not results: return pd.DataFrame()
    return pd.DataFrame(results).sort_values("평균대비",ascending=False).reset_index(drop=True)

# ==================== 백테스트 ====================
def simple_backtest(df,signal_col="golden_cross",hold_days=5):
    if df is None or len(df)<120: return {}
    df=add_indicators(df.copy()); signals=[]
    for i in range(1,len(df)-hold_days):
        sub=df.iloc[:i+1]; sig=basic_signals(sub)
        if sig.get(signal_col):
            entry=float(df["Close"].iloc[i]); exit_=float(df["Close"].iloc[i+hold_days])
            signals.append({"date":str(df.index[i])[:10],"entry":int(entry),"exit":int(exit_),"ret%":round((exit_-entry)/entry*100,2)})
    if not signals: return {"count":0}
    rets=[s["ret%"] for s in signals]
    return {"count":len(rets),"win_rate":round(sum(1 for r in rets if r>0)/len(rets)*100,1),
            "avg_ret":round(float(np.mean(rets)),2),"max_ret":round(float(np.max(rets)),2),
            "min_ret":round(float(np.min(rets)),2),"signals":signals[-10:]}

# ==================== 차트 ====================
def indicator_snapshot(df):
    cols=["Close","SMA5","SMA20","SMA60","SMA120","BBL_20_2.0","BBM_20_2.0","BBU_20_2.0",
          "BBP_20_2.0","RSI14","MACD_12_26_9","MACDs_12_26_9","ADX14","ATR14","Volume"]
    use=[c for c in cols if c in df.columns]
    last=df[use].iloc[[-1]].copy(); last.index=["Latest"]; return last

def make_chart(df,title,fib=None):
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.03,row_heights=[0.55,0.25,0.20])
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],
                                  low=df["Low"],close=df["Close"],name="Price"),row=1,col=1)
    for ma,col in [("SMA5","orange"),("SMA20","blue"),("SMA60","green"),("SMA120","red")]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[ma],name=ma,mode="lines",
                                      line=dict(color=col,width=1.2)),row=1,col=1)
    for col in ["BBL_20_2.0","BBM_20_2.0","BBU_20_2.0"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[col],name=col,mode="lines",opacity=0.4),row=1,col=1)
    if fib and "levels" in fib:
        for label,level in fib["levels"].items():
            fig.add_hline(y=level,line_dash="dot",opacity=0.2,
                          annotation_text=f"Fib {label}",row=1,col=1)
    if "RSI14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["RSI14"],name="RSI14",
                                  line=dict(color="purple",width=1.2)),row=2,col=1)
        fig.add_hline(y=70,line_dash="dash",line_color="red",opacity=0.5,row=2,col=1)
        fig.add_hline(y=30,line_dash="dash",line_color="blue",opacity=0.5,row=2,col=1)
    if "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="Volume"),row=3,col=1)
    fig.update_layout(title=title,xaxis_rangeslider_visible=False,hovermode="x unified",
                      height=900,xaxis3_title="Date",yaxis_title="Price",
                      yaxis2_title="RSI",yaxis3_title="Volume")
    return fig

def make_narrative(code,name,market,df,signals,zones,fib):
    if df is None or df.empty or not signals: return "데이터 부족."
    last=df.iloc[-1]; close=float(last["Close"])
    sma20=float(last.get("SMA20",np.nan)) if not pd.isna(last.get("SMA20",np.nan)) else None
    rsi=float(last.get("RSI14",np.nan)) if not pd.isna(last.get("RSI14",np.nan)) else None
    lines=[f"**{code} {name} ({market})**",f"- 현재가: {close:,.0f}원"]
    if sma20: lines.append(f"- 20일선: {sma20:,.0f}원 ({'위' if close>=sma20 else '아래'})")
    if rsi:   lines.append(f"- RSI(14): {rsi:.1f}")
    if signals.get("near_52w_high"): lines.append(f"- ⚠️ 52주 신고가({signals.get('52w_high',0):,.0f}원) 근접")
    if signals.get("near_52w_low"):  lines.append(f"- 📌 52주 신저가({signals.get('52w_low',0):,.0f}원) 근접")
    if signals.get("turnover_surge"): lines.append(f"- 🔥 거래대금 급등 {signals.get('turnover_ratio',0):.1f}배")
    if zones:
        if "buy_zone"  in zones: lines.append(f"- 매수 후보: {zones['buy_zone'][0]:,.0f}~{zones['buy_zone'][1]:,.0f}원")
        if "sell_zone" in zones: lines.append(f"- 매도 후보: {zones['sell_zone'][0]:,.0f}~{zones['sell_zone'][1]:,.0f}원")
        if "stop_loss" in zones: lines.append(f"- 손절 가이드: {zones['stop_loss']:,.0f}원")
    sug=signals.get("suggestions",[])
    if sug: lines.append("- 시그널: "+(", ".join(sug)))
    return "\n".join(lines)

# ==================== UI ====================
st.markdown("## 📈 KRX Auto Analyzer v0.4")

# ==================== 사이드바 ====================
with st.sidebar:
    if "favorites"  not in st.session_state: st.session_state["favorites"]  = load_favorites()
    if "portfolio"  not in st.session_state: st.session_state["portfolio"]  = load_portfolio()
    if "sel_override" not in st.session_state: st.session_state["sel_override"] = None

    # 코스피 / 코스닥 지수
    idx_info=get_index_info()
    for name in ["KOSPI","KOSDAQ"]:
        ki=idx_info.get(name)
        if ki: st.metric(name, f"{ki['price']:,.2f}", delta=f"{ki['chg']:+.2f}%", delta_color="normal")
        else:  st.metric(name, "조회 중…")
    if st.button("지수 새로고침", key="idx_refresh"):
        get_index_info.clear(); st.rerun()

    st.markdown("---")
    st.subheader("종목 검색")
    all_syms=load_krx_symbols()
    st.caption(f"KRX {len(all_syms)}개 (스팩 제외)")

    query=st.text_input("코드/종목명 검색")
    filtered=all_syms
    if query:
        q=query.strip().upper()
        filtered=all_syms[
            all_syms["Code"].str.upper().str.contains(q,na=False) |
            all_syms["Name"].str.upper().str.contains(q,na=False)
        ].head(50)

    selection=st.selectbox("종목 선택",options=[""]+[
        f"{r.Code} | {r.Name} | {r.Market}" for _,r in filtered.iterrows()])
    if st.session_state.get("sel_override"):
        selection=st.session_state["sel_override"]; st.session_state["sel_override"]=None

    st.markdown("---")
    st.subheader("⭐ 즐겨찾기")
    favs=st.session_state["favorites"]
    if favs:
        for i,fav in enumerate(favs):
            parts=[x.strip() for x in fav.split("|")]
            c1s,c2s=st.columns([5,1],gap="small")
            with c1s:
                if st.button(parts[1],key=f"fav_{i}",use_container_width=True):
                    st.session_state["sel_override"]=fav; st.rerun()
            with c2s:
                if st.button("🗑",key=f"del_{i}"):
                    st.session_state["favorites"].pop(i)
                    save_favorites(st.session_state["favorites"]); st.rerun()
    else:
        st.caption("비어 있음.")

# ==================== TOP10 ====================
st.markdown("---")
st.subheader("🔥 오늘의 점수 TOP10")

@st.cache_data(ttl=3600)
def get_kospi_regime():
    try:
        today=datetime.now(KR_TZ).strftime("%Y%m%d")
        start=(datetime.now(KR_TZ)-timedelta(days=100)).strftime("%Y%m%d")
        df=stock.get_index_ohlcv_by_date(start,today,"1001")
        if df is not None and len(df)>=60:
            df=df.rename(columns={"종가":"Close","시가":"Open","고가":"High","저가":"Low","거래량":"Volume"})
            return market_regime(df)
    except: pass
    return "sideways"

cur_regime=get_kospi_regime()
regime_label = {"bull":"📈 상승장","sideways":"📊 횡보장","bear":"📉 하락장"}.get(cur_regime,"📊")
st.info(f"장 국면: **{regime_label}** (가중치 자동 조정)")


col_uni,col_n,col_days=st.columns(3)
with col_uni:  uni      =st.selectbox("대상",["ALL","KOSPI","KOSDAQ"],index=0)
with col_n:    scan_n   =st.slider("스캔 종목 수",50,1000,1000,50)
with col_days: look_days=st.slider("지표 Lookback(일)",100,500,200,10)

if st.button("계산/업데이트", use_container_width=True):
    try:
        result=rank_top_scores(all_syms,universe=uni,limit=scan_n,lookback=look_days,regime=cur_regime)
        st.session_state["top_rank_df"]=result
    except Exception as e: st.error(f"랭킹 오류: {e}")

topdf=st.session_state.get("top_rank_df")
if isinstance(topdf,pd.DataFrame) and not topdf.empty:
    def _grade(x):
        try: x=float(x)
        except: return "N/A"
        return ("매수 유망" if x>=72 else "매수 관심" if x>=60 else "관망/보유" if x>=48 else "주의")
    show=topdf.copy(); show["Grade"]=show["Score"].apply(_grade)
    disp=[c for c in ["Code","Name","Market","Score","Grade","Trend","Momentum","Breakout","Overheat","LiqRisk","Close"] if c in show.columns]
    st.dataframe(show.head(10)[disp].reset_index(drop=True), use_container_width=True, height=360)
    if any(topdf.get("near_52w_high",pd.Series([False]))):
        nh=topdf[topdf.get("near_52w_high",False)==True]
        if not nh.empty: st.success("🏆 52주 신고가 근접: "+(", ".join(nh["Name"].tolist())))
elif isinstance(topdf,pd.DataFrame): st.warning("결과 없음.")
else: st.caption("버튼 눌러서 계산해라.")

# ==================== 거래대금 급등 스캐너 ====================
st.markdown("---")
st.subheader("⚡ 거래대금 급등 스캐너")
sc1,sc2=st.columns(2)
with sc1: sc_limit    =st.slider("스캔 종목 수",50,500,200,50,key="sc_lim")
with sc2: sc_threshold=st.slider("급등 기준(20일평균 대비 배수)",2.0,10.0,3.0,0.5,key="sc_thr")
if st.button("급등 스캔 실행", use_container_width=True):
    st.session_state["surge_df"]=scan_turnover_surge(all_syms,limit=sc_limit,threshold=sc_threshold)

surge_df=st.session_state.get("surge_df")
if isinstance(surge_df,pd.DataFrame) and not surge_df.empty:
    st.dataframe(surge_df, use_container_width=True, height=300)
elif isinstance(surge_df,pd.DataFrame): st.info("조건에 맞는 종목 없음.")
else: st.caption("버튼 눌러서 스캔해라.")

# ==================== 포트폴리오 ====================
st.markdown("---")
st.subheader("💼 포트폴리오 트래킹")
port=st.session_state["portfolio"]
with st.expander("종목 추가/삭제"):
    pc1,pc2,pc3,pc4=st.columns(4)
    with pc1: p_code=st.text_input("종목코드",key="p_code")
    with pc2: p_name=st.text_input("종목명",key="p_name")
    with pc3: p_avg=st.number_input("평균단가",min_value=0.0,step=100.0,key="p_avg")
    with pc4: p_qty=st.number_input("수량",min_value=0,step=1,key="p_qty")
    if st.button("추가",key="p_add"):
        if p_code and p_avg>0 and p_qty>0:
            port[p_code]={"name":p_name,"avg":float(p_avg),"qty":int(p_qty)}
            save_portfolio(port); st.session_state["portfolio"]=port; st.success(f"{p_code} 추가됨"); st.rerun()
    del_code=st.text_input("삭제할 코드",key="p_del")
    if st.button("삭제",key="p_del_btn"):
        if del_code in port:
            del port[del_code]; save_portfolio(port); st.session_state["portfolio"]=port; st.rerun()

if port:
    rows_p=[]
    for code,info in port.items():
        cur_p=get_realtime_price(code) or info["avg"]
        pnl=(cur_p-info["avg"])*info["qty"]; pnl_pct=(cur_p-info["avg"])/info["avg"]*100 if info["avg"] else 0
        rows_p.append({"코드":code,"종목명":info.get("name",""),"평균단가":f"{info['avg']:,.0f}",
                        "현재가":f"{cur_p:,.0f}","수량":info["qty"],
                        "평가손익":f"{pnl:+,.0f}원","수익률":f"{pnl_pct:+.2f}%"})
    st.dataframe(pd.DataFrame(rows_p), use_container_width=True)
else: st.caption("보유 종목을 추가해라.")

# ==================== 개별 종목 분석 ====================
st.markdown("---")

def run_once():
    if not selection: st.info("좌측에서 종목을 골라라."); return

    code,name,market=[x.strip() for x in selection.split("|")]
    st.markdown(f"### {code} — {name} ({market})")

    fav_key=f"{code} | {name} | {market}"
    is_fav=fav_key in st.session_state["favorites"]
    cf1,cf2=st.columns([1,5])
    with cf1:
        if st.button("⭐ 제거" if is_fav else "⭐ 추가",key="fav_toggle",use_container_width=True):
            if is_fav: st.session_state["favorites"]=[f for f in st.session_state["favorites"] if f!=fav_key]
            else:      st.session_state["favorites"]=[fav_key]+st.session_state["favorites"]
            save_favorites(st.session_state["favorites"]); st.rerun()

    df=fetch_prices_cached(code,market)
    rt=get_realtime_price(code)
    if rt and df is not None and not df.empty:
        df=df.copy(); df.iloc[-1,df.columns.get_loc("Close")]=rt

    if df is None or df.empty: st.error("데이터 불러오기 실패."); return

    df_chart=add_indicators(df.copy())
    score_df=add_indicators(df.tail(look_days).copy())
    fib=find_fib_levels(score_df,lookback=min(LOOKBACK,len(score_df)-1))
    signals=basic_signals(score_df)
    zones=suggest_trade_zones(score_df,fib)
    snap=indicator_snapshot(score_df)
    sc=score_technical(score_df,signals,fib)
    ichi=ichimoku_signals(score_df)
    ichi_sc=(10 if ichi.get("전환-기준")=="강세" else 0)+ \
            (10 if ichi.get("구름")=="상승구간"   else 0)+ \
            (5  if ichi.get("후행스팬")=="상승 확인" else 0)
    flow_sc,flow_raw,flow_detail=_investor_flow_score(code)
    vol_sc=_volume_score(score_df)
    w=regime_weights(cur_regime)
    final_sc=int(w["trend"]*sc["trend_raw"]+w["momentum"]*sc["momentum_raw"]+
                 w["flow"]*flow_sc+w["volume"]*vol_sc+w["ichi"]*ichi_sc)

    if signals.get("near_52w_high"): st.warning(f"⚠️ 52주 신고가({signals.get('52w_high',0):,.0f}원) 근접")
    if signals.get("near_52w_low"):  st.info(f"📌 52주 신저가({signals.get('52w_low',0):,.0f}원) 근접")
    if signals.get("turnover_surge"): st.success(f"🔥 거래대금 급등 {signals.get('turnover_ratio',0):.1f}배")

    st.subheader("📊 종합 점수")
    s1,s2,s3,s4,s5=st.columns(5)
    s1.metric("최종점수",  final_sc)
    s2.metric("추세점수",  sc["trend_score"])
    s3.metric("모멘텀점수",sc["momentum_score"])
    s4.metric("돌파점수",  sc["breakout_score"])
    s5.metric("과열위험",  sc["overheat_score"])

    with st.expander("유형별 점수 상세"):
        st.markdown(f"""
| 유형 | 점수 | 비고 |
|---|---|---|
| 추세 | **{sc['trend_score']}** | SMA·OBV·ADX·일목 |
| 모멘텀 | **{sc['momentum_score']}** | RSI slope·MACD·거래량 맥락 |
| 돌파 | **{sc['breakout_score']}** | 장대양봉·장악형·볼밴·MACD 골든 |
| 과열 위험 ⚠ | **{sc['overheat_score']}** | 높을수록 추격 위험 |
| 저유동성 ⚠ | **{sc['liquidity_risk']}** | 높을수록 유동성 부족 |
| 수급 | **{flow_sc}** | 단기{flow_detail['short']} / 중기{flow_detail['mid']} / 지속{flow_detail['persist']} |
| 거래량 | **{vol_sc}** | 오늘 vs 20일 평균 |
| 일목 | **{ichi_sc}** | 전환·구름·후행스팬 |
""")
        st.markdown("**추세:** "+" | ".join(sc.get("reasons_trend",[])))
        st.markdown("**모멘텀:** "+" | ".join(sc.get("reasons_momentum",[])))

    mtf=multi_tf_trend(score_df,fib)
    st.subheader("⏱️ 다중 타임프레임")
    mtf_rows=[]
    for key,label in [("short","단기(SMA20)"),("mid","중기(SMA60)"),("long","장기(SMA120)")]:
        row={"구간":label,"추세":mtf[key].get("trend","N/A")}
        bz=mtf[key].get("buy_zone"); sz=mtf[key].get("sell_zone")
        row["매수 후보"]=f"{bz[0]:,.0f}~{bz[1]:,.0f}" if bz else "-"
        row["매도 후보"]=f"{sz[0]:,.0f}~{sz[1]:,.0f}" if sz else "-"
        mtf_rows.append(row)
    st.dataframe(pd.DataFrame(mtf_rows), use_container_width=True)

    st.subheader("요약")
    st.markdown(make_narrative(code,name,market,score_df,signals,zones,fib))

    fig=make_chart(df_chart,f"{code} {name}",fib)
    st.plotly_chart(fig, use_container_width=True)

    c1s,c2s=st.columns([2,1])
    with c1s: st.subheader("지표 스냅샷"); st.dataframe(snap.round(2), use_container_width=True)
    with c2s: st.subheader("시그널"); st.json(signals)
    st.subheader("매수/매도 구간 (교육용)"); st.json(zones)

    st.markdown("---")
    st.subheader("🧪 백테스트 간단 요약")
    bt1,bt2=st.columns(2)
    with bt1: bt_signal=st.selectbox("시그널",["golden_cross","macd_bull_cross","bb_breakout_up"],key="bt_sig")
    with bt2: bt_hold=st.slider("보유일수",1,20,5,1,key="bt_hold")
    if st.button("백테스트 실행",key="bt_run",use_container_width=True):
        with st.spinner("계산 중…"):
            bt=simple_backtest(df,signal_col=bt_signal,hold_days=bt_hold)
        if bt.get("count",0)==0: st.info("시그널 발생 없음.")
        else:
            st.markdown(f"- 발생: **{bt['count']}회** / 승률: **{bt['win_rate']}%** / 평균수익: **{bt['avg_ret']}%** / 최대: {bt['max_ret']}% / 최소: {bt['min_ret']}%")
            st.dataframe(pd.DataFrame(bt["signals"]), use_container_width=True)

    st.markdown("---")
    st.subheader("🧮 진입가 계산기")
    last_close=float(df["Close"].iloc[-1])
    entry=st.number_input("진입가(원)",min_value=0.0,step=100.0,format="%.0f",value=last_close)
    rc1,rc2,rc3,rc4=st.columns(4)
    with rc1: risk_pct=st.slider("손절폭(%)",0.5,10.0,3.0,0.5)
    with rc2: tp_pct=st.slider("익절폭(%)",1.0,20.0,5.0,0.5)
    with rc3: capital=st.number_input("총자본(원)",min_value=0.0,step=100000.0,format="%.0f")
    with rc4: risk_pt=st.slider("트레이드 리스크(%)",0.5,5.0,1.0,0.5)
    if entry>0:
        sl=entry*(1-risk_pct/100); tp=entry*(1+tp_pct/100); rr=(tp-entry)/max(entry-sl,1e-9)
        st.write(f"손절가: **{sl:,.0f}원** / 익절가: **{tp:,.0f}원** / 손익비: **{rr:.2f}**")
        if capital>0:
            ml=capital*(risk_pt/100); sz=int(ml/max(entry-sl,1e-9))
            st.write(f"권장 수량: **{sz:,}주** (최대 손실 {ml:,.0f}원)")

    st.download_button("CSV 다운로드",df.to_csv().encode("utf-8"),file_name=f"{code}.csv",mime="text/csv")

run_once()
