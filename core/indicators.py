"""
core/indicators.py
기술 지표 계산 (EMA, SMA, RSI, MACD, BB, ATR, OBV, ADX, 일목)
"""
import numpy as np
import pandas as pd


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d  = close.diff()
    g  = d.clip(lower=0)
    l  = (-d).clip(lower=0)
    ag = g.ewm(alpha=1 / n, adjust=False).mean()
    al = l.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))

def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ml = _ema(close, fast) - _ema(close, slow)
    sl = _ema(ml, signal)
    return ml, sl, ml - sl

def _bbands(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return ma - k * sd, ma, ma + k * sd

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def _obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["Close"].diff().fillna(0)) * df["Volume"]).cumsum()

def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc  = c.shift(1)
    tr  = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    dmp = (h - h.shift(1)).clip(lower=0)
    dmm = (l.shift(1) - l).clip(lower=0)
    dmp = dmp.where(dmp > dmm, 0)
    dmm = dmm.where(dmm > dmp, 0)
    atrs = tr.ewm(alpha=1 / n, adjust=False).mean()
    dip  = 100 * dmp.ewm(alpha=1 / n, adjust=False).mean() / atrs.replace(0, np.nan)
    dim  = 100 * dmm.ewm(alpha=1 / n, adjust=False).mean() / atrs.replace(0, np.nan)
    dx   = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()

def _ichimoku_spans(df: pd.DataFrame):
    h9  = df["High"].rolling(9).max();  l9  = df["Low"].rolling(9).min()
    h26 = df["High"].rolling(26).max(); l26 = df["Low"].rolling(26).min()
    h52 = df["High"].rolling(52).max(); l52 = df["Low"].rolling(52).min()
    tenkan = (h9  + l9)  / 2
    kijun  = (h26 + l26) / 2
    return ((tenkan + kijun) / 2).shift(26), ((h52 + l52) / 2).shift(26)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df is None or df.empty or not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
        return df

    for n in [5, 20, 60, 120]:
        df[f"SMA{n}"] = _sma(df["Close"], n)

    bbl, bbm, bbu = _bbands(df["Close"])
    df["BBL_20_2.0"] = bbl
    df["BBM_20_2.0"] = bbm
    df["BBU_20_2.0"] = bbu
    df["BBP_20_2.0"] = (df["Close"] - bbl) / (bbu - bbl).replace(0, np.nan)

    df["RSI14"]          = _rsi(df["Close"])
    ml, sl, hist         = _macd(df["Close"])
    df["MACD_12_26_9"]   = ml
    df["MACDs_12_26_9"]  = sl
    df["MACDh_12_26_9"]  = hist
    df["ATR14"]          = _atr(df)
    df["OBV"]            = _obv(df)
    df["ADX14"]          = _adx(df)
    spanA, spanB         = _ichimoku_spans(df)
    df["IchiSpanA"]      = spanA
    df["IchiSpanB"]      = spanB
    df["TurnoverAmt"]    = df["Close"] * df["Volume"]
    return df
