"""
core/utils.py
Shared helpers for persistence, regime detection, and trade zones.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

KR_TZ = pytz.timezone("Asia/Seoul")
ATR_MULT = 1.0
MAX_PCT = 0.04
MIN_GAP = 0.01


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, data):
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_favorites(path: Path):
    return _load_json(path, [])


def save_favorites(path: Path, f):
    _save_json(path, f)


def load_portfolio(path: Path):
    return _load_json(path, {})


def save_portfolio(path: Path, p):
    _save_json(path, p)


def is_spac(name: str) -> bool:
    n = str(name).strip()
    return "스팩" in n or "SPAC" in n.upper() or bool(re.search(r"\d+호$", n))


def is_preferred(name: str) -> bool:
    """Heuristic filter for Korean preferred shares (우선주·전환우선주 포함)."""
    n = str(name).strip()
    nu = n.upper()
    if not n:
        return False
    # 전환우선주: 이름에 '우(전환)' 패턴 포함
    if "우(전환)" in n:
        return True
    nu_end_candidates = ["우", "우B", "우C", "1우", "2우B", "3우C"]
    return any(nu.endswith(c.upper()) for c in nu_end_candidates) or "PREF" in nu


def is_managed_issue(name: str) -> bool:
    """Best-effort exclusion for managed/special-status issues."""
    n = str(name).strip()
    keywords = ["관리종목", "투자주의", "투자경고", "투자환기", "정리매매", "불성실공시"]
    return any(kw in n for kw in keywords)


def market_regime(df_kospi) -> str:
    if df_kospi is None or len(df_kospi) < 60:
        return "sideways"
    c = df_kospi["Close"]
    s20 = c.rolling(20).mean().iloc[-1]
    s60 = c.rolling(60).mean().iloc[-1]
    r20 = c.iloc[-1] / c.iloc[-20] - 1
    if s20 > s60 and r20 > 0.03:
        return "bull"
    if s20 < s60 and r20 < -0.03:
        return "bear"
    return "sideways"


def regime_weights(regime: str) -> dict:
    if regime == "bull":
        return {"trend": 0.30, "momentum": 0.40, "flow": 0.20, "volume": 0.05, "ichi": 0.05}
    if regime == "bear":
        return {"trend": 0.45, "momentum": 0.10, "flow": 0.30, "volume": 0.05, "ichi": 0.10}
    return {"trend": 0.35, "momentum": 0.25, "flow": 0.25, "volume": 0.10, "ichi": 0.05}


def find_fib_levels(df: pd.DataFrame, lookback: int = 180) -> dict:
    if df is None or df.empty:
        return {}
    sub = df.tail(lookback)
    high = sub["High"].max()
    low = sub["Low"].min()
    if pd.isna(high) or high == low:
        return {}
    d = high - low
    lv = {
        "0.0%": high,
        "23.6%": high - 0.236 * d,
        "38.2%": high - 0.382 * d,
        "50.0%": high - 0.500 * d,
        "61.8%": high - 0.618 * d,
        "78.6%": high - 0.786 * d,
        "100%": low,
    }
    sma20 = df["SMA20"].iloc[-1] if "SMA20" in df.columns else np.nan
    direction = "up" if not pd.isna(sma20) and df["Close"].iloc[-1] >= sma20 else "down"
    return {"high": high, "low": low, "levels": lv, "direction": direction}


def _atr_half(df: pd.DataFrame, fib: dict) -> float:
    last = df.iloc[-1]
    close = float(last["Close"])
    cands = []
    atr = last.get("ATR14", np.nan)
    if not pd.isna(atr):
        cands.append(float(atr) * ATR_MULT)
    cands.append(close * MAX_PCT)
    if fib and "levels" in fib:
        lv = fib["levels"]
        cands.append(abs(lv["38.2%"] - lv["61.8%"]) / 2)
    return min(cands) if cands else close * 0.02


def suggest_trade_zones(df: pd.DataFrame, fib: dict) -> dict:
    if df is None or df.empty:
        return {}
    cur = float(df.iloc[-1]["Close"])
    half = _atr_half(df, fib)
    gu = lambda x: max(x, cur * (1 + MIN_GAP))
    gd = lambda x: min(x, cur * (1 - MIN_GAP))
    return {
        "buy_zone": (gd(max(cur - 2 * half, 0)), gd(cur - half)),
        "sell_zone": (gu(cur + half), gu(cur + 2 * half)),
        "stop_loss": gd(cur - 3 * half),
        "take_profit": (gu(cur + 2 * half), gu(cur + 3 * half)),
    }


def multi_tf_trend(df: pd.DataFrame, fib: dict) -> dict:
    res = {"short": {}, "mid": {}, "long": {}}
    if df is None or df.empty:
        return res
    last = df.iloc[-1]
    close = float(last["Close"])
    for key, ma_col, tail_n in [("short", "SMA20", 20), ("mid", "SMA60", 60), ("long", "SMA120", 120)]:
        ma = float(last.get(ma_col, np.nan))
        trend = "UP" if not pd.isna(ma) and close >= ma else ("DOWN" if not pd.isna(ma) else "N/A")
        half = _atr_half(df, fib)
        center = ma if not pd.isna(ma) else close
        lo_sw = float(df["Low"].tail(tail_n).min())
        hi_sw = float(df["High"].tail(tail_n).max())
        buy_z = (max(center - half, lo_sw), center)
        sell_z = (center, min(center + half, hi_sw))
        res[key] = {
            "trend": trend,
            "buy_zone": (round(buy_z[0], -1), round(buy_z[1], -1)),
            "sell_zone": (round(sell_z[0], -1), round(sell_z[1], -1)),
        }
    return res
