"""
core/scorer.py
Technical scoring, investor-flow scoring, and composite ranking score.
"""
from datetime import datetime, timedelta
import random
import time

import numpy as np
import pandas as pd
from pykrx import stock

from core.indicators import add_indicators
from core.signals import basic_signals, candle_bull_engulfing, candle_breakout_long, ichimoku_signals
from core.utils import KR_TZ, find_fib_levels, regime_weights

MIN_PRICE = 1_000
MIN_AVG_TURNOVER_20 = 3_000_000_000
SOFT_MIN_AVG_TURNOVER_5 = 5_000_000_000


def score_technical(df: pd.DataFrame, signals: dict, fib: dict) -> dict:
    empty = {
        "score": 0,
        "score_raw": 0.0,
        "trend_score": 0,
        "trend_raw": 0.0,
        "momentum_score": 0,
        "momentum_raw": 0.0,
        "breakout_score": 0,
        "overheat_score": 0,
        "liquidity_risk": 100,
        "grade": "no_data",
        "reasons": [],
        "reasons_trend": [],
        "reasons_momentum": [],
    }
    if df is None or df.empty or not signals:
        return empty

    last = df.iloc[-1]
    close = float(last["Close"])
    open_ = float(last.get("Open", close))
    high = float(last.get("High", close))
    low = float(last.get("Low", close))
    n = len(df)
    rt, rm = [], []

    def pctrank(series, val, window=120):
        sub = series.dropna().tail(window)
        return float((sub < val).mean() * 100) if len(sub) >= 10 else 50.0

    rsi_now = float(last.get("RSI14", np.nan)) if not pd.isna(last.get("RSI14", np.nan)) else None
    sma5 = float(last.get("SMA5", np.nan))
    sma20 = float(last.get("SMA20", np.nan))
    sma60 = float(last.get("SMA60", np.nan))
    sma120 = float(last.get("SMA120", np.nan))
    adx = float(last.get("ADX14", np.nan))
    atr_val = float(last.get("ATR14", np.nan))
    vol_now = float(last["Volume"])
    ta_now = float(last.get("TurnoverAmt", close * vol_now))
    ta20 = float(df["TurnoverAmt"].tail(20).mean()) if "TurnoverAmt" in df.columns else ta_now
    ta5 = float(df["TurnoverAmt"].tail(5).mean()) if "TurnoverAmt" in df.columns else ta_now

    vol_pct = pctrank(df["Volume"], vol_now, 60)
    gap5_series = (df["Close"] / df["SMA5"] - 1).dropna() if "SMA5" in df.columns else pd.Series(dtype=float)
    gap5_now = (close / sma5 - 1) if not pd.isna(sma5) and sma5 else 0
    gap5_pct = pctrank(gap5_series, gap5_now, 120) if len(gap5_series) >= 10 else 50.0
    day_ret = (close / float(df["Close"].iloc[-2]) - 1) if n >= 2 else 0

    trend = 50.0
    if not pd.isna(sma120):
        c = 6.0 if close > sma120 else -6.0
        trend += c
        rt.append(f"close_vs_sma120 {c:+.1f}")
    if not pd.isna(sma20) and not pd.isna(sma60):
        c = 5.0 if sma20 > sma60 else -5.0
        trend += c
        rt.append(f"sma20_vs_sma60 {c:+.1f}")
    if not pd.isna(sma20) and sma20:
        dist20 = close / sma20 - 1
        c = 4.0 if 0 <= dist20 <= 0.05 else (1.0 if dist20 < 0 else -5.0)
        trend += c
        rt.append(f"dist_sma20 {dist20*100:.1f}% {c:+.1f}")
    if not pd.isna(adx):
        c = -2.0 if adx < 18 else (2.0 if adx < 30 else 5.0)
        trend += c
        rt.append(f"adx {adx:.1f} {c:+.1f}")
    if "OBV" in df.columns and n >= 2:
        c = 2.0 if float(df["OBV"].iloc[-1]) > float(df["OBV"].iloc[-2]) else -2.0
        trend += c
        rt.append(f"obv_trend {c:+.1f}")
    if fib and fib.get("direction") == "up" and "levels" in fib:
        lv = fib["levels"]
        lo, hi = min(lv["61.8%"], lv["38.2%"]), max(lv["61.8%"], lv["38.2%"])
        if lo <= close <= hi:
            trend += 2.0
            rt.append("fib_support_zone +2.0")

    momentum = 50.0
    is_breakout = candle_breakout_long(df)
    if not pd.isna(sma5) and sma5:
        above5 = close > sma5
        if above5:
            momentum += 4.0
            rm.append("above_sma5 +4")
        if gap5_pct >= 95:
            momentum -= 4.0
            rm.append("gap5_extreme -4")
        elif 0 < gap5_now <= 0.03:
            momentum += 3.0
            rm.append("gap5_reasonable +3")
    if rsi_now is not None:
        if 50 < rsi_now <= 65:
            momentum += 4.0
            rm.append("rsi_trending +4")
        elif rsi_now >= 80:
            momentum -= 5.0
            rm.append("rsi_overheat -5")
        elif 70 <= rsi_now < 80:
            momentum -= 2.0
            rm.append("rsi_hot -2")
    if "MACD_12_26_9" in df.columns and n >= 3:
        hist = float(df["MACD_12_26_9"].iloc[-1]) - float(df["MACDs_12_26_9"].iloc[-1])
        hist_prev = float(df["MACD_12_26_9"].iloc[-2]) - float(df["MACDs_12_26_9"].iloc[-2])
        if hist > 0 and hist_prev <= 0:
            momentum += 5.0
            rm.append("macd_fresh_cross +5")
        elif hist > hist_prev > 0:
            momentum += 3.0
            rm.append("macd_improving +3")
        elif hist < hist_prev and hist_prev > 0:
            momentum -= 3.0
            rm.append("macd_fading -3")
    if all(c in df.columns for c in ["BBL_20_2.0", "BBU_20_2.0"]) and n >= 2:
        bbu = float(last["BBU_20_2.0"])
        dist = (close - bbu) / (bbu or 1)
        if dist > 0.05:
            momentum -= 6.0
            rm.append("bb_overextended -6")
        elif 0.03 < dist <= 0.05:
            momentum -= 3.0
            rm.append("bb_hot -3")
    if ta20 < MIN_AVG_TURNOVER_20:
        momentum -= 12.0
        rm.append("avg_turnover20_below_3b -12")
    elif ta5 < SOFT_MIN_AVG_TURNOVER_5:
        momentum -= 4.0
        rm.append("avg_turnover5_below_5b -4")
    if vol_pct <= 20:
        momentum -= 6.0
        rm.append("volume_pct_low -6")
    elif signals.get("turnover_surge"):
        momentum += 3.0
        rm.append("turnover_surge +3")
    if is_breakout:
        momentum += 5.0
        rm.append("price_breakout +5")
    if candle_bull_engulfing(df):
        momentum += 3.0
        rm.append("bull_engulfing +3")

    rng = max(high - low, 1e-9)
    body = abs(close - open_)
    tail_up = (high - max(open_, close)) / rng
    tail_dn = (min(open_, close) - low) / rng
    if close > open_ and body / rng >= 0.6 and tail_up < 0.2:
        momentum += 3.0
        rm.append("strong_body +3")
    if tail_dn >= 0.4 and close >= open_:
        momentum += 2.0
        rm.append("lower_tail_recovery +2")
    if tail_up >= 0.4:
        momentum -= 4.0
        rm.append("upper_tail_supply -4")
    if not pd.isna(atr_val) and atr_val > 0 and close > 0:
        atr_pct = atr_val / close
        if abs(day_ret) > atr_pct * 2.5:
            momentum -= 5.0
            rm.append("day_move_too_large -5")
        elif 0.01 <= day_ret <= atr_pct * 1.5:
            momentum += 2.0
            rm.append("controlled_up_day +2")

    breakout = 0
    if is_breakout:
        breakout += 40
    if candle_bull_engulfing(df):
        breakout += 20
    if signals.get("bb_breakout_up"):
        breakout += 20
    if signals.get("macd_bull_cross"):
        breakout += 20
    breakout = min(100, breakout)

    overheat = 0
    if rsi_now and rsi_now >= 70:
        overheat += 30
    if rsi_now and rsi_now >= 80:
        overheat += 20
    if gap5_pct >= 95:
        overheat += 25
    if signals.get("bb_breakout_up"):
        overheat += 25
    overheat = min(100, overheat)

    liquidity_risk = 0
    if close < MIN_PRICE:
        liquidity_risk += 100
    if ta20 < MIN_AVG_TURNOVER_20:
        liquidity_risk += 60
    elif ta20 < 5_000_000_000:
        liquidity_risk += 25
    if ta5 < SOFT_MIN_AVG_TURNOVER_5:
        liquidity_risk += 20
    if vol_pct <= 20:
        liquidity_risk += 20
    liquidity_risk = min(100, liquidity_risk)

    trend = float(np.clip(trend, 0, 100))
    momentum = float(np.clip(momentum, 0, 100))
    score_raw = (trend + momentum) / 2
    score = int(round(score_raw))
    grade = "strong" if score >= 62 else ("watch_buy" if score >= 60 else ("hold_watch" if score >= 48 else "caution"))

    return {
        "score": score,
        "score_raw": score_raw,
        "trend_score": int(round(trend)),
        "trend_raw": trend,
        "momentum_score": int(round(momentum)),
        "momentum_raw": momentum,
        "breakout_score": breakout,
        "overheat_score": overheat,
        "liquidity_risk": liquidity_risk,
        "grade": grade,
        "reasons": rt + rm,
        "reasons_trend": rt,
        "reasons_momentum": rm,
    }


def investor_flow_score(code: str):
    def fetch(days):
        for attempt in range(2):
            try:
                end = datetime.now(KR_TZ).strftime("%Y%m%d")
                start = (datetime.now(KR_TZ) - timedelta(days=days)).strftime("%Y%m%d")
                df = stock.get_market_trading_value_by_investor(start, end, code)
                if df is None or df.empty:
                    return None
                col = df.columns[-1]
                individual = float(df.loc["개인", col]) if "개인" in df.index else 0.0
                institution = float(df.loc["기관합계", col]) if "기관합계" in df.index else 0.0
                foreigner = float(df.loc["외국인합계", col]) if "외국인합계" in df.index else 0.0
                return {
                    "individual": individual,
                    "institution": institution,
                    "foreigner": foreigner,
                    "base": float(df[col].abs().sum()) or 1.0,
                }
            except Exception:
                if attempt == 0:
                    time.sleep(random.uniform(0.3, 0.7))
        return None

    r5 = fetch(5)
    r20 = fetch(20)

    def score_block(r):
        if r is None:
            return None
        weighted = 0.30 * r["individual"] + 0.40 * r["institution"] + 0.30 * r["foreigner"]
        return int(max(0, min(100, 50 + 35 * np.tanh(weighted / r["base"]))))

    short = score_block(r5)
    mid = score_block(r20)
    if short is None and mid is None:
        return None, {}, {"short": None, "mid": None, "persist": None}
    short = short if short is not None else mid
    mid = mid if mid is not None else short

    if r5 is not None and r20 is not None:
        net5 = r5["foreigner"] + r5["institution"]
        net20 = r20["foreigner"] + r20["institution"]
        persist = 70 if (net5 > 0 and net20 > 0) else (30 if (net5 < 0 and net20 < 0) else 50)
    elif r5 is not None:
        net5 = r5["foreigner"] + r5["institution"]
        persist = 60 if net5 > 0 else (40 if net5 < 0 else 50)
    else:
        net20 = r20["foreigner"] + r20["institution"]
        persist = 60 if net20 > 0 else (40 if net20 < 0 else 50)

    final = int(0.40 * short + 0.35 * mid + 0.25 * persist)
    return final, r5 or r20 or {}, {"short": short, "mid": mid, "persist": persist}


def volume_score(df: pd.DataFrame, days: int = 20) -> int:
    try:
        if "TurnoverAmt" in df.columns:
            last_value = float(df["TurnoverAmt"].iloc[-1])
            avg_value = float(df["TurnoverAmt"].tail(days).mean())
        else:
            last_value = float(df["Volume"].iloc[-1])
            avg_value = float(df["Volume"].tail(days).mean())
        return int(max(0, min(100, 50 + 40 * np.tanh(last_value / avg_value - 1)))) if avg_value > 0 else 50
    except Exception:
        return 50


def compute_score(
    code: str,
    name: str,
    market: str,
    df: pd.DataFrame,
    lookback: int = 400,
    regime: str = "sideways",
    include_flow: bool = True,
) -> dict | None:
    try:
        if df is None or df.empty or len(df) < 60:
            return None
        df = add_indicators(df.tail(lookback))
        if len(df) < 60:
            return None
        close = float(df["Close"].iloc[-1])
        avg_turnover20 = float(df["TurnoverAmt"].tail(20).mean())
        avg_turnover5 = float(df["TurnoverAmt"].tail(5).mean())
        if close < MIN_PRICE or avg_turnover20 < MIN_AVG_TURNOVER_20:
            return None

        fib = find_fib_levels(df)
        sig = basic_signals(df)
        sc = score_technical(df, sig, fib)
        ichi = ichimoku_signals(df)
        ichi_sc = (
            (10 if ichi.get("tenkan_kijun") == "bull" else 0)
            + (10 if ichi.get("cloud") == "above" else 0)
            + (5 if ichi.get("chikou") == "bull_confirmed" else 0)
        )
        flow = None
        if include_flow:
            flow, _, _ = investor_flow_score(code)
        vol = volume_score(df)
        w = regime_weights(regime)

        if flow is None:
            fw = w["flow"]
            denom = 1.0 - fw
            fr = (
                (w["trend"] / denom) * sc["trend_raw"]
                + (w["momentum"] / denom) * sc["momentum_raw"]
                + (w["volume"] / denom) * vol
                + (w["ichi"] / denom) * ichi_sc
            )
        else:
            fr = (
                w["trend"] * sc["trend_raw"]
                + w["momentum"] * sc["momentum_raw"]
                + w["flow"] * flow
                + w["volume"] * vol
                + w["ichi"] * ichi_sc
            )

        fr = max(0.0, fr - sc["overheat_score"] * 0.12 - sc["liquidity_risk"] * 0.08)
        if avg_turnover5 < SOFT_MIN_AVG_TURNOVER_5:
            fr -= 2.0
        return {
            "Code": code,
            "Name": name,
            "Market": market,
            "Score": int(round(fr)),
            "ScoreRaw": float(fr),
            "Trend": sc["trend_score"],
            "Momentum": sc["momentum_score"],
            "TrendRaw": sc["trend_raw"],
            "MomentumRaw": sc["momentum_raw"],
            "Breakout": sc["breakout_score"],
            "Overheat": sc["overheat_score"],
            "LiqRisk": sc["liquidity_risk"],
            "Flow": flow,
            "Vol": vol,
            "IchiSc": ichi_sc,
            "Close": close,
            "AvgTurnover20": avg_turnover20,
            "AvgTurnover5": avg_turnover5,
        }
    except Exception:
        return None
