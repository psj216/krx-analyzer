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


def _grade_label(score: float) -> str:
    if score >= 72:
        return "매수 유력"
    if score >= 60:
        return "매수 관심"
    if score >= 48:
        return "관망 보유"
    return "주의"


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
        "grade": "데이터 부족",
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
    reasons_trend, reasons_momentum = [], []

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
    gap5_now = (close / sma5 - 1) if not pd.isna(sma5) and sma5 else 0.0
    gap5_pct = pctrank(gap5_series, gap5_now, 120) if len(gap5_series) >= 10 else 50.0
    day_ret = (close / float(df["Close"].iloc[-2]) - 1) if n >= 2 else 0.0

    trend = 50.0
    if not pd.isna(sma120):
        delta = 6.0 if close > sma120 else -6.0
        trend += delta
        reasons_trend.append(f"SMA120 기준으로 {'상방 유지' if delta > 0 else '하방 위치'} ({delta:+.0f})")
    if not pd.isna(sma20) and not pd.isna(sma60):
        delta = 5.0 if sma20 > sma60 else -5.0
        trend += delta
        reasons_trend.append(f"SMA20과 SMA60 배열이 {'정배열' if delta > 0 else '역배열'} ({delta:+.0f})")
    if not pd.isna(sma20) and sma20:
        dist20 = close / sma20 - 1
        delta = 4.0 if 0 <= dist20 <= 0.05 else (1.0 if dist20 < 0 else -5.0)
        trend += delta
        reasons_trend.append(f"현재가와 SMA20 괴리 {dist20*100:.1f}% ({delta:+.0f})")
    if not pd.isna(adx):
        delta = -2.0 if adx < 18 else (2.0 if adx < 30 else 5.0)
        trend += delta
        reasons_trend.append(f"ADX {adx:.1f}로 추세 강도 {'양호' if delta > 0 else '약함'} ({delta:+.0f})")
    if "OBV" in df.columns and n >= 2:
        delta = 2.0 if float(df["OBV"].iloc[-1]) > float(df["OBV"].iloc[-2]) else -2.0
        trend += delta
        reasons_trend.append(f"OBV 흐름이 {'개선' if delta > 0 else '둔화'} ({delta:+.0f})")
    if fib and fib.get("direction") == "up" and "levels" in fib:
        levels = fib["levels"]
        lo, hi = min(levels["61.8%"], levels["38.2%"]), max(levels["61.8%"], levels["38.2%"])
        if lo <= close <= hi:
            trend += 2.0
            reasons_trend.append("피보나치 38.2~61.8% 지지 구간 안에 위치 (+2)")

    momentum = 50.0
    is_breakout = candle_breakout_long(df)
    if not pd.isna(sma5) and sma5:
        if close > sma5:
            momentum += 4.0
            reasons_momentum.append("현재가가 SMA5 위에서 단기 흐름 유지 (+4)")
        if gap5_pct >= 95:
            momentum -= 4.0
            reasons_momentum.append("SMA5 대비 단기 괴리가 과도함 (-4)")
        elif 0 < gap5_now <= 0.03:
            momentum += 3.0
            reasons_momentum.append("SMA5 대비 괴리가 적당해 추격 부담이 낮음 (+3)")
    if rsi_now is not None:
        if 50 < rsi_now <= 65:
            momentum += 4.0
            reasons_momentum.append(f"RSI {rsi_now:.1f}로 상승 모멘텀이 무난함 (+4)")
        elif rsi_now >= 80:
            momentum -= 5.0
            reasons_momentum.append(f"RSI {rsi_now:.1f}로 과열 부담이 큼 (-5)")
        elif 70 <= rsi_now < 80:
            momentum -= 2.0
            reasons_momentum.append(f"RSI {rsi_now:.1f}로 단기 과열 구간 진입 (-2)")
    if "MACD_12_26_9" in df.columns and n >= 3:
        hist = float(df["MACD_12_26_9"].iloc[-1]) - float(df["MACDs_12_26_9"].iloc[-1])
        hist_prev = float(df["MACD_12_26_9"].iloc[-2]) - float(df["MACDs_12_26_9"].iloc[-2])
        if hist > 0 and hist_prev <= 0:
            momentum += 5.0
            reasons_momentum.append("MACD가 막 상향 전환됨 (+5)")
        elif hist > hist_prev > 0:
            momentum += 3.0
            reasons_momentum.append("MACD 양수 구간에서 모멘텀이 개선 중 (+3)")
        elif hist < hist_prev and hist_prev > 0:
            momentum -= 3.0
            reasons_momentum.append("MACD 상승 탄력이 둔화됨 (-3)")
    if all(c in df.columns for c in ["BBL_20_2.0", "BBU_20_2.0"]) and n >= 2:
        bbu = float(last["BBU_20_2.0"])
        dist = (close - bbu) / (bbu or 1.0)
        if dist > 0.05:
            momentum -= 6.0
            reasons_momentum.append("볼린저 상단을 크게 이탈해 과열 부담이 큼 (-6)")
        elif 0.03 < dist <= 0.05:
            momentum -= 3.0
            reasons_momentum.append("볼린저 상단 근처라 단기 과열 부담이 있음 (-3)")
    if ta20 < MIN_AVG_TURNOVER_20:
        momentum -= 12.0
        reasons_momentum.append("20일 평균 거래대금이 30억 미만으로 약함 (-12)")
    elif ta5 < SOFT_MIN_AVG_TURNOVER_5:
        momentum -= 4.0
        reasons_momentum.append("최근 5일 평균 거래대금이 50억 미만이라 탄력이 약함 (-4)")
    if vol_pct <= 20:
        momentum -= 6.0
        reasons_momentum.append("최근 거래량이 자기 분포 하위권이라 힘이 약함 (-6)")
    elif signals.get("turnover_surge"):
        momentum += 3.0
        reasons_momentum.append("거래대금이 평소보다 크게 붙음 (+3)")
    if is_breakout:
        momentum += 5.0
        reasons_momentum.append("최근 고점을 돌파하는 장대 양봉 패턴 (+5)")
    if candle_bull_engulfing(df):
        momentum += 3.0
        reasons_momentum.append("강한 양봉 감싸기 패턴이 나옴 (+3)")

    rng = max(high - low, 1e-9)
    body = abs(close - open_)
    tail_up = (high - max(open_, close)) / rng
    tail_dn = (min(open_, close) - low) / rng
    if close > open_ and body / rng >= 0.6 and tail_up < 0.2:
        momentum += 3.0
        reasons_momentum.append("몸통이 큰 양봉으로 마감 강도가 좋음 (+3)")
    if tail_dn >= 0.4 and close >= open_:
        momentum += 2.0
        reasons_momentum.append("아래꼬리 반등이 나와 저가 매수 유입이 보임 (+2)")
    if tail_up >= 0.4:
        momentum -= 4.0
        reasons_momentum.append("윗꼬리가 길어 매도 압력이 확인됨 (-4)")
    if not pd.isna(atr_val) and atr_val > 0 and close > 0:
        atr_pct = atr_val / close
        if abs(day_ret) > atr_pct * 2.5:
            momentum -= 5.0
            reasons_momentum.append("당일 변동이 ATR 대비 과도해 추격 위험이 큼 (-5)")
        elif 0.01 <= day_ret <= atr_pct * 1.5:
            momentum += 2.0
            reasons_momentum.append("당일 상승이 과하지 않고 안정적임 (+2)")

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
    score_raw = (trend + momentum) / 2.0

    return {
        "score": int(round(score_raw)),
        "score_raw": score_raw,
        "trend_score": int(round(trend)),
        "trend_raw": trend,
        "momentum_score": int(round(momentum)),
        "momentum_raw": momentum,
        "breakout_score": breakout,
        "overheat_score": overheat,
        "liquidity_risk": liquidity_risk,
        "grade": _grade_label(score_raw),
        "reasons": reasons_trend + reasons_momentum,
        "reasons_trend": reasons_trend,
        "reasons_momentum": reasons_momentum,
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
            return 50
        weighted = 0.30 * r["individual"] + 0.40 * r["institution"] + 0.30 * r["foreigner"]
        return int(max(0, min(100, 50 + 35 * np.tanh(weighted / r["base"]))))

    short = score_block(r5)
    mid = score_block(r20)

    if r5 is not None and r20 is not None:
        net5 = r5["foreigner"] + r5["institution"]
        net20 = r20["foreigner"] + r20["institution"]
        persist = 70 if (net5 > 0 and net20 > 0) else (30 if (net5 < 0 and net20 < 0) else 50)
        status = "실데이터"
    elif r5 is not None:
        net5 = r5["foreigner"] + r5["institution"]
        persist = 60 if net5 > 0 else (40 if net5 < 0 else 50)
        status = "부분데이터"
    elif r20 is not None:
        net20 = r20["foreigner"] + r20["institution"]
        persist = 60 if net20 > 0 else (40 if net20 < 0 else 50)
        status = "부분데이터"
    else:
        persist = 50
        status = "수급실패→기본50"

    final = int(round(0.40 * short + 0.35 * mid + 0.25 * persist))
    detail = {
        "short": short,
        "mid": mid,
        "persist": persist,
        "status": status,
    }
    return final, (r5 or r20 or {}), detail


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

        df = add_indicators(df.tail(lookback).copy())
        if len(df) < 60:
            return None

        close = float(df["Close"].iloc[-1])
        avg_turnover20 = float(df["TurnoverAmt"].tail(20).mean())
        avg_turnover5 = float(df["TurnoverAmt"].tail(5).mean())
        if close < MIN_PRICE or avg_turnover20 < MIN_AVG_TURNOVER_20:
            return None

        fib = find_fib_levels(df)
        signals = basic_signals(df)
        tech = score_technical(df, signals, fib)
        ichi = ichimoku_signals(df)
        ichi_score = (
            (10 if ichi.get("전환-기준") == "강세" else 0)
            + (10 if ichi.get("구름") == "상승구간" else 0)
            + (5 if ichi.get("후행스팬") == "상승 확인" else 0)
        )

        flow_score = 50
        flow_detail = {"short": 50, "mid": 50, "persist": 50, "status": "수급 미사용"}
        if include_flow:
            flow_score, _, flow_detail = investor_flow_score(code)

        volume = volume_score(df)
        weights = regime_weights(regime)
        final_raw = (
            weights["trend"] * tech["trend_raw"]
            + weights["momentum"] * tech["momentum_raw"]
            + weights["flow"] * flow_score
            + weights["volume"] * volume
            + weights["ichi"] * ichi_score
        )
        final_raw = max(0.0, final_raw - tech["overheat_score"] * 0.12 - tech["liquidity_risk"] * 0.08)
        final_score = int(round(final_raw))

        return {
            "Code": code,
            "Name": name,
            "Market": market,
            "Score": final_score,
            "ScoreRaw": float(final_raw),
            "Grade": _grade_label(final_raw),
            "Trend": tech["trend_score"],
            "Momentum": tech["momentum_score"],
            "TrendRaw": tech["trend_raw"],
            "MomentumRaw": tech["momentum_raw"],
            "Breakout": tech["breakout_score"],
            "Overheat": tech["overheat_score"],
            "LiqRisk": tech["liquidity_risk"],
            "Flow": flow_score,
            "FlowDetail": flow_detail,
            "Vol": volume,
            "IchiSc": ichi_score,
            "Close": close,
            "AvgTurnover20": avg_turnover20,
            "AvgTurnover5": avg_turnover5,
            "Signals": signals,
            "Fib": fib,
            "Technical": tech,
        }
    except Exception:
        return None
