"""
core/scorer.py
Six-factor score model used by app and Telegram.
"""
from datetime import datetime, timedelta
import random
import time

import numpy as np
import pandas as pd
from pykrx import stock

from core.indicators import add_indicators
from core.signals import basic_signals, candle_bull_engulfing, candle_breakout_long, ichimoku_signals
from core.utils import KR_TZ, find_fib_levels

MIN_PRICE = 1_000
MIN_AVG_TURNOVER_20 = 3_000_000_000
SOFT_MIN_AVG_TURNOVER_5 = 5_000_000_000

BASE_WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.20,
    "breakout": 0.15,
    "flow": 0.25,
    "volume": 0.05,
    "ichi": 0.05,
}


def _default_technical() -> dict:
    return {
        "score": 50,
        "score_raw": 50.0,
        "trend_score": 50,
        "trend_raw": 50.0,
        "momentum_score": 50,
        "momentum_raw": 50.0,
        "breakout_score": 0,
        "breakout_raw": 0.0,
        "overheat_score": 0,
        "liquidity_risk": 0,
        "grade": _grade_label(50),
        "reasons_trend": [],
        "reasons_momentum": [],
        "reasons_breakout": [],
    }


def _as_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _grade_label(score: float, overheat: float = 0.0) -> str:
    if score >= 72:
        label = "매수 유망"
    elif score >= 60:
        label = "매수 관심"
    elif score >= 48:
        label = "관망/보유"
    else:
        label = "주의"
    return f"{label} / 추격주의" if overheat >= 70 else label


def score_technical(df: pd.DataFrame, signals: dict, fib: dict) -> dict:
    empty = {
        "score": 0,
        "score_raw": 0.0,
        "trend_score": 0,
        "trend_raw": 0.0,
        "momentum_score": 0,
        "momentum_raw": 0.0,
        "breakout_score": 0,
        "breakout_raw": 0.0,
        "overheat_score": 0,
        "liquidity_risk": 0,
        "grade": "데이터 부족",
        "reasons_trend": [],
        "reasons_momentum": [],
        "reasons_breakout": [],
    }
    if df is None or df.empty or not signals:
        return empty

    last = df.iloc[-1]
    close = float(last["Close"])
    open_ = float(last.get("Open", close))
    high = float(last.get("High", close))
    low = float(last.get("Low", close))
    n = len(df)

    def pctrank(series, val, window=120):
        sub = series.dropna().tail(window)
        return float((sub < val).mean() * 100) if len(sub) >= 10 else 50.0

    reasons_trend = []
    reasons_momentum = []
    reasons_breakout = []

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
    day_ret = (close / float(df["Close"].iloc[-2]) - 1) if n >= 2 else 0.0

    trend = 50.0
    if not pd.isna(sma120):
        delta = 6.0 if close > sma120 else -6.0
        trend += delta
        reasons_trend.append(
            f"현재가가 SMA120 {'위에 있어 중장기 추세 기준 상방' if delta > 0 else '아래에 있어 중장기 추세 기준 약세'} → {delta:+.1f}"
        )
    if not pd.isna(sma5) and not pd.isna(sma20) and not pd.isna(sma60) and not pd.isna(sma120):
        if close > sma5 > sma20 > sma60 > sma120:
            trend += 6.0
            reasons_trend.append("이동평균선이 5·20·60·120일선 순으로 정배열이라 추세 정합성이 좋음 → +6.0")
        elif close < sma5 < sma20 < sma60 < sma120:
            trend -= 6.0
            reasons_trend.append("이동평균선이 역배열이라 추세 정합성이 약함 → -6.0")
    if n >= 2 and not pd.isna(sma20):
        prev_sma20 = float(df["SMA20"].iloc[-2])
        if prev_sma20:
            slope20 = (sma20 - prev_sma20) / prev_sma20
            delta = 3.0 if slope20 > 0 else -2.0
            trend += delta
            reasons_trend.append(
                f"SMA20 기울기가 {slope20*100:.2f}%로 {'단기 추세가 우상향' if delta > 0 else '단기 추세가 둔화'} → {delta:+.1f}"
            )
    if n >= 2 and not pd.isna(sma60):
        prev_sma60 = float(df["SMA60"].iloc[-2])
        if prev_sma60:
            slope60 = (sma60 - prev_sma60) / prev_sma60
            delta = 3.0 if slope60 > 0 else -2.0
            trend += delta
            reasons_trend.append(
                f"SMA60 기울기가 {slope60*100:.2f}%로 {'중기 추세가 우상향' if delta > 0 else '중기 추세가 둔화'} → {delta:+.1f}"
            )
    if "OBV" in df.columns and n >= 2:
        delta = 2.0 if float(last["OBV"]) > float(df["OBV"].iloc[-2]) else -2.0
        trend += delta
        reasons_trend.append(
            f"OBV가 {'상승해 거래량 흐름이 긍정' if delta > 0 else '하락해 거래량 흐름이 약화'} → {delta:+.1f}"
        )
    if not pd.isna(adx):
        if adx >= 20:
            delta = 2.0 if adx < 35 else 3.0
            trend += delta
        else:
            delta = -2.0
            trend += delta
        reasons_trend.append(f"ADX {adx:.1f}로 {'추세 강도 양호' if delta > 0 else '추세 강도 부족'} → {delta:+.1f}")
    if "IchiSpanA" in df.columns and "IchiSpanB" in df.columns:
        span_a = float(last["IchiSpanA"])
        span_b = float(last["IchiSpanB"])
        if not pd.isna(span_a) and not pd.isna(span_b):
            upper = max(span_a, span_b)
            lower = min(span_a, span_b)
            if close > upper:
                delta = 3.0
                reasons_trend.append("현재가가 일목 구름 위에 있어 추세 구조가 안정적 → +3.0")
            elif close < lower:
                delta = -3.0
                reasons_trend.append("현재가가 일목 구름 아래에 있어 추세 구조가 약함 → -3.0")
            else:
                delta = 0.0
                reasons_trend.append("현재가가 일목 구름 내부에 있어 추세 판단이 중립적 → +0.0")
            trend += delta
    if fib and "levels" in fib and fib.get("direction") == "up":
        levels = fib["levels"]
        lo = min(levels["61.8%"], levels["38.2%"])
        hi = max(levels["61.8%"], levels["38.2%"])
        if lo <= close <= hi:
            trend += 2.0
            reasons_trend.append("피보나치 38.2~61.8% 구간에 있어 눌림목 위치가 양호 → +2.0")

    momentum = 50.0
    gap5_series = (df["Close"] / df["SMA5"] - 1).dropna() if "SMA5" in df.columns else pd.Series(dtype=float)
    gap5_now = (close / sma5 - 1) if not pd.isna(sma5) and sma5 else 0.0
    gap5_pct = pctrank(gap5_series, gap5_now, 120) if len(gap5_series) >= 10 else 50.0
    if not pd.isna(sma5) and sma5:
        if close > sma5:
            momentum += 4.0
            reasons_momentum.append("현재가가 5일선 위에 있어 단기 모멘텀이 양호 → +4.0")
        else:
            momentum -= 3.0
            reasons_momentum.append("현재가가 5일선 아래에 있어 단기 모멘텀이 약함 → -3.0")
        if n >= 2:
            prev_sma5 = float(df["SMA5"].iloc[-2])
            slope5 = (sma5 - prev_sma5) / prev_sma5 if prev_sma5 else 0.0
            if slope5 > 0:
                momentum += 3.0
                reasons_momentum.append("5일선이 우상향하고 있어 단기 상승 흐름이 유지됨 → +3.0")
            else:
                momentum -= 2.0
                reasons_momentum.append("5일선이 꺾이며 단기 탄력이 둔화 → -2.0")
        if gap5_pct >= 95:
            momentum -= 3.0
            reasons_momentum.append("5일선 대비 이격이 매우 커 단기 과열 부담이 큼 → -3.0")
    if rsi_now is not None:
        if 50 <= rsi_now <= 65:
            momentum += 4.0
            reasons_momentum.append(f"RSI {rsi_now:.1f}로 상승 모멘텀 구간이 유지됨 → +4.0")
        elif 65 < rsi_now < 75:
            momentum += 1.0
            reasons_momentum.append(f"RSI {rsi_now:.1f}로 힘은 있으나 과열 경계가 필요 → +1.0")
        elif rsi_now >= 75:
            momentum -= 4.0
            reasons_momentum.append(f"RSI {rsi_now:.1f}로 과열 부담이 큼 → -4.0")
        elif rsi_now < 35:
            momentum -= 3.0
            reasons_momentum.append(f"RSI {rsi_now:.1f}로 약세/과매도 구간 → -3.0")
    if n >= 3 and "RSI14" in df.columns and rsi_now is not None:
        rs1 = float(df["RSI14"].iloc[-1]) - float(df["RSI14"].iloc[-2])
        rs2 = float(df["RSI14"].iloc[-2]) - float(df["RSI14"].iloc[-3])
        if rs1 > 0 and rs2 > 0:
            momentum += 2.0
            reasons_momentum.append("RSI가 연속 상승해 모멘텀 강화가 확인됨 → +2.0")
        elif rs1 < 0 and rs2 < 0:
            momentum -= 2.0
            reasons_momentum.append("RSI가 연속 둔화돼 모멘텀 약화가 확인됨 → -2.0")
    if "MACD_12_26_9" in df.columns and n >= 3:
        macd_val = float(last["MACD_12_26_9"])
        signal_val = float(last["MACDs_12_26_9"])
        hist = macd_val - signal_val
        hist_prev = float(df["MACD_12_26_9"].iloc[-2]) - float(df["MACDs_12_26_9"].iloc[-2])
        if hist > 0 and hist_prev <= 0:
            momentum += 4.0
            reasons_momentum.append("MACD가 양전 초입으로 전환되며 상승 모멘텀이 강화 → +4.0")
        elif hist > hist_prev:
            momentum += 3.0
            reasons_momentum.append("MACD 흐름이 개선되며 상승 모멘텀이 강화 → +3.0")
        elif hist < hist_prev:
            momentum -= 3.0
            reasons_momentum.append("MACD 히스토그램이 둔화돼 상승 탄력이 약해짐 → -3.0")

    breakout = 35.0
    is_breakout = candle_breakout_long(df)
    rng = max(high - low, 1e-9)
    body = abs(close - open_)
    tail_up = (high - max(open_, close)) / rng
    if close > open_ and body / rng >= 0.6 and tail_up < 0.2:
        delta = 8.0
        breakout += delta
        reasons_breakout.append("몸통이 강한 양봉으로 매수세 우위가 확인됨 → +8.0")
    if candle_bull_engulfing(df):
        breakout += 7.0
        reasons_breakout.append("상승 장악형 캔들이 나와 반전/돌파 신호가 강화됨 → +7.0")
    if is_breakout:
        breakout += 10.0
        reasons_breakout.append("최근 고점을 돌파해 단기 돌파 신호가 발생 → +10.0")
    if signals.get("bb_breakout_up"):
        breakout += 6.0
        reasons_breakout.append("볼린저밴드 상단을 돌파해 강한 돌파 신호가 확인됨 → +6.0")
    elif "BBU_20_2.0" in df.columns:
        bbu = float(last["BBU_20_2.0"])
        if bbu > 0 and close >= bbu * 0.985:
            breakout += 3.0
            reasons_breakout.append("볼린저밴드 상단에 근접해 돌파 시도 흐름이 보임 → +3.0")
    if signals.get("macd_bull_cross"):
        breakout += 5.0
        reasons_breakout.append("MACD 골든크로스가 발생해 돌파 동력이 추가됨 → +5.0")
    elif "MACD_12_26_9" in df.columns and "MACDs_12_26_9" in df.columns:
        hist = float(last["MACD_12_26_9"]) - float(last["MACDs_12_26_9"])
        if hist > 0:
            breakout += 3.0
            reasons_breakout.append("MACD가 양전 초입에 있어 돌파 확률을 높여줌 → +3.0")
    if "Volume" in df.columns:
        vol20 = float(df["Volume"].tail(20).mean())
        ratio = vol_now / vol20 if vol20 > 0 else 1.0
        if ratio >= 1.5 and (is_breakout or body / rng >= 0.6):
            breakout += 6.0
            reasons_breakout.append(f"오늘 거래량이 20일 평균 대비 {ratio:.2f}배로 돌파형 거래량이 동반됨 → +6.0")
        elif ratio >= 1.2:
            breakout += 2.0
            reasons_breakout.append(f"오늘 거래량이 20일 평균 대비 {ratio:.2f}배로 늘어 돌파 동력이 보강됨 → +2.0")
    if 0.01 <= day_ret <= 0.03:
        breakout += 2.0
        reasons_breakout.append("과도하지 않은 상승폭으로 안정적 상승 흐름이 유지됨 → +2.0")
    elif day_ret > 0.06:
        breakout -= 3.0
        reasons_breakout.append("급등폭이 과도해 돌파 지속성에는 부담이 있음 → -3.0")

    overheat = 0
    if rsi_now is not None and rsi_now >= 70:
        overheat += 30
    if rsi_now is not None and rsi_now >= 80:
        overheat += 20
    if gap5_pct >= 95:
        overheat += 20
    if signals.get("bb_breakout_up"):
        overheat += 20
    if tail_up >= 0.4:
        overheat += 15
    overheat = min(100, overheat)

    liquidity_risk = 0
    if vol_now < 10_000:
        liquidity_risk += 50
    elif vol_now < 30_000:
        liquidity_risk += 25
    if close < MIN_PRICE:
        liquidity_risk += 20
    if ta20 < MIN_AVG_TURNOVER_20:
        liquidity_risk += 20
    if ta5 < SOFT_MIN_AVG_TURNOVER_5:
        liquidity_risk += 10
    liquidity_risk = min(100, liquidity_risk)

    trend = float(np.clip(trend, 0, 100))
    momentum = float(np.clip(momentum, 0, 100))
    breakout = float(np.clip(breakout, 0, 100))

    return {
        "score": int(round((trend + momentum + breakout) / 3.0)),
        "score_raw": float((trend + momentum + breakout) / 3.0),
        "trend_score": int(round(trend)),
        "trend_raw": trend,
        "momentum_score": int(round(momentum)),
        "momentum_raw": momentum,
        "breakout_score": int(round(breakout)),
        "breakout_raw": breakout,
        "overheat_score": overheat,
        "liquidity_risk": liquidity_risk,
        "grade": _grade_label((trend + momentum + breakout) / 3.0),
        "reasons_trend": reasons_trend,
        "reasons_momentum": reasons_momentum,
        "reasons_breakout": reasons_breakout,
    }


def score_technical_practical(df: pd.DataFrame, signals: dict, fib: dict) -> dict:
    empty = {
        "score": 0,
        "score_raw": 0.0,
        "trend_score": 0,
        "trend_raw": 0.0,
        "momentum_score": 0,
        "momentum_raw": 0.0,
        "breakout_score": 0,
        "breakout_raw": 0.0,
        "overheat_score": 0,
        "liquidity_risk": 0,
        "grade": "데이터 부족",
        "reasons_trend": [],
        "reasons_momentum": [],
        "reasons_breakout": [],
        "overheat_reasons": [],
        "positive_factors": [],
        "risk_factors": [],
    }
    if df is None or df.empty or len(df) < 60:
        return empty

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    close = _as_float(last.get("Close"), 0.0)
    open_ = _as_float(last.get("Open"), close)
    high = _as_float(last.get("High"), close)
    low = _as_float(last.get("Low"), close)
    vol_now = _as_float(last.get("Volume"), 0.0)
    rng = max(high - low, 1e-9)
    body = abs(close - open_)
    upper_tail = max(high - max(open_, close), 0.0) / rng
    lower_tail = max(min(open_, close) - low, 0.0) / rng
    day_ret = close / _as_float(prev.get("Close"), close) - 1 if _as_float(prev.get("Close"), 0.0) else 0.0
    atr = _as_float(last.get("ATR14"), 0.0)
    day_range_atr = rng / atr if atr > 0 else 0.0
    prior_gain_atr = 0.0
    if len(df) >= 3 and atr > 0:
        p1 = _as_float(df["Close"].iloc[-2], 0.0)
        p2 = _as_float(df["Close"].iloc[-3], 0.0)
        prior_gain_atr = max(p1 - p2, 0.0) / atr

    def pctrank(series, val, window=120):
        sub = series.dropna().tail(window)
        return float((sub <= val).mean() * 100) if len(sub) >= 10 else 50.0

    def slope(col, periods=1):
        if col not in df.columns or len(df) <= periods:
            return 0.0
        now = _as_float(df[col].iloc[-1], 0.0)
        old = _as_float(df[col].iloc[-1 - periods], 0.0)
        return (now / old - 1.0) if old else 0.0

    sma5 = _as_float(last.get("SMA5"), np.nan)
    sma20 = _as_float(last.get("SMA20"), np.nan)
    sma60 = _as_float(last.get("SMA60"), np.nan)
    sma120 = _as_float(last.get("SMA120"), np.nan)
    rsi = _as_float(last.get("RSI14"), np.nan)
    adx = _as_float(last.get("ADX14"), np.nan)
    vol20 = _as_float(df["Volume"].tail(20).mean(), 0.0) if "Volume" in df.columns else 0.0
    vol_ratio = vol_now / vol20 if vol20 > 0 else 1.0
    gap5 = close / sma5 - 1 if not pd.isna(sma5) and sma5 else 0.0
    gap20 = close / sma20 - 1 if not pd.isna(sma20) and sma20 else 0.0
    gap5_series = (df["Close"] / df["SMA5"] - 1).replace([np.inf, -np.inf], np.nan) if "SMA5" in df.columns else pd.Series(dtype=float)
    gap5_pct = pctrank(gap5_series, gap5, 120)

    reasons_trend, reasons_momentum, reasons_breakout = [], [], []
    positives, risks, overheat_reasons = [], [], []

    trend = 50.0
    if not pd.isna(sma120) and close > sma120:
        trend += 4
        reasons_trend.append("SMA120 상방 → +4")
        positives.append("SMA120 상방")
    s20 = slope("SMA20")
    if s20 > 0:
        pts = 3 if s20 >= 0.01 else (2 if s20 >= 0.003 else 1)
        trend += pts
        reasons_trend.append(f"SMA20 기울기 {s20*100:.2f}% → +{pts}")
    s60 = slope("SMA60")
    if s60 > 0:
        pts = 2 if s60 >= 0.003 else 1
        trend += pts
        reasons_trend.append(f"SMA60 기울기 {s60*100:.2f}% → +{pts}")
    if not pd.isna(sma20) and close > sma20:
        trend += 2
        reasons_trend.append("볼린저 중단선 상방 → +2")
    ma_bonus = 0
    if not pd.isna(sma20) and not pd.isna(sma60) and sma20 > sma60:
        ma_bonus += 3
        reasons_trend.append("20/60 정배열 → +3")
        positives.append("20/60 정배열")
    if all(not pd.isna(x) for x in [sma5, sma20, sma60, sma120]) and sma5 > sma20 > sma60 > sma120:
        add = min(3, 6 - ma_bonus)
        if add > 0:
            ma_bonus += add
            reasons_trend.append(f"완전 정배열 추가 → +{add}")
            positives.append("완전 정배열")
    trend += min(ma_bonus, 6)
    if "OBV" in df.columns and _as_float(last.get("OBV"), 0.0) > _as_float(prev.get("OBV"), 0.0):
        trend += 2
        reasons_trend.append("OBV 상승 → +2")
        positives.append("OBV 상승")
    if not pd.isna(adx):
        if adx >= 30:
            trend += 6
            reasons_trend.append(f"ADX {adx:.1f} → +6")
            positives.append("ADX 30 이상")
        elif adx >= 20:
            trend += 3
            reasons_trend.append(f"ADX {adx:.1f} → +3")
    span_a = _as_float(last.get("IchiSpanA"), np.nan)
    span_b = _as_float(last.get("IchiSpanB"), np.nan)
    if not pd.isna(span_a) and not pd.isna(span_b) and close > max(span_a, span_b):
        cloud_pts = 3 if span_a > span_b else 1
        trend += cloud_pts
        reasons_trend.append(f"일목 구름 상방 → +{cloud_pts}")

    momentum = 50.0
    above5 = not pd.isna(sma5) and sma5 and close > sma5
    up5 = slope("SMA5") > 0
    if above5:
        momentum += 4
        reasons_momentum.append("5일선 상방 → +4")
    if up5:
        momentum += 3
        reasons_momentum.append("5일선 우상향 → +3")
    if above5 and up5:
        momentum += 2
        reasons_momentum.append("5일선 상방+우상향 → +2")
    if gap5_pct >= 95:
        momentum -= 4
        reasons_momentum.append("5일선 이격 상위5% → -4")
        risks.append("5일선 이격 상위5%")
    elif gap5_pct >= 90:
        momentum -= 2
        reasons_momentum.append("5일선 이격 상위10% → -2")
        risks.append("5일선 이격 상위10%")
    if not pd.isna(rsi):
        if 55 <= rsi <= 65:
            momentum += 3
            reasons_momentum.append("RSI 55~65 → +3")
        elif 70 <= rsi < 75:
            momentum -= 1
            reasons_momentum.append("RSI 70~75 → -1")
            risks.append("RSI 70 이상")
        elif rsi >= 75:
            risks.append("RSI 75 이상")
    if "MACD_12_26_9" in df.columns and "MACDs_12_26_9" in df.columns and len(df) >= 3:
        hist = _as_float(last.get("MACD_12_26_9"), 0.0) - _as_float(last.get("MACDs_12_26_9"), 0.0)
        hist_prev = _as_float(df["MACD_12_26_9"].iloc[-2], 0.0) - _as_float(df["MACDs_12_26_9"].iloc[-2], 0.0)
        hist_prev2 = _as_float(df["MACD_12_26_9"].iloc[-3], 0.0) - _as_float(df["MACDs_12_26_9"].iloc[-3], 0.0)
        if hist > hist_prev and (hist - hist_prev) > (hist_prev - hist_prev2):
            momentum += 5
            reasons_momentum.append("MACD 증가+가속 → +5")
            positives.append("MACD 증가+가속")
        elif hist > hist_prev or (hist > 0 and hist_prev <= 0):
            momentum += 3
            reasons_momentum.append("MACD 개선 → +3")

    breakout = 50.0
    if 1.2 <= vol_ratio < 2.5:
        breakout += 3
        reasons_breakout.append("거래량 1.2~2.5배 → +3")
        positives.append("거래량 1.2~2.5배")
    elif 2.5 <= vol_ratio < 5:
        breakout += 1
        reasons_breakout.append("거래량 2.5~5배 → +1")
    elif 5 <= vol_ratio < 8:
        breakout -= 2
        reasons_breakout.append("거래량 5~8배 → -2")
        risks.append("거래량 5배 이상")
    elif vol_ratio >= 8:
        risks.append("거래량 8배 이상")
    long_bull = close > open_ and body / rng >= 0.55
    if long_bull and vol_ratio >= 1.2:
        breakout += 5
        reasons_breakout.append("장대양봉+거래량 → +5")
    if candle_bull_engulfing(df) and vol_ratio >= 1.2:
        breakout += 4
        reasons_breakout.append("상승 장악형+거래량 → +4")
    if lower_tail >= 0.35 and close >= open_:
        breakout += 3
        reasons_breakout.append("아래꼬리 반등형 → +3")
    if signals.get("turnover_surge"):
        pts = 3 if upper_tail < 0.25 else 1
        breakout += pts
        reasons_breakout.append(f"거래대금 급등 보조 → +{pts}")
    if upper_tail >= 0.40:
        breakout -= 4
        reasons_breakout.append("윗꼬리 40%↑ → -4")
        risks.append("윗꼬리 40% 이상")
    if prior_gain_atr > 1.0:
        breakout -= 3
        reasons_breakout.append("전일 급등 ATR초과 → -3")
        risks.append("전일 급등 ATR 초과")
    else:
        breakout += 2
        reasons_breakout.append("전일 상승폭 적정 → +2")
    if day_range_atr > 2.0:
        breakout -= 5
        reasons_breakout.append("ATR 변동 과도 → -5")
        risks.append("ATR 대비 변동폭 과도")
    if len(df) >= 21 and close > _as_float(df["High"].iloc[-21:-1].max(), close * 2):
        breakout += 2
        reasons_breakout.append("최근 고점 돌파 참고 → +2")
    if signals.get("bb_breakout_up"):
        reasons_breakout.append("볼린저 상단 돌파 참고")

    overheat = 0
    if not pd.isna(rsi):
        if rsi >= 75:
            overheat += 25
            overheat_reasons.append("RSI 75↑")
        elif rsi >= 70:
            overheat += 10
            overheat_reasons.append("RSI 70~75")
    if gap5_pct >= 95:
        overheat += 25
        overheat_reasons.append("5일선 이격 상위5%")
    elif gap5_pct >= 90:
        overheat += 15
        overheat_reasons.append("5일선 이격 상위10%")
    if gap20 >= 0.12:
        overheat += 20
        overheat_reasons.append("20일선 이격 12%↑")
        risks.append("20일선 이격 12% 이상")
    elif gap20 >= 0.08:
        overheat += 10
        overheat_reasons.append("20일선 이격 8%↑")
        risks.append("20일선 이격 8% 이상")
    if signals.get("bb_breakout_up"):
        overheat += 15
        overheat_reasons.append("볼린저 상단 과도 이탈")
    if vol_ratio >= 8:
        overheat += 25
        overheat_reasons.append("거래량 8배↑")
    elif vol_ratio >= 5:
        overheat += 15
        overheat_reasons.append("거래량 5~8배")
    if upper_tail >= 0.50:
        overheat += 30
        overheat_reasons.append("윗꼬리 50%↑")
    elif upper_tail >= 0.40:
        overheat += 20
        overheat_reasons.append("윗꼬리 40%↑")
    if prior_gain_atr > 1.0:
        overheat += 20
        overheat_reasons.append("전일 급등 ATR초과")
    if day_range_atr > 2.0:
        overheat += 25
        overheat_reasons.append("당일 변동폭 ATR 과도")

    liquidity_risk = 0
    ta20 = _as_float(df["TurnoverAmt"].tail(20).mean(), close * vol20) if "TurnoverAmt" in df.columns else close * vol20
    ta5 = _as_float(df["TurnoverAmt"].tail(5).mean(), close * vol_now) if "TurnoverAmt" in df.columns else close * vol_now
    if vol_now < 10_000:
        liquidity_risk += 50
    elif vol_now < 30_000:
        liquidity_risk += 25
    if close < MIN_PRICE:
        liquidity_risk += 20
    if ta20 < MIN_AVG_TURNOVER_20:
        liquidity_risk += 20
    if ta5 < SOFT_MIN_AVG_TURNOVER_5:
        liquidity_risk += 10

    trend = float(np.clip(trend, 0, 100))
    momentum = float(np.clip(momentum, 0, 100))
    breakout = float(np.clip(breakout, 0, 100))
    overheat = int(min(100, overheat))
    liquidity_risk = int(min(100, liquidity_risk))

    return {
        "score": int(round((trend + momentum + breakout) / 3.0)),
        "score_raw": float((trend + momentum + breakout) / 3.0),
        "trend_score": int(round(trend)),
        "trend_raw": trend,
        "momentum_score": int(round(momentum)),
        "momentum_raw": momentum,
        "breakout_score": int(round(breakout)),
        "breakout_raw": breakout,
        "overheat_score": overheat,
        "liquidity_risk": liquidity_risk,
        "grade": _grade_label((trend + momentum + breakout) / 3.0, overheat),
        "reasons_trend": reasons_trend,
        "reasons_momentum": reasons_momentum,
        "reasons_breakout": reasons_breakout,
        "overheat_reasons": list(dict.fromkeys(overheat_reasons)),
        "positive_factors": list(dict.fromkeys(positives))[:6],
        "risk_factors": list(dict.fromkeys(risks))[:6],
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
                return {
                    "individual": float(df.loc["개인", col]) if "개인" in df.index else 0.0,
                    "institution": float(df.loc["기관합계", col]) if "기관합계" in df.index else 0.0,
                    "foreigner": float(df.loc["외국인합계", col]) if "외국인합계" in df.index else 0.0,
                    "base": float(df[col].abs().sum()) or 1.0,
                }
            except Exception:
                if attempt == 0:
                    time.sleep(random.uniform(0.3, 0.7))
        return None

    r5 = fetch(5)
    r20 = fetch(20)

    def block_score(r):
        if r is None:
            return None
        weighted = 0.30 * r["individual"] + 0.40 * r["institution"] + 0.30 * r["foreigner"]
        return int(max(0, min(100, 50 + 35 * np.tanh(weighted / r["base"]))))

    short = block_score(r5)
    mid = block_score(r20)

    if r5 is not None and r20 is not None:
        net5 = r5["foreigner"] + r5["institution"]
        net20 = r20["foreigner"] + r20["institution"]
        persist = 70 if (net5 > 0 and net20 > 0) else (30 if (net5 < 0 and net20 < 0) else 50)
        status = "full_data"
    elif r5 is not None:
        net5 = r5["foreigner"] + r5["institution"]
        persist = 60 if net5 > 0 else (40 if net5 < 0 else 50)
        status = "partial_data"
    elif r20 is not None:
        net20 = r20["foreigner"] + r20["institution"]
        persist = 60 if net20 > 0 else (40 if net20 < 0 else 50)
        status = "partial_data"
    else:
        return None, {}, {"short": None, "mid": None, "persist": None, "status": "N/A"}

    final = int(round(0.40 * (short if short is not None else 50) + 0.35 * (mid if mid is not None else 50) + 0.25 * persist))
    return final, (r5 or r20 or {}), {"short": short, "mid": mid, "persist": persist, "status": status}


def volume_score(df: pd.DataFrame, days: int = 20) -> int:
    try:
        if "Volume" not in df.columns or df.empty:
            return 50
        last_vol = float(df["Volume"].iloc[-1])
        avg_vol = float(df["Volume"].tail(days).mean())
        if avg_vol <= 0:
            return 50
        ratio = last_vol / avg_vol
        return int(max(0, min(100, 50 + 40 * np.tanh(ratio - 1))))
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
        if df is None or df.empty:
            return None
        df = df.copy()
        if "Close" not in df.columns or "Volume" not in df.columns:
            return None
        for col in ["Open", "High", "Low"]:
            if col not in df.columns:
                df[col] = df["Close"]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close", "Volume"])
        if len(df) < 60:
            return None

        try:
            df = add_indicators(df.tail(lookback).copy())
        except Exception:
            df = df.tail(lookback).copy()
            df["TurnoverAmt"] = df["Close"] * df["Volume"]
        if len(df) < 60:
            return None

        close = _as_float(df["Close"].iloc[-1], 0.0)
        if close <= 0:
            return None
        avg_turnover20 = _as_float(df["TurnoverAmt"].tail(20).mean(), close * _as_float(df["Volume"].tail(20).mean(), 0.0)) if "TurnoverAmt" in df.columns else close * _as_float(df["Volume"].tail(20).mean(), 0.0)
        avg_turnover5 = _as_float(df["TurnoverAmt"].tail(5).mean(), close * _as_float(df["Volume"].tail(5).mean(), 0.0)) if "TurnoverAmt" in df.columns else close * _as_float(df["Volume"].tail(5).mean(), 0.0)

        try:
            fib = find_fib_levels(df)
        except Exception:
            fib = {}
        try:
            signals = basic_signals(df)
        except Exception:
            signals = {}
        try:
            tech = score_technical_practical(df, signals, fib) or _default_technical()
        except Exception:
            tech = _default_technical()
        for key, value in _default_technical().items():
            tech.setdefault(key, value)
        flow_score = None
        flow_detail = {"short": None, "mid": None, "persist": None, "status": "N/A"}
        if include_flow:
            try:
                flow_score, _, flow_detail = investor_flow_score(code)
            except Exception:
                flow_score = None
                flow_detail = {"short": None, "mid": None, "persist": None, "status": "N/A"}

        vol_score = volume_score(df)
        vol_avg20 = _as_float(df["Volume"].tail(20).mean(), 0.0) if "Volume" in df.columns else 0.0
        vol_ratio = _as_float(df["Volume"].iloc[-1], 0.0) / vol_avg20 if vol_avg20 > 0 else None

        try:
            ichi = ichimoku_signals(df)
        except Exception:
            ichi = {}
        ichi_score = (
            (10 if ichi.get("tenkan_kijun") == "bull" else 0)
            + (10 if ichi.get("cloud") == "above" else 0)
            + (5 if ichi.get("chikou") == "bull_confirmed" else 0)
        )
        ichi_detail = [
            "전환-기준 강세" if ichi.get("tenkan_kijun") == "bull" else "전환-기준 약세",
            "구름 상승구간" if ichi.get("cloud") == "above" else f"구름 {ichi.get('cloud', 'N/A')}",
            "후행스팬 상승 확인" if ichi.get("chikou") == "bull_confirmed" else "후행스팬 미확인",
        ]
        ichi_calc_score = ichi_score * 4

        if flow_score is None:
            denom = (
                BASE_WEIGHTS["trend"]
                + BASE_WEIGHTS["momentum"]
                + BASE_WEIGHTS["breakout"]
                + BASE_WEIGHTS["volume"]
                + BASE_WEIGHTS["ichi"]
            )
            weights = {
                "trend": BASE_WEIGHTS["trend"] / denom,
                "momentum": BASE_WEIGHTS["momentum"] / denom,
                "breakout": BASE_WEIGHTS["breakout"] / denom,
                "flow": 0.0,
                "volume": BASE_WEIGHTS["volume"] / denom,
                "ichi": BASE_WEIGHTS["ichi"] / denom,
            }
        else:
            weights = dict(BASE_WEIGHTS)

        base_score = (
            weights["trend"] * _as_float(tech.get("trend_raw"), 50.0)
            + weights["momentum"] * _as_float(tech.get("momentum_raw"), 50.0)
            + weights["breakout"] * _as_float(tech.get("breakout_raw"), 0.0)
            + weights["flow"] * (float(flow_score) if flow_score is not None else 0.0)
            + weights["volume"] * vol_score
            + weights["ichi"] * ichi_calc_score
        )
        base_score = max(0.0, min(100.0, base_score))
        overheat_score = int(round(_as_float(tech.get("overheat_score"), 0.0)))
        liquidity_risk = int(round(_as_float(tech.get("liquidity_risk"), 0.0)))
        overheat_penalty = 10.0 * (float(overheat_score) / 100.0)
        liq_penalty = 8.0 * (float(liquidity_risk) / 100.0)
        final_raw = max(0.0, min(100.0, base_score - overheat_penalty - liq_penalty))
        final_score = int(round(final_raw))
        trend_i = int(round(_as_float(tech.get("trend_score"), _as_float(tech.get("trend_raw"), 50.0))))
        momentum_i = int(round(_as_float(tech.get("momentum_score"), _as_float(tech.get("momentum_raw"), 50.0))))
        breakout_i = int(round(_as_float(tech.get("breakout_score"), _as_float(tech.get("breakout_raw"), 50.0))))
        volume_summary = (
            f"오늘 거래량 / 20일 평균 = {vol_ratio:.2f}배 → 거래량 점수 {vol_score}"
            if vol_ratio is not None else "거래량 데이터 부족 → 거래량 점수 50"
        )
        ichi_summary = " | ".join(ichi_detail) + f" → {ichi_score}/25"
        risk_summary = f"Overheat {overheat_score} → -{overheat_penalty:.1f} | LiqRisk {liquidity_risk} → -{liq_penalty:.1f}"
        if tech.get("overheat_reasons"):
            risk_summary += "\n주요 과열: " + ", ".join(tech.get("overheat_reasons", [])[:4])
        positives = list(tech.get("positive_factors", []))[:6]
        risks = list(tech.get("risk_factors", []))[:6]
        if flow_score is None and "Flow N/A" not in risks:
            risks.append("Flow N/A")
        if liquidity_risk >= 40 and "LiqRisk 높음" not in risks:
            risks.append("LiqRisk 높음")
        if signals.get("near_52w_high") and "전고점 근접" not in risks:
            risks.append("전고점 근접")
        if trend_i >= 65 and overheat_score < 50:
            judgment = "추세와 모멘텀이 양호해 관심권이나, 진입가는 눌림 확인이 유리합니다."
        elif trend_i >= 65 and overheat_score >= 50:
            judgment = "추세는 강하지만 단기 과열 부담이 있어 신규 추격보다 눌림 대기가 유리합니다."
        elif breakout_i >= 65 and overheat_score < 50:
            judgment = "거래량과 캔들 품질이 좋아 건강한 돌파성 흐름이 확인됩니다."
        elif breakout_i >= 65 and overheat_score >= 50:
            judgment = "돌파 신호는 강하지만 과열과 윗꼬리 부담이 있어 추격 진입은 주의가 필요합니다."
        else:
            judgment = "점수는 중립권이며 추세, 돌파 품질, 과열 부담을 함께 확인할 구간입니다."
        if flow_score is None:
            judgment += " 다만 수급 데이터가 없어 수급 판단은 제외했습니다."

        return {
            "Code": code,
            "Name": name,
            "Market": market,
            "BaseScore": float(base_score),
            "OverheatPenalty": float(overheat_penalty),
            "LiqPenalty": float(liq_penalty),
            "Score": final_score,
            "ScoreRaw": float(final_raw),
            "Grade": _grade_label(final_raw, overheat_score),
            "Trend": trend_i,
            "Momentum": momentum_i,
            "Breakout": breakout_i,
            "Flow": flow_score,
            "Vol": vol_score,
            "IchiSc": ichi_score,
            "Overheat": overheat_score,
            "LiqRisk": liquidity_risk,
            "Close": close,
            "AvgTurnover20": avg_turnover20,
            "AvgTurnover5": avg_turnover5,
            "VolumeRatio": vol_ratio,
            "Signals": signals,
            "Fib": fib,
            "FlowDetail": flow_detail,
            "IchiDetail": ichi_detail or [],
            "IchiScore": ichi_calc_score,
            "TrendSummary": " | ".join(tech.get("reasons_trend", [])) or "추세 근거 부족",
            "MomentumSummary": " | ".join(tech.get("reasons_momentum", [])) or "모멘텀 근거 부족",
            "BreakoutSummary": " | ".join(tech.get("reasons_breakout", [])) or "Breakout 근거 부족",
            "VolumeSummary": volume_summary,
            "IchiSummary": ichi_summary,
            "RiskSummary": risk_summary,
            "PositiveFactors": positives,
            "RiskFactors": risks[:6],
            "Judgment": judgment,
            "Weights": weights,
            "Technical": tech,
        }
    except Exception:
        return None
