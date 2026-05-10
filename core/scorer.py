"""
core/scorer.py
Five-factor scoring model:
Score = 0.35*Trend + 0.25*Momentum + 0.25*Flow + 0.10*Volume + 0.05*Ichi
When Flow is unavailable, the 25% flow weight is redistributed proportionally.
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


def _grade_label(score: float) -> str:
    if score >= 72:
        return "매수 유력"
    if score >= 60:
        return "매수 관심"
    if score >= 48:
        return "관망/보유"
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
        "liquidity_risk": 0,
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
        delta = 4.0 if close > sma120 else -5.0
        trend += delta
        reasons_trend.append(f"현재가가 SMA120 {'위에 있어 중장기 추세 기준 상방' if delta > 0 else '아래에 있어 중장기 추세 기준 약세'} → {delta:+.1f}")
    if n >= 2 and not pd.isna(sma20):
        prev_sma20 = float(df["SMA20"].iloc[-2])
        if prev_sma20:
            slope20 = (sma20 - prev_sma20) / prev_sma20
            delta = 3.0 if 0 < slope20 < 0.003 else (1.0 if slope20 < 0.01 else (-4.0 if slope20 >= 0.01 else -1.0))
            trend += delta
            reasons_trend.append(f"SMA20 기울기가 {slope20*100:.2f}%로 {'단기 추세가 우상향' if delta > 0 else '과열 또는 둔화 부담이 존재'} → {delta:+.1f}")
    if n >= 2 and not pd.isna(sma60):
        prev_sma60 = float(df["SMA60"].iloc[-2])
        if prev_sma60:
            slope60 = (sma60 - prev_sma60) / prev_sma60
            delta = 2.0 if 0 < slope60 < 0.003 else (1.0 if slope60 < 0.007 else (-3.0 if slope60 >= 0.007 else -1.0))
            trend += delta
            reasons_trend.append(f"SMA60 기울기가 {slope60*100:.2f}%로 {'중기 추세 흐름이 양호' if delta > 0 else '중기 추세가 둔화되거나 과열'} → {delta:+.1f}")
    if not pd.isna(sma20) and sma20:
        dist20 = close / sma20 - 1
        delta = 3.0 if abs(dist20) <= 0.05 else (0.0 if dist20 <= 0.10 else (-6.0 if dist20 > 0.10 else -2.0))
        trend += delta
        reasons_trend.append(f"20일선 대비 이격이 {dist20*100:.1f}%로 {'적정 범위에 있어 부담이 적음' if delta > 0 else '단기 과열 또는 이탈 부담이 있음'} → {delta:+.1f}")
    if not pd.isna(sma20) and not pd.isna(sma60) and sma20 and sma60:
        delta = 3.0 if sma20 > sma60 else -5.0
        trend += delta
        reasons_trend.append(f"20일선이 60일선 {'위에 있어 정배열 유지' if delta > 0 else '아래에 있어 역배열 상태'} → {delta:+.1f}")
    if "BBM_20_2.0" in df.columns:
        mid = float(last["BBM_20_2.0"])
        delta = 2.0 if close > mid else -3.0
        trend += delta
        reasons_trend.append(f"현재가가 볼린저 중단선 {'위에 있어 추세가 우호적' if delta > 0 else '아래에 있어 추세가 약함'} → {delta:+.1f}")
    if "OBV" in df.columns and n >= 2:
        delta = 2.0 if float(last["OBV"]) > float(df["OBV"].iloc[-2]) else -2.0
        trend += delta
        reasons_trend.append(f"OBV가 {'상승해 거래량 흐름이 긍정적' if delta > 0 else '하락해 거래량 흐름이 약화'} → {delta:+.1f}")
    if not pd.isna(adx):
        delta = -3.0 if adx < 20 else (3.0 if adx < 30 else (6.0 if adx < 40 else -2.0))
        trend += delta
        reasons_trend.append(f"ADX {adx:.1f}로 {'추세 강도가 양호' if delta > 0 else '추세 강도가 약하거나 과열'} → {delta:+.1f}")
    if "IchiSpanA" in df.columns and "IchiSpanB" in df.columns:
        span_a = float(last["IchiSpanA"])
        span_b = float(last["IchiSpanB"])
        if not pd.isna(span_a) and not pd.isna(span_b):
            cloud_width = abs(span_a - span_b) / close
            delta = -3.0 if cloud_width < 0.02 else (1.0 if cloud_width < 0.05 else (3.0 if cloud_width < 0.10 else -2.0))
            trend += delta
            reasons_trend.append(f"일목 구름 두께가 {cloud_width*100:.1f}%로 {'추세 해석에 무리가 적음' if delta > 0 else '추세 해석에 불리한 구조'} → {delta:+.1f}")
    if fib and "levels" in fib and fib.get("direction") == "up":
        levels = fib["levels"]
        lo, hi = min(levels["61.8%"], levels["38.2%"]), max(levels["61.8%"], levels["38.2%"])
        if lo <= close <= hi:
            trend += 2.0
            reasons_trend.append("피보나치 38.2~61.8% 구간에 있어 눌림목 위치가 양호 → +2.0")

    momentum = 50.0
    is_breakout = candle_breakout_long(df)

    if not pd.isna(sma5) and sma5:
        above5 = close > sma5
        if above5:
            momentum += 4.0
            reasons_momentum.append("현재가가 5일선 위에 있어 단기 모멘텀이 양호 → +4.0")
        if n >= 2:
            prev_sma5 = float(df["SMA5"].iloc[-2])
            slope5 = (sma5 - prev_sma5) / prev_sma5 if prev_sma5 else 0.0
            rising5 = slope5 > 0
            if rising5:
                momentum += 3.0
                reasons_momentum.append("5일선이 우상향하고 있어 단기 상승 흐름이 유지됨 → +3.0")
            if above5 and rising5:
                momentum += 2.0
                reasons_momentum.append("현재가가 5일선 위에 있고 5일선도 상승 중이라 단기 탄력이 좋음 → +2.0")
            if slope5 > 0.015:
                momentum -= 2.0
                reasons_momentum.append("5일선 기울기가 너무 가팔라 단기 과열 부담이 있음 → -2.0")
        if gap5_pct >= 95:
            momentum -= 4.0
            reasons_momentum.append("5일선 대비 이격이 자기 분포 상위 5% 수준이라 과열 부담이 큼 → -4.0")
        elif gap5_pct >= 90:
            momentum -= 2.0
            reasons_momentum.append("5일선 대비 이격이 자기 분포 상위 10% 수준이라 추격 부담이 있음 → -2.0")
        elif 0 < gap5_now <= 0.03:
            momentum += 3.0
            reasons_momentum.append("5일선 대비 이격이 0~3% 범위라 무리 없는 상승 구간 → +3.0")

    if rsi_now is not None:
        if 50 < rsi_now <= 55:
            momentum += 4.0
            reasons_momentum.append(f"RSI {rsi_now:.1f}로 모멘텀 초입 구간에 진입 → +4.0")
        elif 55 < rsi_now <= 65:
            momentum += 3.0
            reasons_momentum.append(f"RSI {rsi_now:.1f}로 상승 모멘텀 구간이 유지됨 → +3.0")
        if 70 <= rsi_now < 75:
            momentum -= 1.0
            reasons_momentum.append(f"RSI {rsi_now:.1f}로 단기 과열 경고 구간 진입 → -1.0")
        elif 75 <= rsi_now < 80:
            momentum -= 3.0
            reasons_momentum.append(f"RSI {rsi_now:.1f}로 과열 부담이 커짐 → -3.0")
        elif rsi_now >= 80:
            momentum -= 5.0
            reasons_momentum.append(f"RSI {rsi_now:.1f}로 과열이 매우 강함 → -5.0")

    if n >= 3 and "RSI14" in df.columns and rsi_now is not None:
        rs1 = float(df["RSI14"].iloc[-1]) - float(df["RSI14"].iloc[-2])
        rs2 = float(df["RSI14"].iloc[-2]) - float(df["RSI14"].iloc[-3])
        if rs1 > 0 and rs2 > 0 and 30 < rsi_now < 65:
            momentum += 4.0
            reasons_momentum.append("RSI가 연속 상승해 모멘텀 강화가 확인됨 → +4.0")
        elif rs1 < 0 and rs2 < 0 and rsi_now > 50:
            momentum -= 3.0
            reasons_momentum.append("RSI 상승 탄력이 둔화돼 힘이 약해짐 → -3.0")

    if "MACD_12_26_9" in df.columns and n >= 3:
        macd_val = float(last["MACD_12_26_9"])
        signal_val = float(last["MACDs_12_26_9"])
        hist = macd_val - signal_val
        hist_prev = float(df["MACD_12_26_9"].iloc[-2]) - float(df["MACDs_12_26_9"].iloc[-2])
        macd_slope1 = float(df["MACD_12_26_9"].iloc[-1]) - float(df["MACD_12_26_9"].iloc[-2])
        macd_slope2 = float(df["MACD_12_26_9"].iloc[-2]) - float(df["MACD_12_26_9"].iloc[-3])
        if hist > 0 and hist_prev <= 0:
            momentum += 5.0
            reasons_momentum.append("MACD가 상향 전환되며 상승 모멘텀 초입 신호가 발생 → +5.0")
        if hist > 0 and hist_prev > 0 and hist > hist_prev:
            if macd_slope1 > 0 and macd_slope1 > macd_slope2:
                momentum += 5.0
                reasons_momentum.append("MACD 흐름이 개선되며 상승 모멘텀이 더 강해짐 → +5.0")
            else:
                momentum += 3.0
                reasons_momentum.append("MACD 흐름이 개선되며 상승 모멘텀이 강화 → +3.0")
        if hist_prev > 0 and hist < hist_prev:
            momentum -= 4.0
            reasons_momentum.append("MACD 히스토그램이 둔화돼 상승 탄력이 약해짐 → -4.0")

    if all(c in df.columns for c in ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]) and n >= 2:
        bbu = float(last["BBU_20_2.0"])
        bbl = float(last["BBL_20_2.0"])
        bbm = float(last["BBM_20_2.0"])
        width_now = (bbu - bbl) / (bbm or 1.0)
        prev_bbu = float(df["BBU_20_2.0"].iloc[-2])
        prev_bbl = float(df["BBL_20_2.0"].iloc[-2])
        prev_bbm = float(df["BBM_20_2.0"].iloc[-2])
        width_prev = (prev_bbu - prev_bbl) / (prev_bbm or 1.0)
        if width_prev < 0.05 and width_now > width_prev * 1.1:
            momentum += 3.0
            reasons_momentum.append("볼린저 밴드 수축 이후 확장이 시작돼 변동성 돌파 가능성이 커짐 → +3.0")
        dist_bbu = (close - bbu) / (bbu or 1.0)
        if 0.03 < dist_bbu <= 0.05:
            momentum -= 3.0
            reasons_momentum.append("볼린저 상단 대비 3~5% 위라 단기 과열 부담이 있음 → -3.0")
        if dist_bbu > 0.05:
            momentum -= 6.0
            reasons_momentum.append("볼린저 상단 대비 5% 이상 과열돼 추격 부담이 큼 → -6.0")

    if "Volume" in df.columns:
        vol20 = float(df["Volume"].tail(20).mean())
        ratio = vol_now / vol20 if vol20 > 0 else 1.0
        if vol_now < 10_000:
            momentum -= 8.0
            reasons_momentum.append("거래량이 1만주 미만이라 모멘텀 신뢰도가 낮음 → -8.0")
        elif vol_now < 30_000:
            momentum -= 4.0
            reasons_momentum.append("거래량이 1~3만주 수준이라 탄력이 제한적일 수 있음 → -4.0")
        if vol_pct <= 20:
            momentum -= 6.0
            reasons_momentum.append("거래량이 최근 분포 하위 20%라 수급 탄력이 약함 → -6.0")
        elif 1.2 <= ratio <= 2.5:
            delta = 5.0 if is_breakout else 3.0
            momentum += delta
            reasons_momentum.append(f"거래량이 평균 대비 1.2~2.5배로 {'돌파와 함께 증가해' if is_breakout else '자연스럽게 늘어'} 모멘텀이 양호 → {delta:+.1f}")
        elif 2.5 < ratio <= 5:
            delta = 4.0 if is_breakout else 1.0
            momentum += delta
            reasons_momentum.append(f"거래량이 평균 대비 2.5~5배로 {'돌파를 동반해' if is_breakout else '강하게 증가해'} 매수세가 유입됨 → {delta:+.1f}")
        elif 5 < ratio <= 8:
            momentum -= 2.0
            reasons_momentum.append("거래량이 5~8배 급증해 단기 과열 가능성을 경계해야 함 → -2.0")
        elif ratio > 8:
            momentum -= 5.0
            reasons_momentum.append("거래량이 8배 이상 급증해 과열 신호로 해석됨 → -5.0")

    if signals.get("turnover_surge"):
        rng = max(high - low, 1e-9)
        tail_up = (high - max(open_, close)) / rng
        if tail_up < 0.3:
            momentum += 3.0
            reasons_momentum.append("거래대금이 급증했고 윗꼬리가 짧아 매수 우위가 확인됨 → +3.0")
        else:
            momentum += 1.0
            reasons_momentum.append("거래대금은 늘었지만 윗꼬리 부담이 남아 있어 가점은 제한적 → +1.0")

    rng = max(high - low, 1e-9)
    body = abs(close - open_)
    tail_up = (high - max(open_, close)) / rng
    tail_dn = (min(open_, close) - low) / rng
    if close > open_ and body / rng >= 0.6 and tail_up < 0.2:
        delta = 5.0 if vol_pct >= 60 else 2.0
        momentum += delta
        reasons_momentum.append(f"몸통이 강한 양봉으로 {'거래량까지 동반해 ' if vol_pct >= 60 else ''}매수세 우위가 확인됨 → {delta:+.1f}")
    if tail_dn >= 0.4 and close >= open_:
        momentum += 3.0
        reasons_momentum.append("아래꼬리가 길어 저가 매수 유입이 확인됨 → +3.0")
    if tail_up >= 0.4:
        momentum -= 4.0
        reasons_momentum.append("윗꼬리 비중이 40% 이상이라 매물 부담이 큼 → -4.0")
    if candle_bull_engulfing(df):
        delta = 4.0 if vol_pct >= 60 else 2.0
        momentum += delta
        reasons_momentum.append(f"양봉 감싸기 패턴이 {'거래량과 함께 ' if vol_pct >= 60 else ''}나와 반전 신호로 해석됨 → {delta:+.1f}")

    if not pd.isna(atr_val) and atr_val > 0 and close > 0:
        atr_pct = atr_val / close
        if abs(day_ret) > atr_pct * 2.5:
            momentum -= 5.0
            reasons_momentum.append("당일 변동성이 ATR 대비 과도해 단기 추격 위험이 큼 → -5.0")
        elif 0.01 <= day_ret <= atr_pct * 1.5:
            momentum += 2.0
            reasons_momentum.append("상승폭이 과도하지 않아 안정적인 상승 흐름으로 해석됨 → +2.0")
        elif day_ret > atr_pct * 1.5:
            momentum -= 3.0
            reasons_momentum.append("당일 급등폭이 커서 단기 과열 부담이 생김 → -3.0")
    else:
        if 0.01 <= day_ret <= 0.03:
            momentum += 2.0
        elif 0.03 < day_ret <= 0.06:
            momentum += 3.0
        elif day_ret > 0.06:
            momentum -= 6.0

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
    if vol_now < 10_000:
        liquidity_risk += 50
    elif vol_now < 30_000:
        liquidity_risk += 25
    if vol_pct <= 20:
        liquidity_risk += 30
    if close < MIN_PRICE:
        liquidity_risk += 20
    if ta20 < MIN_AVG_TURNOVER_20:
        liquidity_risk += 20
    if ta5 < SOFT_MIN_AVG_TURNOVER_5:
        liquidity_risk += 10
    liquidity_risk = min(100, liquidity_risk)

    trend = float(np.clip(trend, 0, 100))
    momentum = float(np.clip(momentum, 0, 100))
    score_raw = (trend + momentum) / 2.0
    score = int(round(score_raw))

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
        return int(max(0, min(100, round(ratio * 50))))
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
        avg_turnover20 = (
            float(df["TurnoverAmt"].tail(20).mean())
            if "TurnoverAmt" in df.columns
            else float(df["Close"].iloc[-1] * df["Volume"].tail(20).mean())
        )
        avg_turnover5 = (
            float(df["TurnoverAmt"].tail(5).mean())
            if "TurnoverAmt" in df.columns
            else float(df["Close"].iloc[-1] * df["Volume"].tail(5).mean())
        )
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

        flow_score = None
        flow_detail = {"short": None, "mid": None, "persist": None, "status": "N/A"}
        if include_flow:
            flow_score, _, flow_detail = investor_flow_score(code)

        vol_score = volume_score(df)
        if flow_score is None:
            weights = {
                "trend": 0.35 / 0.75,
                "momentum": 0.25 / 0.75,
                "flow": 0.0,
                "volume": 0.10 / 0.75,
                "ichi": 0.05 / 0.75,
            }
        else:
            weights = {
                "trend": 0.35,
                "momentum": 0.25,
                "flow": 0.25,
                "volume": 0.10,
                "ichi": 0.05,
            }

        final_raw = (
            weights["trend"] * tech["trend_raw"]
            + weights["momentum"] * tech["momentum_raw"]
            + weights["flow"] * (float(flow_score) if flow_score is not None else 0.0)
            + weights["volume"] * vol_score
            + weights["ichi"] * ichi_score
        )
        final_raw = max(0.0, min(100.0, final_raw))
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
            "Vol": vol_score,
            "IchiSc": ichi_score,
            "Weights": weights,
            "Close": close,
            "AvgTurnover20": avg_turnover20,
            "AvgTurnover5": avg_turnover5,
            "Signals": signals,
            "Fib": fib,
            "Technical": tech,
        }
    except Exception:
        return None
