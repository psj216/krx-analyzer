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
        if df is None or df.empty or len(df) < 60:
            return None

        df = add_indicators(df.tail(lookback).copy())
        if len(df) < 60:
            return None

        close = float(df["Close"].iloc[-1])
        avg_turnover20 = float(df["TurnoverAmt"].tail(20).mean()) if "TurnoverAmt" in df.columns else float(df["Close"].iloc[-1] * df["Volume"].tail(20).mean())
        avg_turnover5 = float(df["TurnoverAmt"].tail(5).mean()) if "TurnoverAmt" in df.columns else float(df["Close"].iloc[-1] * df["Volume"].tail(5).mean())

        fib = find_fib_levels(df)
        signals = basic_signals(df)
        tech = score_technical(df, signals, fib)
        flow_score = None
        flow_detail = {"short": None, "mid": None, "persist": None, "status": "N/A"}
        if include_flow:
            flow_score, _, flow_detail = investor_flow_score(code)

        vol_score = volume_score(df)
        vol_avg20 = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else 0.0
        vol_ratio = float(df["Volume"].iloc[-1] / vol_avg20) if vol_avg20 > 0 else None

        ichi = ichimoku_signals(df)
        ichi_score = (
            (10 if ichi.get("전환-기준") == "강세" else 0)
            + (10 if ichi.get("구름") == "상승구간" else 0)
            + (5 if ichi.get("후행스팬") == "상승 확인" else 0)
        )
        ichi_detail = [
            f"전환선 > 기준선 여부: {'강세' if ichi.get('전환-기준') == '강세' else '비강세'}",
            f"구름 위치: {ichi.get('구름', 'N/A')}",
            f"후행스팬 확인: {ichi.get('후행스팬', 'N/A')}",
        ]

        if flow_score is None:
            denom = BASE_WEIGHTS["trend"] + BASE_WEIGHTS["momentum"] + BASE_WEIGHTS["breakout"] + BASE_WEIGHTS["volume"] + BASE_WEIGHTS["ichi"]
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

        final_raw = (
            weights["trend"] * tech["trend_raw"]
            + weights["momentum"] * tech["momentum_raw"]
            + weights["breakout"] * tech["breakout_raw"]
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
            "Breakout": tech["breakout_score"],
            "Flow": flow_score,
            "Vol": vol_score,
            "IchiSc": ichi_score,
            "Overheat": tech["overheat_score"],
            "LiqRisk": tech["liquidity_risk"],
            "Close": close,
            "AvgTurnover20": avg_turnover20,
            "AvgTurnover5": avg_turnover5,
            "VolumeRatio": vol_ratio,
            "Signals": signals,
            "Fib": fib,
            "FlowDetail": flow_detail,
            "IchiDetail": ichi_detail,
            "Weights": weights,
            "Technical": tech,
        }
    except Exception:
        return None
