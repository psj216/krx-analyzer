"""
core/signals.py
Technical signals, candle patterns, and Ichimoku summaries.
"""
import numpy as np
import pandas as pd


def basic_signals(df: pd.DataFrame) -> dict:
    if df is None or df.empty or len(df) < 20:
        return {}
    n = len(df)
    last = df.iloc[-1]
    prev = df.iloc[-2] if n >= 2 else last

    def xo(an, bn, ap, bp):
        if any(pd.isna(x) for x in [an, bn, ap, bp]):
            return False
        return ap <= bp and an > bn

    def xu(an, bn, ap, bp):
        if any(pd.isna(x) for x in [an, bn, ap, bp]):
            return False
        return ap >= bp and an < bn

    sig = {}
    rsi = last.get("RSI14", np.nan)
    close = last.get("Close", np.nan)
    bb_u = last.get("BBU_20_2.0", np.nan)
    bb_l = last.get("BBL_20_2.0", np.nan)

    sig["rsi_overbought"] = (not pd.isna(rsi)) and rsi >= 70
    sig["rsi_oversold"] = (not pd.isna(rsi)) and rsi <= 30
    sig["bb_breakout_up"] = (not pd.isna(bb_u)) and close > bb_u
    sig["bb_breakout_dn"] = (not pd.isna(bb_l)) and close < bb_l

    if n >= 60:
        m = last.get("MACD_12_26_9", np.nan)
        ms = last.get("MACDs_12_26_9", np.nan)
        sig["macd_bull_cross"] = xo(m, ms, prev.get("MACD_12_26_9"), prev.get("MACDs_12_26_9"))
        sig["macd_bear_cross"] = xu(m, ms, prev.get("MACD_12_26_9"), prev.get("MACDs_12_26_9"))

    sma20 = last.get("SMA20", np.nan)
    sma60 = last.get("SMA60", np.nan)
    if n >= 120 and not pd.isna(sma20) and not pd.isna(sma60):
        sig["trend_ma20_gt_ma60"] = "UP" if sma20 > sma60 else ("DOWN" if sma20 < sma60 else "FLAT")
        sig["golden_cross"] = xo(sma20, sma60, prev.get("SMA20"), prev.get("SMA60"))
        sig["death_cross"] = xu(sma20, sma60, prev.get("SMA20"), prev.get("SMA60"))
    else:
        sig["trend_ma20_gt_ma60"] = "N/A"
        sig["golden_cross"] = False
        sig["death_cross"] = False

    if n >= 250:
        h52 = df["High"].tail(252).max()
        l52 = df["Low"].tail(252).min()
        sig["near_52w_high"] = close >= h52 * 0.97
        sig["near_52w_low"] = close <= l52 * 1.03
        sig["52w_high"] = h52
        sig["52w_low"] = l52
    else:
        sig["near_52w_high"] = False
        sig["near_52w_low"] = False

    if "TurnoverAmt" in df.columns and n >= 20:
        ta = float(df["TurnoverAmt"].iloc[-1])
        ta_avg = float(df["TurnoverAmt"].tail(20).mean())
        sig["turnover_surge"] = ta > ta_avg * 3.0
        sig["turnover_ratio"] = ta / ta_avg if ta_avg > 0 else 0

    suggestions = []
    if sig.get("golden_cross"):
        suggestions.append("golden_cross")
    if sig.get("death_cross"):
        suggestions.append("death_cross")
    if sig.get("near_52w_high"):
        suggestions.append("near_52w_high")
    if sig.get("near_52w_low"):
        suggestions.append("near_52w_low")
    if sig.get("turnover_surge"):
        suggestions.append(f"turnover_surge {sig.get('turnover_ratio', 0):.1f}x")
    sig["suggestions"] = suggestions
    return sig


def candle_bull_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, cur = df.iloc[-2], df.iloc[-1]
    if not (prev["Close"] < prev["Open"] and cur["Close"] > cur["Open"]):
        return False
    body_prev = abs(prev["Close"] - prev["Open"])
    body_cur = abs(cur["Close"] - cur["Open"])
    return body_cur > body_prev and cur["Open"] <= prev["Close"] and cur["Close"] >= prev["Open"]


def candle_breakout_long(df: pd.DataFrame) -> bool:
    if len(df) < 21:
        return False
    last = df.iloc[-1]
    if last["Close"] <= last["Open"]:
        return False
    bodies = (df["Close"] - df["Open"]).tail(40).abs()
    med = bodies.median()
    if pd.isna(med) or med == 0:
        return False
    return abs(last["Close"] - last["Open"]) > 1.5 * med and last["Close"] > df["High"].iloc[-21:-1].max()


def ichimoku_signals(df: pd.DataFrame) -> dict:
    try:
        if df is None or len(df) < 78:
            return {}
        h9 = df["High"].rolling(9).max()
        l9 = df["Low"].rolling(9).min()
        h26 = df["High"].rolling(26).max()
        l26 = df["Low"].rolling(26).min()
        h52 = df["High"].rolling(52).max()
        l52 = df["Low"].rolling(52).min()

        tenkan = (h9 + l9) / 2
        kijun = (h26 + l26) / 2
        span_a = ((tenkan + kijun) / 2).shift(26)
        span_b = ((h52 + l52) / 2).shift(26)
        close = df["Close"]
        last_close = float(close.iloc[-1])

        sig = {}
        tk_state = "bull" if tenkan.iloc[-1] > kijun.iloc[-1] else "bear"
        sig["tenkan_kijun"] = tk_state
        sig["전환-기준"] = "강세" if tk_state == "bull" else "약세"
        if last_close > max(span_a.iloc[-1], span_b.iloc[-1]):
            sig["cloud"] = "above"
            sig["구름"] = "상승구간"
        elif last_close < min(span_a.iloc[-1], span_b.iloc[-1]):
            sig["cloud"] = "below"
            sig["구름"] = "하락구간"
        else:
            sig["cloud"] = "inside"
            sig["구름"] = "중립"

        # Chikou span is today's close compared against price 26 periods ago.
        ref_close = float(close.iloc[-27]) if len(close) >= 27 else np.nan
        chikou_state = "bull_confirmed" if not pd.isna(ref_close) and last_close > ref_close else "bear_confirmed"
        sig["chikou"] = chikou_state
        sig["후행스팬"] = "상승 확인" if chikou_state == "bull_confirmed" else "하락 확인"
        return sig
    except Exception:
        return {}
