"""
core/charts.py
Charts, narratives, and backtesting helpers.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.indicators import add_indicators
from core.signals import basic_signals


def indicator_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Close",
        "SMA5",
        "SMA20",
        "SMA60",
        "SMA120",
        "BBL_20_2.0",
        "BBM_20_2.0",
        "BBU_20_2.0",
        "BBP_20_2.0",
        "RSI14",
        "MACD_12_26_9",
        "MACDs_12_26_9",
        "ADX14",
        "ATR14",
        "Volume",
        "TurnoverAmt",
    ]
    use = [c for c in cols if c in df.columns]
    last = df[use].iloc[[-1]].copy()
    last.index = ["Latest"]
    return last


def make_chart(df: pd.DataFrame, title: str, fib: dict = None):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.25, 0.20],
    )
    fig.add_trace(
        go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"),
        row=1,
        col=1,
    )
    for ma, color in [("SMA5", "orange"), ("SMA20", "blue"), ("SMA60", "green"), ("SMA120", "red")]:
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[ma], name=ma, mode="lines", line=dict(color=color, width=1.2)),
                row=1,
                col=1,
            )
    for col in ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode="lines", opacity=0.4), row=1, col=1)
    if fib and "levels" in fib:
        for label, level in fib["levels"].items():
            fig.add_hline(y=level, line_dash="dot", opacity=0.2, annotation_text=f"Fib {label}", row=1, col=1)
    if "RSI14" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI14"], name="RSI14", line=dict(color="purple", width=1.2)),
            row=2,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="blue", opacity=0.5, row=2, col=1)
    if "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=3, col=1)
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=900,
        xaxis3_title="Date",
        yaxis_title="Price",
        yaxis2_title="RSI",
        yaxis3_title="Volume",
    )
    return fig


def make_narrative(code, name, market, df, signals, zones, fib) -> str:
    if df is None or df.empty or not signals:
        return "insufficient_data"
    last = df.iloc[-1]
    close = float(last["Close"])
    sma20 = float(last.get("SMA20", np.nan)) if not pd.isna(last.get("SMA20", np.nan)) else None
    rsi = float(last.get("RSI14", np.nan)) if not pd.isna(last.get("RSI14", np.nan)) else None
    lines = [f"**{code} {name} ({market})**", f"- close: {close:,.0f}"]
    if sma20:
        lines.append(f"- SMA20: {sma20:,.0f} ({'above' if close >= sma20 else 'below'})")
    if rsi:
        lines.append(f"- RSI(14): {rsi:.1f}")
    if signals.get("near_52w_high"):
        lines.append(f"- near 52w high ({signals.get('52w_high', 0):,.0f})")
    if signals.get("near_52w_low"):
        lines.append(f"- near 52w low ({signals.get('52w_low', 0):,.0f})")
    if signals.get("turnover_surge"):
        lines.append(f"- turnover surge {signals.get('turnover_ratio', 0):.1f}x")
    if zones:
        if "buy_zone" in zones:
            lines.append(f"- buy zone: {zones['buy_zone'][0]:,.0f}~{zones['buy_zone'][1]:,.0f}")
        if "sell_zone" in zones:
            lines.append(f"- sell zone: {zones['sell_zone'][0]:,.0f}~{zones['sell_zone'][1]:,.0f}")
        if "stop_loss" in zones:
            lines.append(f"- stop guide: {zones['stop_loss']:,.0f}")
    suggestions = signals.get("suggestions", [])
    if suggestions:
        lines.append("- signals: " + ", ".join(suggestions))
    return "\n".join(lines)


def _approx_entry_price(row: pd.Series, entry_mode: str) -> float:
    if entry_mode == "vwap":
        return float((row["Open"] + row["High"] + row["Low"] + row["Close"]) / 4.0)
    return float(row["Open"])


def simple_backtest(
    df: pd.DataFrame,
    signal_col: str = "golden_cross",
    hold_days: int = 5,
    entry_mode: str = "open",
    fee_bps: float = 5.0,
    tax_bps: float = 15.0,
    slippage_bps: float = 10.0,
    stop_loss_pct: float = 7.0,
    take_profit_pct: float = 12.0,
    trailing_stop_pct: float = 8.0,
    cooldown_days: int = 3,
) -> dict:
    if df is None or len(df) < 120:
        return {}

    df = add_indicators(df.copy())
    trades = []
    i = 1
    next_entry_idx_allowed = 1

    while i < len(df) - 1:
        if i < next_entry_idx_allowed:
            i += 1
            continue
        sig = basic_signals(df.iloc[: i + 1])
        if not sig.get(signal_col):
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= len(df):
            break
        entry_row = df.iloc[entry_idx]
        entry_raw = _approx_entry_price(entry_row, entry_mode)
        entry_price = entry_raw * (1 + slippage_bps / 10000.0 + fee_bps / 10000.0)
        stop_price = entry_raw * (1 - stop_loss_pct / 100.0)
        take_price = entry_raw * (1 + take_profit_pct / 100.0)
        trailing_high = float(entry_row["High"])
        trailing_stop = trailing_high * (1 - trailing_stop_pct / 100.0)

        exit_idx = None
        exit_reason = "time"
        exit_raw = float(df.iloc[min(entry_idx + hold_days, len(df) - 1)]["Close"])

        last_idx = min(entry_idx + hold_days, len(df) - 1)
        for j in range(entry_idx, last_idx + 1):
            row = df.iloc[j]
            trailing_high = max(trailing_high, float(row["High"]))
            trailing_stop = max(trailing_stop, trailing_high * (1 - trailing_stop_pct / 100.0))
            low = float(row["Low"])
            high = float(row["High"])
            open_ = float(row["Open"])

            # Conservative ordering when both hit intraday.
            if low <= stop_price:
                exit_idx = j
                exit_reason = "stop_loss"
                exit_raw = min(open_, stop_price) if open_ < stop_price else stop_price
                break
            if low <= trailing_stop:
                exit_idx = j
                exit_reason = "trailing_stop"
                exit_raw = min(open_, trailing_stop) if open_ < trailing_stop else trailing_stop
                break
            if high >= take_price:
                exit_idx = j
                exit_reason = "take_profit"
                exit_raw = max(open_, take_price) if open_ > take_price else take_price
                break

        if exit_idx is None:
            exit_idx = last_idx
            exit_raw = float(df.iloc[exit_idx]["Close"])

        exit_price = exit_raw * (1 - slippage_bps / 10000.0 - fee_bps / 10000.0 - tax_bps / 10000.0)
        ret = (exit_price - entry_price) / entry_price * 100.0
        hold = max(1, exit_idx - entry_idx + 1)
        trades.append(
            {
                "signal_date": str(df.index[i])[:10],
                "entry_date": str(df.index[entry_idx])[:10],
                "exit_date": str(df.index[exit_idx])[:10],
                "entry": round(entry_price, 2),
                "exit": round(exit_price, 2),
                "ret%": round(ret, 2),
                "hold_days": hold,
                "exit_reason": exit_reason,
            }
        )
        next_entry_idx_allowed = exit_idx + cooldown_days + 1
        i = max(i + 1, exit_idx + 1)

    if not trades:
        return {"count": 0}

    rets = np.array([t["ret%"] for t in trades], dtype=float)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    equity = np.cumprod(1 + rets / 100.0)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0

    return {
        "count": len(trades),
        "win_rate": round(float((rets > 0).mean() * 100), 1),
        "avg_ret": round(float(rets.mean()), 2),
        "max_ret": round(float(rets.max()), 2),
        "min_ret": round(float(rets.min()), 2),
        "avg_hold_days": round(float(np.mean([t["hold_days"] for t in trades])), 2),
        "profit_loss_ratio": round(float(wins.mean() / abs(losses.mean())), 2) if len(wins) and len(losses) else None,
        "mdd": round(float(drawdown.min() * 100), 2),
        "signals": trades[-20:],
    }
