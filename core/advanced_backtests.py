"""
core/advanced_backtests.py
Walk-forward reporting, relative strength, and portfolio-level simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from core.data import fetch_index_history
from core.ranking import KOSDAQ_TOP_N_DEFAULT, prepare_symbol_universe
from core.scorer import compute_score
from core.utils import KR_TZ, market_regime


DEFAULT_PERIODS = [
    ("2024 H1", "2024-01-01", "2024-06-30"),
    ("2024 H2", "2024-07-01", "2024-12-31"),
    ("2025 H1", "2025-01-01", "2025-06-30"),
    ("2025 H2", "2025-07-01", "2025-12-31"),
    ("2026 YTD", "2026-01-01", datetime.now(KR_TZ).strftime("%Y-%m-%d")),
]


@dataclass
class TradeConfig:
    hold_days: int = 20
    entry_mode: str = "open"
    fee_bps: float = 5.0
    tax_bps: float = 15.0
    slippage_bps: float = 10.0
    stop_loss_pct: float = 7.0
    take_profit_pct: float = 12.0
    trailing_stop_pct: float = 8.0


def relative_strength_metrics(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame | None) -> dict:
    if benchmark_df is None or benchmark_df.empty or stock_df is None or len(stock_df) < 61:
        return {"score": 50.0, "short": 0.0, "long": 0.0}
    stock_close = stock_df["Close"].dropna()
    bench_close = benchmark_df["Close"].dropna()
    common = stock_close.index.intersection(bench_close.index)
    if len(common) < 61:
        return {"score": 50.0, "short": 0.0, "long": 0.0}
    s = stock_close.loc[common]
    b = bench_close.loc[common]
    short = (s.iloc[-1] / s.iloc[-21] - 1) - (b.iloc[-1] / b.iloc[-21] - 1) if len(common) >= 21 else 0.0
    long = (s.iloc[-1] / s.iloc[-61] - 1) - (b.iloc[-1] / b.iloc[-61] - 1) if len(common) >= 61 else short
    score = float(np.clip(50 + short * 400 + long * 200, 0, 100))
    return {"score": score, "short": short, "long": long}


def _approx_entry_price(row: pd.Series, entry_mode: str) -> float:
    if entry_mode == "vwap":
        return float((row["Open"] + row["High"] + row["Low"] + row["Close"]) / 4.0)
    return float(row["Open"])


def simulate_trade_after_signal(df: pd.DataFrame, signal_date, cfg: TradeConfig) -> dict | None:
    if df is None or df.empty:
        return None
    signal_date = pd.Timestamp(signal_date)
    future = df[df.index > signal_date]
    if future.empty:
        return None
    entry_idx = future.index[0]
    entry_pos = df.index.get_loc(entry_idx)
    entry_row = df.iloc[entry_pos]
    entry_raw = _approx_entry_price(entry_row, cfg.entry_mode)
    entry_price = entry_raw * (1 + cfg.slippage_bps / 10000.0 + cfg.fee_bps / 10000.0)
    stop_price = entry_raw * (1 - cfg.stop_loss_pct / 100.0)
    take_price = entry_raw * (1 + cfg.take_profit_pct / 100.0)
    trailing_high = float(entry_row["High"])
    trailing_stop = trailing_high * (1 - cfg.trailing_stop_pct / 100.0)
    last_pos = min(entry_pos + cfg.hold_days, len(df) - 1)

    exit_pos = last_pos
    exit_reason = "time"
    exit_raw = float(df.iloc[last_pos]["Close"])
    for pos in range(entry_pos, last_pos + 1):
        row = df.iloc[pos]
        trailing_high = max(trailing_high, float(row["High"]))
        trailing_stop = max(trailing_stop, trailing_high * (1 - cfg.trailing_stop_pct / 100.0))
        low = float(row["Low"])
        high = float(row["High"])
        open_ = float(row["Open"])
        if low <= stop_price:
            exit_pos = pos
            exit_reason = "stop_loss"
            exit_raw = min(open_, stop_price) if open_ < stop_price else stop_price
            break
        if low <= trailing_stop:
            exit_pos = pos
            exit_reason = "trailing_stop"
            exit_raw = min(open_, trailing_stop) if open_ < trailing_stop else trailing_stop
            break
        if high >= take_price:
            exit_pos = pos
            exit_reason = "take_profit"
            exit_raw = max(open_, take_price) if open_ > take_price else take_price
            break

    exit_price = exit_raw * (1 - cfg.slippage_bps / 10000.0 - cfg.fee_bps / 10000.0 - cfg.tax_bps / 10000.0)
    ret = (exit_price - entry_price) / entry_price * 100.0
    hold = max(1, exit_pos - entry_pos + 1)
    return {
        "signal_date": str(signal_date.date()),
        "entry_date": str(df.index[entry_pos].date()),
        "exit_date": str(df.index[exit_pos].date()),
        "entry": round(entry_price, 2),
        "exit": round(exit_price, 2),
        "ret%": round(float(ret), 2),
        "hold_days": hold,
        "exit_reason": exit_reason,
    }


def _period_return(df: pd.DataFrame, start, end) -> float:
    if df is None or df.empty:
        return 0.0
    sub = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    if len(sub) < 2:
        return 0.0
    return float(sub["Close"].iloc[-1] / sub["Close"].iloc[0] - 1)


def _basket_mdd(returns_pct: list[float]) -> float:
    if not returns_pct:
        return 0.0
    equity = np.cumprod(1 + np.array(returns_pct, dtype=float) / 100.0)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1
    return float(dd.min() * 100.0)


def _score_bucket(score: float) -> str:
    if score >= 80:
        return "80+"
    if score >= 70:
        return "70-79"
    if score >= 60:
        return "60-69"
    if score >= 50:
        return "50-59"
    return "<50"


def _calendar_from_benchmarks(benchmarks: dict[str, pd.DataFrame], start, end) -> pd.DatetimeIndex:
    base = benchmarks.get("KOSPI")
    if base is None or base.empty:
        base = benchmarks.get("KOSDAQ")
    if base is None or base.empty:
        return pd.DatetimeIndex([])
    return base[(base.index >= pd.Timestamp(start)) & (base.index <= pd.Timestamp(end))].index


def _rebalance_dates(calendar: pd.DatetimeIndex, step: int) -> list[pd.Timestamp]:
    if len(calendar) == 0:
        return []
    step = max(int(step), 1)
    return [calendar[i] for i in range(0, len(calendar), step)]


def _benchmarks_for_periods(periods) -> dict[str, pd.DataFrame]:
    start = min(pd.Timestamp(p[1]) for p in periods) - pd.Timedelta(days=180)
    end = max(pd.Timestamp(p[2]) for p in periods) + pd.Timedelta(days=5)
    return {
        "KOSPI": fetch_index_history("KOSPI", start.strftime("%Y%m%d"), end.strftime("%Y%m%d")),
        "KOSDAQ": fetch_index_history("KOSDAQ", start.strftime("%Y%m%d"), end.strftime("%Y%m%d")),
    }


def rank_snapshot(
    candidates: list[dict],
    as_of_date,
    lookback: int,
    top_n: int = 20,
    include_flow: bool = False,
    benchmarks: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    as_of_date = pd.Timestamp(as_of_date)
    rows = []
    benchmarks = benchmarks or {}
    for item in candidates:
        df = item.get("DF")
        if df is None or df.empty:
            continue
        hist = df[df.index <= as_of_date].tail(lookback)
        fut = df[df.index > as_of_date]
        if len(hist) < 60 or fut.empty:
            continue
        benchmark_df = benchmarks.get(item["Market"])
        benchmark_hist = None if benchmark_df is None else benchmark_df[benchmark_df.index <= as_of_date].tail(max(lookback, 80))
        regime = market_regime(benchmark_hist.tail(100)) if benchmark_hist is not None and not benchmark_hist.empty else "sideways"
        res = compute_score(item["Code"], item["Name"], item["Market"], hist, lookback=lookback, regime=regime, include_flow=include_flow)
        if not res:
            continue
        rs = relative_strength_metrics(hist, benchmark_hist)
        res["RelStrength"] = round(rs["score"], 2)
        res["RelShort"] = round(rs["short"] * 100, 2)
        res["RelLong"] = round(rs["long"] * 100, 2)
        res["ScoreAdj"] = float(res["ScoreRaw"] * 0.9 + rs["score"] * 0.1)
        res["Regime"] = regime
        res["DF"] = df
        rows.append(res)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["ScoreAdj", "ScoreRaw"], ascending=False).head(top_n).reset_index(drop=True)


def walk_forward_report(
    all_syms: pd.DataFrame,
    universe: str = "ALL",
    limit: int = KOSDAQ_TOP_N_DEFAULT,
    pick_count: int = 10,
    lookback: int = 180,
    rebalance_interval: int = 20,
    trade_cfg: TradeConfig | None = None,
    periods: list[tuple[str, str, str]] | None = None,
    compare_flow: bool = False,
) -> dict:
    trade_cfg = trade_cfg or TradeConfig()
    periods = periods or DEFAULT_PERIODS
    candidates, rejected = prepare_symbol_universe(all_syms, universe=universe, limit=limit, lookback=max(lookback, 260))
    benchmarks = _benchmarks_for_periods(periods)
    benchmark_key = "KOSDAQ" if universe == "KOSDAQ" else "KOSPI"

    summary_rows = []
    details = {}
    for label, start, end in periods:
        calendar = _calendar_from_benchmarks(benchmarks, start, end)
        dates = _rebalance_dates(calendar, rebalance_interval)
        trades = []
        basket_returns = []
        overheat_before = []
        overheat_after = []
        flow_diff = []

        for as_of_date in dates:
            ranked = rank_snapshot(candidates, as_of_date, lookback, top_n=max(pick_count * 3, 20), include_flow=False, benchmarks=benchmarks)
            if ranked.empty:
                continue
            top_df = ranked.head(pick_count)
            this_returns = []
            for _, row in top_df.iterrows():
                trade = simulate_trade_after_signal(row["DF"], as_of_date, trade_cfg)
                if not trade:
                    continue
                bench = benchmarks.get(row["Market"])
                bench_trade = simulate_trade_after_signal(bench, as_of_date, trade_cfg) if bench is not None and not bench.empty else None
                trade["Code"] = row["Code"]
                trade["Name"] = row["Name"]
                trade["Score"] = row["Score"]
                trade["ScoreAdj"] = row["ScoreAdj"]
                trade["Overheat"] = row["Overheat"]
                trade["Regime"] = row["Regime"]
                trade["ScoreBucket"] = _score_bucket(row["ScoreAdj"])
                trade["ExcessRet%"] = round(trade["ret%"] - (bench_trade["ret%"] if bench_trade else 0.0), 2)
                trades.append(trade)
                this_returns.append(trade["ret%"])
            if this_returns:
                basket_returns.append(float(np.mean(this_returns)))
                overheat_before.append(float(np.mean(this_returns)))

            cool_df = ranked[ranked["Overheat"] <= 50].head(pick_count)
            cool_returns = []
            for _, row in cool_df.iterrows():
                trade = simulate_trade_after_signal(row["DF"], as_of_date, trade_cfg)
                if trade:
                    cool_returns.append(trade["ret%"])
            if cool_returns:
                overheat_after.append(float(np.mean(cool_returns)))

            if compare_flow:
                ranked_flow = rank_snapshot(candidates, as_of_date, lookback, top_n=pick_count, include_flow=True, benchmarks=benchmarks)
                flow_returns = []
                for _, row in ranked_flow.iterrows():
                    trade = simulate_trade_after_signal(row["DF"], as_of_date, trade_cfg)
                    if trade:
                        flow_returns.append(trade["ret%"])
                if flow_returns and this_returns:
                    flow_diff.append(float(np.mean(flow_returns) - np.mean(this_returns)))

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            summary_rows.append({
                "Period": label,
                "Top10AvgRet%": np.nan,
                "WinRate%": np.nan,
                "MDD%": np.nan,
                "AvgHoldDays": np.nan,
                "ProfitLossRatio": np.nan,
                "MarketExcessRet%": np.nan,
            })
            details[label] = {"trades": pd.DataFrame(), "regime_perf": pd.DataFrame(), "score_buckets": pd.DataFrame(), "compare": {}}
            continue

        wins = trades_df[trades_df["ret%"] > 0]["ret%"]
        losses = trades_df[trades_df["ret%"] <= 0]["ret%"]
        strategy_total = float(np.prod(1 + np.array(basket_returns, dtype=float) / 100.0) - 1) if basket_returns else 0.0
        market_total = _period_return(benchmarks[benchmark_key], start, end)
        summary_rows.append({
            "Period": label,
            "Top10AvgRet%": round(float(np.mean(basket_returns)) if basket_returns else 0.0, 2),
            "WinRate%": round(float((trades_df["ret%"] > 0).mean() * 100), 1),
            "MDD%": round(_basket_mdd(basket_returns), 2),
            "AvgHoldDays": round(float(trades_df["hold_days"].mean()), 2),
            "ProfitLossRatio": round(float(wins.mean() / abs(losses.mean())), 2) if len(wins) and len(losses) else np.nan,
            "MarketExcessRet%": round((strategy_total - market_total) * 100, 2),
        })
        regime_perf = trades_df.groupby("Regime")["ret%"].agg(["count", "mean", "median"]).reset_index()
        score_buckets = trades_df.groupby("ScoreBucket")["ret%"].agg(["count", "mean", "median"]).reset_index()
        details[label] = {
            "trades": trades_df.sort_values("signal_date", ascending=False).reset_index(drop=True),
            "regime_perf": regime_perf,
            "score_buckets": score_buckets,
            "compare": {
                "overheat_before_avg_ret": round(float(np.mean(overheat_before)), 2) if overheat_before else np.nan,
                "overheat_after_avg_ret": round(float(np.mean(overheat_after)), 2) if overheat_after else np.nan,
                "flow_included_minus_excluded_avg_ret": round(float(np.mean(flow_diff)), 2) if flow_diff else np.nan,
            },
        }

    return {
        "summary": pd.DataFrame(summary_rows),
        "details": details,
        "rejected": rejected,
        "candidate_count": len(candidates),
    }


def portfolio_backtest(
    all_syms: pd.DataFrame,
    start: str,
    end: str,
    universe: str = "ALL",
    limit: int = KOSDAQ_TOP_N_DEFAULT,
    lookback: int = 180,
    rebalance_interval: int = 20,
    max_positions: int = 3,
    max_weight: float = 0.30,
    initial_capital: float = 100_000_000.0,
    reentry_cooldown: int = 10,
    trade_cfg: TradeConfig | None = None,
) -> dict:
    trade_cfg = trade_cfg or TradeConfig()
    candidates, rejected = prepare_symbol_universe(all_syms, universe=universe, limit=limit, lookback=max(lookback, 260))
    benchmarks = _benchmarks_for_periods([("P", start, end)])
    benchmark_key = "KOSDAQ" if universe == "KOSDAQ" else "KOSPI"
    calendar = _calendar_from_benchmarks(benchmarks, start, end)
    rebalance_dates = set(_rebalance_dates(calendar, rebalance_interval))
    data_map = {item["Code"]: item["DF"] for item in candidates if item.get("DF") is not None}
    name_map = {item["Code"]: item["Name"] for item in candidates}
    market_map = {item["Code"]: item["Market"] for item in candidates}

    cash = float(initial_capital)
    positions = {}
    pending = []
    cooldown_until = {}
    trades = []
    equity_rows = []

    def current_equity(date):
        total = cash
        for pos in positions.values():
            df = data_map[pos["Code"]]
            hist = df[df.index <= date]
            if hist.empty:
                continue
            mark = float(hist["Close"].iloc[-1])
            total += pos["Qty"] * mark
        return total

    for idx, current_date in enumerate(calendar):
        new_pending = []
        for order in pending:
            if order["EntryDate"] != current_date:
                new_pending.append(order)
                continue
            df = data_map.get(order["Code"])
            if df is None or current_date not in df.index:
                continue
            row = df.loc[current_date]
            entry_raw = _approx_entry_price(row, trade_cfg.entry_mode)
            entry_price = entry_raw * (1 + trade_cfg.slippage_bps / 10000.0 + trade_cfg.fee_bps / 10000.0)
            alloc = min(order["Allocation"], cash)
            if alloc <= 0:
                continue
            qty = alloc / entry_price
            cash -= qty * entry_price
            positions[order["Code"]] = {
                "Code": order["Code"],
                "EntryDate": current_date,
                "EntryIdx": idx,
                "EntryPrice": entry_price,
                "Qty": qty,
                "Peak": float(row["High"]),
                "Stop": entry_raw * (1 - trade_cfg.stop_loss_pct / 100.0),
                "Take": entry_raw * (1 + trade_cfg.take_profit_pct / 100.0),
            }
        pending = new_pending

        closed_codes = []
        for code, pos in positions.items():
            df = data_map.get(code)
            if df is None or current_date not in df.index:
                continue
            row = df.loc[current_date]
            pos["Peak"] = max(pos["Peak"], float(row["High"]))
            trailing = pos["Peak"] * (1 - trade_cfg.trailing_stop_pct / 100.0)
            low = float(row["Low"])
            high = float(row["High"])
            open_ = float(row["Open"])
            exit_reason = None
            exit_raw = None
            if low <= pos["Stop"]:
                exit_reason = "stop_loss"
                exit_raw = min(open_, pos["Stop"]) if open_ < pos["Stop"] else pos["Stop"]
            elif low <= trailing:
                exit_reason = "trailing_stop"
                exit_raw = min(open_, trailing) if open_ < trailing else trailing
            elif high >= pos["Take"]:
                exit_reason = "take_profit"
                exit_raw = max(open_, pos["Take"]) if open_ > pos["Take"] else pos["Take"]
            elif idx - pos["EntryIdx"] >= trade_cfg.hold_days:
                exit_reason = "max_hold"
                exit_raw = float(row["Close"])

            if exit_reason:
                exit_price = exit_raw * (1 - trade_cfg.slippage_bps / 10000.0 - trade_cfg.fee_bps / 10000.0 - trade_cfg.tax_bps / 10000.0)
                proceeds = pos["Qty"] * exit_price
                cash += proceeds
                ret = (exit_price - pos["EntryPrice"]) / pos["EntryPrice"] * 100.0
                trades.append({
                    "Code": code,
                    "Name": name_map.get(code, code),
                    "EntryDate": str(pd.Timestamp(pos["EntryDate"]).date()),
                    "ExitDate": str(pd.Timestamp(current_date).date()),
                    "Entry": round(pos["EntryPrice"], 2),
                    "Exit": round(exit_price, 2),
                    "Ret%": round(float(ret), 2),
                    "HoldDays": max(1, (current_date - pos["EntryDate"]).days),
                    "Reason": exit_reason,
                })
                cooldown_until[code] = idx + reentry_cooldown
                closed_codes.append(code)
        for code in closed_codes:
            positions.pop(code, None)

        if current_date in rebalance_dates:
            ranked = rank_snapshot(candidates, current_date, lookback, top_n=max_positions * 5, include_flow=False, benchmarks=benchmarks)
            benchmark_hist = benchmarks[benchmark_key][benchmarks[benchmark_key].index <= current_date].tail(100)
            regime = market_regime(benchmark_hist) if not benchmark_hist.empty else "sideways"
            target_slots = max_positions if regime != "bear" else max(1, max_positions - 1)
            target_weight = max_weight if regime != "bear" else max_weight * 0.5
            open_count = len(positions) + len(pending)
            slots = max(0, target_slots - open_count)
            eq = current_equity(current_date)
            for _, row in ranked.iterrows():
                code = row["Code"]
                if slots <= 0:
                    break
                if code in positions or any(p["Code"] == code for p in pending):
                    continue
                if cooldown_until.get(code, -1) > idx:
                    continue
                df = data_map.get(code)
                if df is None:
                    continue
                next_dates = df.index[df.index > current_date]
                if len(next_dates) == 0:
                    continue
                alloc = min(eq * target_weight, cash)
                if alloc <= 0:
                    break
                pending.append({"Code": code, "EntryDate": next_dates[0], "Allocation": alloc})
                slots -= 1

        eq = current_equity(current_date)
        equity_rows.append({
            "Date": current_date,
            "Equity": eq,
            "Cash": cash,
            "CashPct": round(cash / eq * 100, 2) if eq > 0 else 0.0,
            "OpenPositions": len(positions),
        })

    if positions:
        last_date = calendar[-1]
        for code, pos in list(positions.items()):
            df = data_map.get(code)
            if df is None:
                continue
            hist = df[df.index <= last_date]
            if hist.empty:
                continue
            exit_raw = float(hist["Close"].iloc[-1])
            exit_price = exit_raw * (1 - trade_cfg.slippage_bps / 10000.0 - trade_cfg.fee_bps / 10000.0 - trade_cfg.tax_bps / 10000.0)
            proceeds = pos["Qty"] * exit_price
            cash += proceeds
            ret = (exit_price - pos["EntryPrice"]) / pos["EntryPrice"] * 100.0
            trades.append({
                "Code": code,
                "Name": name_map.get(code, code),
                "EntryDate": str(pd.Timestamp(pos["EntryDate"]).date()),
                "ExitDate": str(pd.Timestamp(last_date).date()),
                "Entry": round(pos["EntryPrice"], 2),
                "Exit": round(exit_price, 2),
                "Ret%": round(float(ret), 2),
                "HoldDays": max(1, (last_date - pos["EntryDate"]).days),
                "Reason": "final_close",
            })
            positions.pop(code, None)
        if equity_rows:
            equity_rows[-1]["Equity"] = cash
            equity_rows[-1]["Cash"] = cash
            equity_rows[-1]["CashPct"] = 100.0
            equity_rows[-1]["OpenPositions"] = 0

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)
    total_return = (cash / initial_capital - 1) * 100 if initial_capital > 0 else 0.0
    mdd = 0.0
    if not equity_df.empty:
        eq = equity_df["Equity"].astype(float)
        peak = eq.cummax()
        mdd = float(((eq / peak) - 1).min() * 100)
    wins = trades_df[trades_df["Ret%"] > 0]["Ret%"] if not trades_df.empty else pd.Series(dtype=float)
    losses = trades_df[trades_df["Ret%"] <= 0]["Ret%"] if not trades_df.empty else pd.Series(dtype=float)
    market_return = _period_return(benchmarks[benchmark_key], start, end) * 100
    summary = {
        "InitialCapital": initial_capital,
        "FinalCapital": round(float(cash), 2),
        "TotalReturn%": round(float(total_return), 2),
        "MarketReturn%": round(float(market_return), 2),
        "ExcessReturn%": round(float(total_return - market_return), 2),
        "MDD%": round(float(mdd), 2),
        "TradeCount": int(len(trades_df)),
        "WinRate%": round(float((trades_df["Ret%"] > 0).mean() * 100), 1) if not trades_df.empty else 0.0,
        "AvgHoldDays": round(float(trades_df["HoldDays"].mean()), 2) if not trades_df.empty else 0.0,
        "ProfitLossRatio": round(float(wins.mean() / abs(losses.mean())), 2) if len(wins) and len(losses) else np.nan,
        "AvgCashPct": round(float(equity_df["CashPct"].mean()), 2) if not equity_df.empty else 100.0,
    }
    return {
        "summary": summary,
        "equity_curve": equity_df,
        "trades": trades_df.sort_values("EntryDate", ascending=False).reset_index(drop=True) if not trades_df.empty else pd.DataFrame(),
        "rejected": rejected,
        "candidate_count": len(candidates),
    }
