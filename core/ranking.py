"""
core/ranking.py
Cross-sectional ranking and turnover-surge scanning.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from datetime import datetime
from typing import Callable, Optional

import pandas as pd

from core.data import FETCH_TIMEOUT, fetch_prices, get_market_caps
from core.scorer import MIN_AVG_TURNOVER_20, MIN_PRICE, compute_score, investor_flow_score
from core.utils import KR_TZ, regime_weights

KOSDAQ_TOP_N_DEFAULT = 1000


def _prefilter_candidate(row, lookback: int):
    code = str(row["Code"]).zfill(6)
    name = str(row["Name"])
    market = str(row.get("Market", "KOSPI"))
    try:
        df = fetch_prices(code, market)
        if df is None or df.empty:
            return "no_data", None
        if len(df) < max(60, int(lookback * 0.8)):
            return "too_short", None
        recent60 = df.tail(60)
        if len(recent60) < 60:
            return "too_short", None
        if (recent60["Volume"] <= 0).any():
            return "halted", None
        close = float(df["Close"].iloc[-1])
        if close < MIN_PRICE:
            return "low_price", None
        avg_turnover20 = float((df["Close"] * df["Volume"]).tail(20).mean())
        avg_turnover5 = float((df["Close"] * df["Volume"]).tail(5).mean())
        if avg_turnover20 < MIN_AVG_TURNOVER_20:
            return "low_turnover", None
        return "ok", {
            "Code": code,
            "Name": name,
            "Market": market,
            "Close": close,
            "AvgTurnover20": avg_turnover20,
            "AvgTurnover5": avg_turnover5,
            "DF": df,
        }
    except Exception:
        return "exc", None


def _select_symbols_for_ranking(
    all_syms: pd.DataFrame,
    universe: str,
    limit: int,
    lookback: int,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    syms = all_syms.copy()
    if universe == "KOSPI":
        syms = syms[syms["Market"] == "KOSPI"]
    elif universe == "KOSDAQ":
        syms = syms[syms["Market"] == "KOSDAQ"]
    syms = syms[syms["Market"].isin(["KOSPI", "KOSDAQ"])].reset_index(drop=True)

    kospi_rows = [row for _, row in syms[syms["Market"] == "KOSPI"].iterrows()]
    kosdaq_rows = [row for _, row in syms[syms["Market"] == "KOSDAQ"].iterrows()]

    selected = []
    rejected = {"no_data": 0, "too_short": 0, "halted": 0, "low_price": 0, "low_turnover": 0, "exc": 0, "ok": 0}
    total = len(kosdaq_rows)
    done = 0
    market_caps = get_market_caps("KOSDAQ") if kosdaq_rows else {}
    kosdaq_candidates = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_prefilter_candidate, row, lookback): row for row in kosdaq_rows}
        try:
            for fut in as_completed(futures, timeout=max(total, 1) * 5 + 180):
                try:
                    status, meta = fut.result(timeout=FETCH_TIMEOUT + 3)
                    rejected[status] = rejected.get(status, 0) + 1
                    if meta:
                        meta["MarketCap"] = float(market_caps.get(meta["Code"], 0.0))
                        kosdaq_candidates.append(meta)
                except Exception:
                    rejected["exc"] += 1
                done += 1
                if progress_cb:
                    progress_cb(done, total)
        except FuturesTimeout:
            pass

    if kosdaq_candidates:
        kosdaq_df = pd.DataFrame(kosdaq_candidates)
        kosdaq_df = kosdaq_df.sort_values(["AvgTurnover20", "MarketCap", "AvgTurnover5"], ascending=False)
        kosdaq_limit = limit if limit and limit > 0 else KOSDAQ_TOP_N_DEFAULT
        kosdaq_candidates = kosdaq_df.head(kosdaq_limit).to_dict("records")

    for row in kospi_rows:
        code = str(row["Code"]).zfill(6)
        selected.append({"Code": code, "Name": str(row["Name"]), "Market": "KOSPI", "DF": None})
    selected.extend(kosdaq_candidates)
    return selected, rejected


def prepare_symbol_universe(
    all_syms: pd.DataFrame,
    universe: str = "ALL",
    limit: int = KOSDAQ_TOP_N_DEFAULT,
    lookback: int = 400,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    """Public wrapper returning deterministic candidate universe with prefetched histories when available."""
    return _select_symbols_for_ranking(all_syms, universe, limit, lookback, progress_cb)


def rank_top_scores(
    all_syms: pd.DataFrame,
    universe: str = "ALL",
    limit: int = 1000,
    lookback: int = 400,
    regime: str = "sideways",
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    candidates, _ = _select_symbols_for_ranking(all_syms, universe, limit, lookback, progress_cb)
    rows = []

    def process_one(item):
        code = item["Code"]
        name = item["Name"]
        market = item["Market"]
        df = item.get("DF")
        if df is None:
            df = fetch_prices(code, market)
        if df is None or df.empty:
            return None
        return compute_score(code, name, market, df, lookback, regime, include_flow=True)

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(process_one, item): item["Code"] for item in candidates}
        try:
            for fut in as_completed(futures, timeout=max(len(candidates), 1) * 5 + 180):
                try:
                    res = fut.result(timeout=FETCH_TIMEOUT + 5)
                    if res:
                        rows.append(res)
                except Exception:
                    pass
        except FuturesTimeout:
            pass

    if not rows:
        return pd.DataFrame(columns=["Code", "Name", "Market", "Score", "Close"])
    return pd.DataFrame(rows).sort_values("ScoreRaw", ascending=False).reset_index(drop=True)


def enrich_flow(result_df: pd.DataFrame, regime: str = "sideways", top_n: int = 50) -> pd.DataFrame:
    return result_df


def scan_turnover_surge(
    all_syms: pd.DataFrame,
    limit: int = 200,
    threshold: float = 3.0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    syms = all_syms[all_syms["Market"].isin(["KOSPI", "KOSDAQ"])].reset_index(drop=True)
    pool_limit = max(limit * 5, 500)
    candidates, _ = _select_symbols_for_ranking(syms, "ALL", pool_limit, lookback=120)

    results = []
    total = len(candidates)
    done = 0

    def check_one(item):
        code = item["Code"]
        name = item["Name"]
        market = item["Market"]
        df = item.get("DF")
        if df is None:
            df = fetch_prices(code, market)
        if df is None or len(df) < 21:
            return None
        close = float(df["Close"].iloc[-1])
        ta = float((df["Close"] * df["Volume"]).iloc[-1])
        ta_avg = float((df["Close"] * df["Volume"]).tail(20).mean())
        ratio = ta / ta_avg if ta_avg > 0 else 0
        if ratio < threshold:
            return None
        ret1d = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100 if len(df) >= 2 else 0
        return {
            "Code": code,
            "Name": name,
            "Market": market,
            "TurnoverAmtEok": round(ta / 1e8, 1),
            "AvgTurnover20Eok": round(ta_avg / 1e8, 1),
            "TurnoverRatio": round(ratio, 1),
            "Ret1D": round(float(ret1d), 2),
            "Close": round(close, 0),
        }

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(check_one, item): item["Code"] for item in candidates}
        try:
            for fut in as_completed(futures, timeout=max(total, 1) * 4 + 120):
                try:
                    res = fut.result(timeout=FETCH_TIMEOUT + 3)
                    if res:
                        results.append(res)
                except Exception:
                    pass
                done += 1
                if progress_cb:
                    progress_cb(done, total)
        except FuturesTimeout:
            pass

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values(["TurnoverRatio", "AvgTurnover20Eok"], ascending=False).head(limit).reset_index(drop=True)
