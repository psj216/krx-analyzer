"""
core/ranking.py
Cross-sectional ranking and turnover-surge scanning.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
import random
import time
from typing import Callable, Optional

import pandas as pd

from core.data import FETCH_TIMEOUT, fetch_prices
from core.scorer import compute_score

KOSDAQ_TOP_N_DEFAULT = 1000
REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _has_required_price_columns(df: pd.DataFrame) -> bool:
    if df is None or df.empty or not all(col in df.columns for col in REQUIRED_PRICE_COLUMNS):
        return False
    return not df.dropna(subset=["Close", "Volume"]).empty


def fetch_prices_with_retry(code: str, market: str, retry_count: int = 2) -> pd.DataFrame | None:
    """Light retry around one symbol fetch for the sequential TOP10 loop."""
    attempts = max(1, retry_count)
    for attempt in range(attempts):
        try:
            df = fetch_prices(code, market)
            if _has_required_price_columns(df):
                clean = df[REQUIRED_PRICE_COLUMNS].dropna(subset=["Close", "Volume"])
                if not clean.empty:
                    return clean
        except Exception:
            pass
        if attempt < attempts - 1:
            time.sleep(random.uniform(0.2, 0.5))
    return None


def _select_symbols_for_ranking(
    all_syms: pd.DataFrame,
    universe: str,
    limit: int,
    lookback: int,
):
    syms = all_syms.copy()
    if universe == "KOSPI":
        syms = syms[syms["Market"] == "KOSPI"]
    elif universe == "KOSDAQ":
        syms = syms[syms["Market"] == "KOSDAQ"]
    syms = syms[syms["Market"].isin(["KOSPI", "KOSDAQ"])].reset_index(drop=True)

    if limit and limit > 0:
        syms = syms.head(int(limit))

    selected = [
        {
            "Code": str(row["Code"]).zfill(6),
            "Name": str(row["Name"]),
            "Market": str(row.get("Market", "KOSPI")),
        }
        for _, row in syms.iterrows()
    ]
    return selected, {"price_data_missing": 0, "score_failed": 0}


def prepare_symbol_universe(
    all_syms: pd.DataFrame,
    universe: str = "ALL",
    limit: int = KOSDAQ_TOP_N_DEFAULT,
    lookback: int = 400,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    candidates, rejected = _select_symbols_for_ranking(all_syms, universe, limit, lookback)
    return candidates, rejected


def rank_top_scores(
    all_syms: pd.DataFrame,
    universe: str = "ALL",
    limit: int = 1000,
    lookback: int = 400,
    regime: str = "sideways",
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    candidates, _ = _select_symbols_for_ranking(all_syms, universe, limit, lookback)
    rows = []
    total = len(candidates)
    excluded = {"price_data_missing": 0, "score_failed": 0}

    for i, item in enumerate(candidates, start=1):
        code = item["Code"]
        name = item["Name"]
        market = item["Market"]

        try:
            df = fetch_prices_with_retry(code, market, retry_count=2)
            if not _has_required_price_columns(df):
                excluded["price_data_missing"] += 1
                continue

            res = compute_score(code, name, market, df, lookback, regime, include_flow=True)
            if res is None:
                res = compute_score(code, name, market, df, lookback, regime, include_flow=False)

            if res:
                rows.append(res)
            else:
                excluded["score_failed"] += 1
        except Exception:
            excluded["score_failed"] += 1
        finally:
            if progress_cb:
                progress_cb(i, total)

    if not rows:
        result = pd.DataFrame(columns=["Code", "Name", "Market", "Score", "Close"])
        result.attrs["rank_stats"] = {"total": total, "scored": 0, "excluded": excluded}
        return result

    result = (
        pd.DataFrame(rows)
        .sort_values(
            ["ScoreRaw", "Score", "Trend", "Momentum", "Vol", "IchiSc", "Name"],
            ascending=[False, False, False, False, False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    result.attrs["rank_stats"] = {"total": total, "scored": len(rows), "excluded": excluded}
    return result


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
