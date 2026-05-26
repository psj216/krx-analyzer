"""
core/data.py
KRX symbol loading, market/index data, and price fetching helpers.
"""
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from pykrx import stock

from core.utils import KR_TZ, is_managed_issue, is_preferred, is_spac

warnings.filterwarnings("ignore", category=DeprecationWarning)

FETCH_TIMEOUT = 5
_SYMBOLS_CACHE = Path("krx_symbols_cache.csv")


def load_krx_symbols(cache_path: Path = _SYMBOLS_CACHE) -> pd.DataFrame:
    """Load KOSPI/KOSDAQ symbols, preferring pykrx and falling back to cache."""

    def from_cache():
        try:
            df = pd.read_csv(cache_path, encoding="utf-8-sig")
            if {"Code", "Name", "Market"}.issubset(df.columns) and len(df) > 0:
                df["Code"] = df["Code"].astype(str).str.zfill(6)
                return df
        except Exception:
            pass
        return None

    def from_pykrx():
        try:
            rows = []
            for mkt in ["KOSPI", "KOSDAQ"]:
                for code in (stock.get_market_ticker_list(market=mkt) or []):
                    rows.append({
                        "Code": str(code).zfill(6),
                        "Name": stock.get_market_ticker_name(code),
                        "Market": mkt,
                    })
            df = pd.DataFrame(rows).dropna().drop_duplicates("Code").reset_index(drop=True)
            return df if len(df) > 0 else None
        except Exception:
            return None

    def from_fdr():
        try:
            import FinanceDataReader as fdr
            raw = fdr.StockListing("KRX")
            df = raw[raw["Market"].isin(["KOSPI", "KOSDAQ"])][["Code", "Name", "Market", "Volume"]].copy()
            df["Code"] = df["Code"].astype(str).str.zfill(6)
            df = df[df["Code"].str.match(r"^\d{6}$")]
            df = df[df["Volume"].fillna(0) > 0].drop(columns=["Volume"])
            df = df.drop_duplicates("Code").reset_index(drop=True)
            return df if len(df) > 0 else None
        except Exception:
            return None

    df = from_pykrx()
    if df is None:
        df = from_fdr()
    if df is None:
        df = from_cache()
    if df is None:
        return pd.DataFrame(columns=["Code", "Name", "Market"])

    mask = ~df["Name"].apply(lambda x: is_spac(x) or is_preferred(x) or is_managed_issue(x))
    df = df[mask].reset_index(drop=True)
    try:
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass
    return df


def get_index_info() -> dict:
    """Fetch KOSPI/KOSDAQ last value and daily change."""
    result = {}
    today = datetime.now(KR_TZ).strftime("%Y%m%d")
    start = (datetime.now(KR_TZ) - timedelta(days=10)).strftime("%Y%m%d")

    for name, krx_code, yf_ticker in [("KOSPI", "1001", "^KS11"), ("KOSDAQ", "2001", "^KQ11")]:
        try:
            df = stock.get_index_ohlcv_by_date(start, today, krx_code)
            if df is not None and len(df) >= 2:
                cols = list(df.columns)
                close_col = cols[0]
                last = float(df[close_col].iloc[-1])
                prev = float(df[close_col].iloc[-2])
                result[name] = {"price": last, "chg": (last - prev) / prev * 100}
                continue
        except Exception:
            pass

        try:
            df = yf.download(yf_ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = df.dropna(subset=["Close"])
            if len(df) >= 2:
                last = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                result[name] = {"price": last, "chg": (last - prev) / prev * 100}
                continue
        except Exception:
            pass

        try:
            from bs4 import BeautifulSoup

            nv_code = "KOSPI" if name == "KOSPI" else "KOSDAQ"
            url = f"https://finance.naver.com/sise/sise_index.naver?code={nv_code}"
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            soup = BeautifulSoup(res.text, "html.parser")
            now_tag = soup.select_one("#now_value")
            prev_tag = soup.select_one("#prev_value")
            if now_tag and prev_tag:
                last = float(now_tag.text.replace(",", ""))
                prev = float(prev_tag.text.replace(",", ""))
                result[name] = {"price": last, "chg": (last - prev) / prev * 100}
                continue
        except Exception:
            pass

        result[name] = None
    return result


def market_suffix(market: str) -> str:
    m = (market or "").upper()
    if "KOSPI" in m:
        return ".KS"
    if "KOSDAQ" in m:
        return ".KQ"
    return ".KS"


def to_yf_symbol(code: str, market: str) -> str:
    return f"{str(code).zfill(6)}{market_suffix(market)}"


def _fetch_pykrx(code: str, start_krx: str, end_krx: str) -> pd.DataFrame:
    df = stock.get_market_ohlcv_by_date(start_krx, end_krx, code)
    if df is not None and not df.empty:
        df = df.copy()
        rename_map = {
            "시가": "Open",
            "고가": "High",
            "저가": "Low",
            "종가": "Close",
            "거래량": "Volume",
        }
        df = df.rename(columns=rename_map)
        if not set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns):
            first_five = list(df.columns[:5])
            df = df.rename(columns=dict(zip(first_five, ["Open", "High", "Low", "Close", "Volume"])))
        df.index = pd.to_datetime(df.index)
        return df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]].dropna()
    return pd.DataFrame()


def _fetch_yf(code: str, market: str) -> pd.DataFrame:
    sym = to_yf_symbol(code, market)
    df = yf.download(sym, period="2y", interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.index = pd.to_datetime(df.index)
    return df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]].dropna()


def fetch_prices(code: str, market: str, allow_fallback: bool = True) -> pd.DataFrame:
    start = (datetime.now(KR_TZ) - timedelta(days=730)).strftime("%Y%m%d")
    end = (datetime.now(KR_TZ) + timedelta(days=1)).strftime("%Y%m%d")
    fetchers = [(_fetch_pykrx, (code, start, end))]
    if allow_fallback:
        fetchers.append((_fetch_yf, (code, market)))
    for fn, args in fetchers:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn, *args)
            try:
                df = fut.result(timeout=FETCH_TIMEOUT)
                if df is not None and not df.empty:
                    if "Volume" in df.columns:
                        last_valid = df["Volume"].replace(0, pd.NA).last_valid_index()
                        if last_valid is not None:
                            df = df.loc[:last_valid]
                    if not df.empty:
                        return df
            except Exception:
                pass
    return pd.DataFrame()


def get_realtime_price(code: str):
    """Best-effort realtime price via Naver Finance."""
    try:
        from bs4 import BeautifulSoup

        res = requests.get(
            f"https://finance.naver.com/item/main.naver?code={code}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=3,
        )
        tag = BeautifulSoup(res.text, "html.parser").select_one("p.no_today .blind")
        if tag:
            return float(tag.text.replace(",", ""))
    except Exception:
        pass
    return None


def get_market_caps(market: str) -> dict[str, float]:
    """Fetch current market cap map keyed by 6-digit code."""
    try:
        today = datetime.now(KR_TZ).strftime("%Y%m%d")
        df = stock.get_market_cap_by_ticker(today, market=market)
        if df is None or df.empty:
            return {}
        cap_col = next((c for c in df.columns if "시가총액" in str(c)), None)
        if not cap_col:
            return {}
        return {str(idx).zfill(6): float(val) for idx, val in df[cap_col].items()}
    except Exception:
        return {}


def fetch_index_history(index_name: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV history for KOSPI/KOSDAQ benchmarks."""
    name = (index_name or "").upper()
    krx_code = "1001" if name == "KOSPI" else "2001"
    yf_ticker = "^KS11" if name == "KOSPI" else "^KQ11"
    start = start or (datetime.now(KR_TZ) - timedelta(days=730)).strftime("%Y%m%d")
    end = end or datetime.now(KR_TZ).strftime("%Y%m%d")

    try:
        df = stock.get_index_ohlcv_by_date(start, end, krx_code)
        if df is not None and not df.empty:
            df = df.copy()
            cols = list(df.columns)
            rename_map = {cols[0]: "Close"}
            if len(cols) > 1:
                rename_map[cols[1]] = "Open"
            if len(cols) > 2:
                rename_map[cols[2]] = "High"
            if len(cols) > 3:
                rename_map[cols[3]] = "Low"
            if len(cols) > 4:
                rename_map[cols[4]] = "Volume"
            df = df.rename(columns=rename_map)
            df.index = pd.to_datetime(df.index)
            return df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]].dropna()
    except Exception:
        pass

    try:
        df = yf.download(yf_ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.index = pd.to_datetime(df.index)
        return df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]].dropna()
    except Exception:
        return pd.DataFrame()
