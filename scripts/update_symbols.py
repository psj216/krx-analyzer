"""
scripts/update_symbols.py
2026-05-26 기준 KRX 심볼 캐시 갱신 스크립트.
- 상장폐지: FinanceDataReader 최신 목록에서 자동 제외
- 거래정지: 직전 거래일 거래량 = 0 인 종목 제외
- 환기종목/관리종목: 이름 패턴 필터
- 신규상장: FinanceDataReader 최신 목록에 자동 포함
"""
import re
import sys
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "krx_symbols_cache.csv"

MARKETS = ["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"]


# ── 이름 기반 필터 ─────────────────────────────────────────────
def is_spac(name: str) -> bool:
    n = str(name).strip()
    return "스팩" in n or "SPAC" in n.upper() or bool(re.search(r"\d+호$", n))


def is_preferred(name: str) -> bool:
    n = str(name).strip()
    nu = n.upper()
    if not n:
        return False
    if "우(전환)" in n:
        return True
    nu_end_candidates = ["우", "우B", "우C", "1우", "2우B", "3우C"]
    return any(nu.endswith(c.upper()) for c in nu_end_candidates) or "PREF" in nu


def is_excluded_by_name(name: str) -> bool:
    n = str(name).strip()
    keywords = ["관리종목", "투자주의", "투자경고", "투자환기", "정리매매", "불성실공시"]
    return any(kw in n for kw in keywords)


def should_exclude(name: str) -> bool:
    return is_spac(name) or is_preferred(name) or is_excluded_by_name(name)


# ── 메인 ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("KRX 심볼 캐시 갱신 (2026-05-26 기준)")
    print("=" * 60)

    # 1. 현재 상장 종목 (상장폐지 자동 제외)
    print("\nFinanceDataReader로 KRX 전체 상장 목록 조회 중...", flush=True)
    krx_all = fdr.StockListing("KRX")
    print(f"  KRX 전체: {len(krx_all)}개 (KOSPI+KOSDAQ+KONEX+ETC 포함)")

    # KOSPI / KOSDAQ 만 유지 (KOSDAQ GLOBAL → KOSDAQ 으로 정규화)
    df = krx_all[krx_all["Market"].isin(MARKETS)][["Code", "Name", "Market", "Volume"]].copy()
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    df["Market"] = df["Market"].replace("KOSDAQ GLOBAL", "KOSDAQ")
    df = df.drop_duplicates("Code").reset_index(drop=True)
    print(f"  KOSPI+KOSDAQ(+GLOBAL): {len(df)}개")

    # 2. 거래정지 종목 제거 (직전 거래일 거래량 = 0)
    suspended_mask = df["Volume"].fillna(0) == 0
    removed_suspended = df[suspended_mask][["Code", "Name", "Market"]].copy()
    df = df[~suspended_mask].reset_index(drop=True)
    print(f"\n[거래정지 필터] {len(removed_suspended)}개 제거")
    if len(removed_suspended) > 0:
        for _, row in removed_suspended.iterrows():
            print(f"  - {row['Code']} {row['Name']} ({row['Market']})")

    # 3. 표준 6자리 숫자 코드 아닌 것 제거 (alphanumeric KRX 코드 - yfinance/pykrx 미지원)
    non_numeric_mask = ~df["Code"].str.match(r"^\d{6}$")
    removed_non_numeric = df[non_numeric_mask][["Code", "Name", "Market"]].copy()
    df = df[~non_numeric_mask].reset_index(drop=True)
    if len(removed_non_numeric):
        print(f"\n[비표준 코드 필터] {len(removed_non_numeric)}개 제거 (알파뉴메릭 코드)")
        for _, row in removed_non_numeric.iterrows():
            print(f"  - {row['Code']} {row['Name']} ({row['Market']})")

    # 4. 이름 패턴 필터 (SPAC·우선주·관리·환기 등)
    df = df.drop(columns=["Volume"])
    name_mask = df["Name"].apply(should_exclude)
    removed_by_name = df[name_mask][["Code", "Name", "Market"]].copy()
    df = df[~name_mask].reset_index(drop=True)
    print(f"\n[이름 필터] {len(removed_by_name)}개 제거 (SPAC·우선주·관리종목 등)")
    if len(removed_by_name) > 0:
        for _, row in removed_by_name.iterrows():
            print(f"  - {row['Code']} {row['Name']} ({row['Market']})")

    # 4. 결과 저장
    df.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 60}")
    print(f"최종 저장: {len(df)}개 → {CACHE_PATH}")
    print("=" * 60)
    for mkt in ["KOSPI", "KOSDAQ"]:
        cnt = (df["Market"] == mkt).sum()
        print(f"  {mkt}: {cnt}개")


if __name__ == "__main__":
    main()

