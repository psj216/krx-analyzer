import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from pykrx import stock

from core.charts import indicator_snapshot, make_chart, make_narrative
from core.data import fetch_prices, get_index_info, get_realtime_price, load_krx_symbols
from core.indicators import add_indicators
from core.ranking import rank_top_scores, scan_turnover_surge
from core.scorer import compute_score
from core.utils import (
    KR_TZ,
    load_favorites,
    load_portfolio,
    market_regime,
    multi_tf_trend,
    save_favorites,
    save_portfolio,
    suggest_trade_zones,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*use_container_width.*")

FAV_PATH = Path("favorites.json")
PORTFOLIO_PATH = Path("portfolio.json")
DEFAULT_LOOKBACK = 200

st.set_page_config(page_title="KRX Auto Analyzer", layout="wide", page_icon="📈")


@st.cache_data(ttl=3600 * 6)
def _load_krx_symbols_cached():
    return load_krx_symbols()


@st.cache_data(ttl=300)
def _get_index_info_cached():
    return get_index_info()


@st.cache_data(ttl=60)
def _fetch_prices_cached(code, market):
    return fetch_prices(code, market)


@st.cache_data(ttl=3600)
def _get_kospi_regime():
    try:
        today = datetime.now(KR_TZ).strftime("%Y%m%d")
        start = (datetime.now(KR_TZ) - timedelta(days=120)).strftime("%Y%m%d")
        df = stock.get_index_ohlcv_by_date(start, today, "1001")
        if df is not None and len(df) >= 60:
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
            return market_regime(df.rename(columns=rename_map))
    except Exception:
        pass
    return "sideways"


def _grade_label(score):
    try:
        score = float(score)
    except Exception:
        return "N/A"
    if score >= 72:
        return "매수 유망"
    if score >= 60:
        return "매수 관심"
    if score >= 48:
        return "관망/보유"
    return "주의"


def _fmt_score(value, default="N/A"):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return str(int(round(float(value))))
    except Exception:
        return default


def _fmt_eok(value):
    try:
        return f"{float(value) / 1e8:,.1f}억"
    except Exception:
        return "-"


def _flow_status_text(status: str) -> str:
    return {
        "full_data": "수급 정상 조회",
        "partial_data": "수급 부분 조회",
        "N/A": "수급 N/A",
    }.get(status, status or "N/A")


def _summary_text(result: dict, key: str, default: str = "근거 없음") -> str:
    summary = result.get("Summary") or {}
    value = summary.get(key)
    if value:
        return value
    reasons = (result.get("Reasons") or {}).get(key) or []
    if reasons:
        return " | ".join(str(x) for x in reasons)
    legacy_key = {
        "trend": "TrendSummary",
        "momentum": "MomentumSummary",
        "breakout": "BreakoutSummary",
        "volume": "VolumeSummary",
        "ichi": "IchiSummary",
        "risk": "RiskSummary",
    }.get(key)
    if legacy_key and result.get(legacy_key):
        return result.get(legacy_key)
    return default


st.markdown("## 📈 KRX Auto Analyzer")

with st.sidebar:
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = load_favorites(FAV_PATH)
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = load_portfolio(PORTFOLIO_PATH)
    if "sel_override" not in st.session_state:
        st.session_state["sel_override"] = None

    idx_info = _get_index_info_cached()
    for name in ["KOSPI", "KOSDAQ"]:
        item = idx_info.get(name)
        if item:
            st.metric(name, f"{item['price']:,.2f}", delta=f"{item['chg']:+.2f}%")
        else:
            st.metric(name, "조회 실패")
    if st.button("지수 새로고침", key="idx_refresh"):
        _get_index_info_cached.clear()
        st.rerun()
    debug_score = st.checkbox("Debug score", value=False)

    st.markdown("---")
    st.subheader("종목 검색")
    all_syms = _load_krx_symbols_cached()
    st.caption(f"KRX 종목 수: {len(all_syms)}")

    query = st.text_input("코드/종목명 검색")
    filtered = all_syms
    if query:
        q = query.strip().upper()
        filtered = all_syms[
            all_syms["Code"].str.upper().str.contains(q, na=False)
            | all_syms["Name"].str.upper().str.contains(q, na=False)
        ].head(50)

    selection = st.selectbox(
        "종목 선택",
        options=[""] + [f"{r.Code} | {r.Name} | {r.Market}" for _, r in filtered.iterrows()],
    )
    if st.session_state.get("sel_override"):
        selection = st.session_state["sel_override"]
        st.session_state["sel_override"] = None

    st.markdown("---")
    st.subheader("즐겨찾기")
    favorites = st.session_state["favorites"]
    if favorites:
        for idx, fav in enumerate(favorites):
            parts = [x.strip() for x in fav.split("|")]
            c1, c2 = st.columns([5, 1], gap="small")
            with c1:
                if st.button(parts[1], key=f"fav_{idx}", use_container_width=True):
                    st.session_state["sel_override"] = fav
                    st.rerun()
            with c2:
                if st.button("삭제", key=f"fav_del_{idx}"):
                    st.session_state["favorites"].pop(idx)
                    save_favorites(FAV_PATH, st.session_state["favorites"])
                    st.rerun()
    else:
        st.caption("즐겨찾기 없음")

st.markdown("---")
st.subheader("🏆 오늘의 종합점수 TOP10")

cur_regime = _get_kospi_regime()
regime_label = {"bull": "상승장", "sideways": "횡보장", "bear": "하락장"}.get(cur_regime, "횡보장")
st.info(f"현재 시장 국면: **{regime_label}**")

col1, col2, col3 = st.columns(3)
with col1:
    universe = st.selectbox("대상 시장", ["ALL", "KOSPI", "KOSDAQ"], index=0)
with col2:
    candidate_limit = st.slider("후보 수", 100, 1000, 1000, 50)
with col3:
    look_days = st.slider("점수 계산 Lookback", 100, 500, DEFAULT_LOOKBACK, 10)

if st.button("점수 계산 / 업데이트", use_container_width=True):
    try:
        progress = st.progress(0.0, text="점수 계산 중...")
        progress_total = {"value": 0}

        def _cb(done, total):
            progress_total["value"] = total
            progress.progress(done / max(total, 1), text=f"점수 계산 중... {done}/{total}")

        result = rank_top_scores(
            all_syms,
            universe=universe,
            limit=candidate_limit,
            lookback=look_days,
            regime=cur_regime,
            progress_cb=_cb,
        )
        progress.empty()
        stats = result.attrs.get("rank_stats", {}) if isinstance(result, pd.DataFrame) else {}
        final_total = int(stats.get("total") or progress_total["value"] or candidate_limit)
        scored_count = int(stats.get("scored") or len(result))
        msg = f"TOP10 계산 완료: 점수 산출 {scored_count}개 / 후보 {final_total}개"
        st.success(msg)
        st.session_state["top_rank_df"] = result
    except Exception as exc:
        st.error(f"점수 계산 오류: {exc}")

topdf = st.session_state.get("top_rank_df")
if isinstance(topdf, pd.DataFrame) and not topdf.empty:
    show = topdf.copy()
    if "Grade" not in show.columns:
        show["Grade"] = show["Score"].apply(_grade_label)
    if debug_score and "ScoreRaw" in show.columns:
        show["RankScore"] = show["ScoreRaw"].map(lambda v: round(float(v), 2) if pd.notna(v) else None)
    if "Flow" in show.columns:
        show["Flow"] = show["Flow"].apply(_fmt_score)
    cols = ["Code", "Name", "Market", "Score", "Grade"]
    if debug_score:
        cols.append("RankScore")
    cols += ["Trend", "Momentum", "Breakout", "Flow", "Vol", "IchiRaw", "Overheat", "LiqRisk", "Close"]
    cols = [c for c in cols if c in show.columns]
    rename = {"Vol": "Volume", "IchiRaw": "Ichi"}
    st.dataframe(show.head(10)[cols].rename(columns=rename).reset_index(drop=True), use_container_width=True, height=360)
elif isinstance(topdf, pd.DataFrame):
    st.warning("결과가 없습니다.")
else:
    st.caption("버튼을 눌러 TOP10 점수를 계산하세요.")

st.markdown("---")
st.subheader("🔍 거래대금 급등 스캔")
scan1, scan2 = st.columns(2)
with scan1:
    surge_limit = st.slider("스캔 종목 수", 50, 500, 200, 50, key="sc_lim")
with scan2:
    surge_threshold = st.slider("급등 기준(20일 평균 대비 배수)", 2.0, 10.0, 3.0, 0.5, key="sc_thr")

if st.button("급등 스캔 실행", use_container_width=True):
    progress = st.progress(0.0, text="스캔 중...")

    def _cb2(done, total):
        progress.progress(done / max(total, 1), text=f"스캔 중... {done}/{total}")

    st.session_state["surge_df"] = scan_turnover_surge(all_syms, limit=surge_limit, threshold=surge_threshold, progress_cb=_cb2)
    progress.empty()

surge_df = st.session_state.get("surge_df")
if isinstance(surge_df, pd.DataFrame) and not surge_df.empty:
    st.dataframe(surge_df, use_container_width=True, height=280)
elif isinstance(surge_df, pd.DataFrame):
    st.info("조건에 맞는 종목이 없습니다.")
else:
    st.caption("버튼을 눌러 급등 스캔을 실행하세요.")

st.markdown("---")
st.subheader("💼 포트폴리오 트래킹")
portfolio = st.session_state["portfolio"]
with st.expander("종목 추가 / 삭제"):
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        p_code = st.text_input("종목코드", key="p_code")
    with p2:
        p_name = st.text_input("종목명", key="p_name")
    with p3:
        p_avg = st.number_input("평균단가", min_value=0.0, step=100.0, key="p_avg")
    with p4:
        p_qty = st.number_input("수량", min_value=0, step=1, key="p_qty")
    if st.button("추가", key="p_add"):
        if p_code and p_avg > 0 and p_qty > 0:
            portfolio[p_code] = {"name": p_name, "avg": float(p_avg), "qty": int(p_qty)}
            save_portfolio(PORTFOLIO_PATH, portfolio)
            st.session_state["portfolio"] = portfolio
            st.rerun()
    del_code = st.text_input("삭제할 코드", key="p_del")
    if st.button("삭제", key="p_del_btn"):
        if del_code in portfolio:
            del portfolio[del_code]
            save_portfolio(PORTFOLIO_PATH, portfolio)
            st.session_state["portfolio"] = portfolio
            st.rerun()

if portfolio:
    rows = []
    for code, info in portfolio.items():
        current_price = get_realtime_price(code) or info["avg"]
        pnl = (current_price - info["avg"]) * info["qty"]
        pnl_pct = (current_price - info["avg"]) / info["avg"] * 100 if info["avg"] else 0
        rows.append(
            {
                "Code": code,
                "Name": info.get("name", ""),
                "Avg": f"{info['avg']:,.0f}",
                "Current": f"{current_price:,.0f}",
                "Qty": info["qty"],
                "PnL": f"{pnl:+,.0f}",
                "PnL%": f"{pnl_pct:+.2f}%",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.caption("보유 종목이 없습니다.")

st.markdown("---")


def run_once():
    if not selection:
        st.info("왼쪽에서 종목을 선택하세요.")
        return

    code, name, market = [x.strip() for x in selection.split("|")]
    st.markdown(f"### {code} | {name} ({market})")

    fav_key = f"{code} | {name} | {market}"
    is_fav = fav_key in st.session_state["favorites"]
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("즐겨찾기 해제" if is_fav else "즐겨찾기 추가", key="fav_toggle", use_container_width=True):
            if is_fav:
                st.session_state["favorites"] = [f for f in st.session_state["favorites"] if f != fav_key]
            else:
                st.session_state["favorites"] = [fav_key] + st.session_state["favorites"]
            save_favorites(FAV_PATH, st.session_state["favorites"])
            st.rerun()

    df = _fetch_prices_cached(code, market)
    rt = get_realtime_price(code)
    if df is None or df.empty:
        st.error("가격 데이터를 불러오지 못했습니다.")
        return

    score_df = add_indicators(df.tail(look_days).copy())
    full_df = add_indicators(df.copy())
    score_result = compute_score(code, name, market, score_df, lookback=look_days, regime=cur_regime, include_flow=True)
    if not score_result:
        st.warning("현재 필터 조건에서는 점수를 계산할 수 없습니다.")
        return

    signals = score_result.get("Signals", {})
    zones = suggest_trade_zones(score_df, score_result.get("Fib", {}))
    snapshot = indicator_snapshot(score_df)
    tech = score_result.get("Technical", {})
    flow_detail = score_result.get("FlowDetail", {})
    last_close = float(score_result.get("Close", score_df["Close"].iloc[-1]))
    flow_display = _fmt_score(score_result.get("Flow"))
    vol_display = _fmt_score(score_result.get("Vol"))
    breakout_display = _fmt_score(score_result.get("Breakout"))
    summary = score_result.get("Summary") or {}
    reasons = score_result.get("Reasons") or {}
    ichi_display = f"{_fmt_score(score_result.get('IchiRaw'))}/25"

    st.subheader("📊 종합 점수")
    m0, m1, m2, m3, m4, m5, m6 = st.columns(7)
    if rt and rt != last_close:
        m0.metric("현재가", f"{int(rt):,}", delta=f"{(rt - last_close) / last_close * 100:+.2f}%")
    else:
        m0.metric("종가", f"{int(last_close):,}")
    score_value = score_result.get("Score", 0)
    grade_value = score_result.get("Grade", _grade_label(score_value))
    m1.metric("Score", f"{score_value} ({grade_value})")
    m2.metric("Trend", score_result.get("Trend", 0))
    m3.metric("Momentum", score_result.get("Momentum", 0))
    m4.metric("Breakout", breakout_display)
    m5.metric("Flow", flow_display)
    m6.metric("Volume / Ichi", f"{vol_display} / {ichi_display}")

    overheat_value = score_result.get("Overheat", 0)
    liq_risk_value = score_result.get("LiqRisk", 0)
    st.caption(
        f"Overheat {overheat_value} → -{score_result.get('OverheatPenalty', 0.0):.1f} | "
        f"LiqRisk {liq_risk_value} → -{score_result.get('LiqPenalty', 0.0):.1f}"
    )
    st.markdown("**🧠 한줄판단**")
    st.write(summary.get("one_liner") or score_result.get("Judgment") or "판단 근거 없음")

    st.markdown("**📈 추세**")
    st.write(_summary_text(score_result, "trend"))
    st.markdown("**⚡ 모멘텀**")
    st.write(_summary_text(score_result, "momentum"))

    breakout_reasons = list(reasons.get("breakout") or tech.get("reasons_breakout", []))
    breakout_brief = " | ".join(breakout_reasons[:5]) if breakout_reasons else "돌파 신호 없음"
    st.markdown("**💥 돌파성**")
    st.write(breakout_brief)

    st.markdown("**☁️ 일목**")
    st.write(_summary_text(score_result, "ichi"))

    section_rows = [
        {"구분": "추세 점수", "점수": score_result.get("Trend", 0), "설명": "추세 방향 + 기울기 과도/이격 과열 반영"},
        {"구분": "모멘텀 점수", "점수": score_result.get("Momentum", 0), "설명": "5일선, RSI 티어, MACD — 과열 감점 포함"},
        {"구분": "돌파 점수", "점수": breakout_display, "설명": "거래량 배수, 캔들 품질, 윗꼬리/ATR"},
        {"구분": "수급 점수", "점수": flow_display, "설명": f"단기 {_fmt_score(flow_detail.get('short'))} / 중기 {_fmt_score(flow_detail.get('mid'))} / 지속성 {_fmt_score(flow_detail.get('persist'))}"},
        {"구분": "거래량 점수", "점수": vol_display, "설명": f"오늘/{20}일 평균 {score_result.get('VolumeRatio'):.2f}배" if score_result.get("VolumeRatio") is not None else "거래량 비율 계산 불가"},
        {"구분": "일목 점수", "점수": ichi_display, "설명": "전환선/기준선, 구름 위치, 후행스팬"},
        {"구분": "과열 보정", "점수": f"-{score_result.get('OverheatPenalty', 0.0):.2f}", "설명": f"Overheat {overheat_value}"},
        {"구분": "유동성 보정", "점수": f"-{score_result.get('LiqPenalty', 0.0):.2f}", "설명": f"LiqRisk {liq_risk_value}"},
    ]

    with st.expander("🔎 유형별 점수 상세", expanded=False):
        st.dataframe(pd.DataFrame(section_rows), use_container_width=True, hide_index=True)

        st.markdown("**💰 수급 점수**")
        st.markdown(f"- 수급 상태: {_flow_status_text(flow_detail.get('status', 'N/A'))}")
        st.markdown(f"- 최근 5일: {_fmt_score(flow_detail.get('short'))} / 최근 20일: {_fmt_score(flow_detail.get('mid'))} / 지속성: {_fmt_score(flow_detail.get('persist'))}")

        st.markdown("**📊 거래량 점수**")
        volume_ratio = score_result.get("VolumeRatio")
        if volume_ratio is not None:
            st.markdown(f"- 오늘 거래량 / 20일 평균 = {volume_ratio:.2f}배 → 거래량 점수 {score_result.get('Vol', 50)}")
        else:
            st.markdown("- 거래량 데이터가 부족해 기본 점수 처리")

        warn_lines = []
        if overheat_value >= 50:
            warn_lines.append(f"과열 경고: 추격 주의 ({overheat_value}점)")
        if liq_risk_value >= 40:
            warn_lines.append(f"유동성 경고: 평균 거래대금 {_fmt_eok(score_result.get('AvgTurnover20'))} ({liq_risk_value}점)")
        if signals.get("near_52w_high"):
            warn_lines.append(f"52주 신고가 부근 {signals.get('52w_high', 0):,.0f}")
        if signals.get("near_52w_low"):
            warn_lines.append(f"52주 신저가 부근 {signals.get('52w_low', 0):,.0f}")
        if signals.get("turnover_surge"):
            warn_lines.append(f"거래대금 급증 {signals.get('turnover_ratio', 0):.1f}배")
        if warn_lines:
            st.markdown("**🚨 보조 경고**")
            for line in warn_lines:
                st.markdown(f"- {line}")

    mtf = multi_tf_trend(score_df, score_result.get("Fib", {}))
    st.subheader("📅 다중 타임프레임")
    mtf_rows = []
    for key, label in [("short", "단기(SMA20)"), ("mid", "중기(SMA60)"), ("long", "장기(SMA120)")]:
        buy_zone = mtf[key].get("buy_zone")
        sell_zone = mtf[key].get("sell_zone")
        mtf_rows.append(
            {
                "구간": label,
                "추세": mtf[key].get("trend", "N/A"),
                "매수 구간": f"{buy_zone[0]:,.0f} ~ {buy_zone[1]:,.0f}" if buy_zone else "-",
                "매도 구간": f"{sell_zone[0]:,.0f} ~ {sell_zone[1]:,.0f}" if sell_zone else "-",
            }
        )
    st.dataframe(pd.DataFrame(mtf_rows), use_container_width=True, hide_index=True)

    st.subheader("📝 요약")
    st.markdown(make_narrative(code, name, market, score_df, signals, zones, score_result.get("Fib", {})))

    fig = make_chart(full_df, f"{code} {name}", score_result.get("Fib", {}))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("📐 지표 스냅샷")
        st.dataframe(snapshot.round(2), use_container_width=True)
    with right:
        st.subheader("🔔 시그널")
        st.json(signals)

    st.subheader("🎯 매수 / 매도 구간")
    st.json(zones)

    st.markdown("---")
    st.subheader("🧮 진입가 계산기")
    entry = st.number_input("진입가", min_value=0.0, step=100.0, format="%.0f", value=last_close)
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        risk_pct = st.slider("손절폭(%)", 0.5, 10.0, 3.0, 0.5)
    with r2:
        tp_pct = st.slider("목표수익(%)", 1.0, 20.0, 5.0, 0.5)
    with r3:
        capital = st.number_input("총자본", min_value=0.0, step=100000.0, format="%.0f")
    with r4:
        risk_budget = st.slider("트레이드 리스크(%)", 0.5, 5.0, 1.0, 0.5)
    if entry > 0:
        stop_loss = entry * (1 - risk_pct / 100)
        take_profit = entry * (1 + tp_pct / 100)
        rr = (take_profit - entry) / max(entry - stop_loss, 1e-9)
        st.write(f"손절가: **{stop_loss:,.0f}** / 목표가: **{take_profit:,.0f}** / 손익비: **{rr:.2f}**")
        if capital > 0:
            max_loss = capital * (risk_budget / 100)
            qty = int(max_loss / max(entry - stop_loss, 1e-9))
            st.write(f"권장 수량: **{qty:,}주** (최대 허용 손실 {max_loss:,.0f})")

    st.download_button("CSV 다운로드", df.to_csv().encode("utf-8"), file_name=f"{code}.csv", mime="text/csv")


run_once()
