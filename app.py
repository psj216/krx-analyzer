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
from core.scorer import compute_score, score_technical
from core.signals import basic_signals
from core.utils import (
    KR_TZ,
    find_fib_levels,
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
            df = df.rename(columns=rename_map)
            return market_regime(df)
    except Exception:
        pass
    return "sideways"


def _grade_label(score):
    try:
        score = float(score)
    except Exception:
        return "N/A"
    if score >= 72:
        return "매수 유력"
    if score >= 60:
        return "매수 관심"
    if score >= 48:
        return "관망/보유"
    return "주의"


def _format_eok(value):
    try:
        return f"{float(value) / 1e8:,.1f}억"
    except Exception:
        return "-"


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


def _fallback_technical(score_df, score_result):
    tech = score_result.get("Technical")
    if tech:
        return tech
    fib = score_result.get("Fib") or find_fib_levels(score_df)
    signals = score_result.get("Signals") or basic_signals(score_df)
    return score_technical(score_df, signals, fib)


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
st.subheader("오늘의 종합점수 TOP10")

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
        status_box = st.empty()
        progress = st.progress(0.0, text="점수 계산 중...")
        status_box.info("TOP10 계산 준비 중... 0/0")

        def _cb(done, total):
            progress.progress(done / max(total, 1), text=f"점수 계산 중... {done}/{total}")
            status_box.info(f"TOP10 계산 진행 중... {done}/{total}")

        result = rank_top_scores(
            all_syms,
            universe=universe,
            limit=candidate_limit,
            lookback=look_days,
            regime=cur_regime,
            progress_cb=_cb,
        )
        progress.empty()
        status_box.success(f"TOP10 계산 완료: {len(result)}/{candidate_limit}")
        st.session_state["top_rank_df"] = result
    except Exception as exc:
        st.error(f"점수 계산 오류: {exc}")

topdf = st.session_state.get("top_rank_df")
if isinstance(topdf, pd.DataFrame) and not topdf.empty:
    show = topdf.copy()
    show["Grade"] = show["Score"].apply(_grade_label)
    if "Flow" in show.columns:
        show["Flow"] = show["Flow"].apply(_fmt_score)
    if "AvgTurnover20" in show.columns:
        show["AvgTurnover20(Eok)"] = show["AvgTurnover20"].apply(lambda x: round(float(x) / 1e8, 1))
    cols = [
        "Code",
        "Name",
        "Market",
        "Score",
        "Grade",
        "Trend",
        "Momentum",
        "Flow",
        "Vol",
        "IchiSc",
        "Overheat",
        "LiqRisk",
        "AvgTurnover20(Eok)",
        "Close",
    ]
    cols = [c for c in cols if c in show.columns]
    st.dataframe(show.head(10)[cols].reset_index(drop=True), use_container_width=True, height=360)
elif isinstance(topdf, pd.DataFrame):
    st.warning("결과가 없습니다.")
else:
    st.caption("버튼을 눌러 TOP10 점수를 계산하세요.")

st.markdown("---")
st.subheader("거래대금 급등 스캔")

scan1, scan2 = st.columns(2)
with scan1:
    surge_limit = st.slider("스캔 종목 수", 50, 500, 200, 50, key="sc_lim")
with scan2:
    surge_threshold = st.slider("급등 기준(20일 평균 대비 배수)", 2.0, 10.0, 3.0, 0.5, key="sc_thr")

if st.button("급등 스캔 실행", use_container_width=True):
    progress = st.progress(0.0, text="스캔 중...")

    def _cb2(done, total):
        progress.progress(done / max(total, 1), text=f"스캔 중... {done}/{total}")

    st.session_state["surge_df"] = scan_turnover_surge(
        all_syms,
        limit=surge_limit,
        threshold=surge_threshold,
        progress_cb=_cb2,
    )
    progress.empty()

surge_df = st.session_state.get("surge_df")
if isinstance(surge_df, pd.DataFrame) and not surge_df.empty:
    st.dataframe(surge_df, use_container_width=True, height=300)
elif isinstance(surge_df, pd.DataFrame):
    st.info("조건에 맞는 종목이 없습니다.")
else:
    st.caption("버튼을 눌러 급등 스캔을 실행하세요.")

st.markdown("---")
st.subheader("포트폴리오 트래킹")

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

    signals = score_result.get("Signals") or basic_signals(score_df)
    fib = score_result.get("Fib") or find_fib_levels(score_df, lookback=min(len(score_df), 180))
    zones = suggest_trade_zones(score_df, fib)
    snapshot = indicator_snapshot(score_df)
    tech = _fallback_technical(score_df, score_result)
    flow_detail = score_result.get("FlowDetail", {})
    final_score = score_result.get("Score", 0)
    final_grade = score_result.get("Grade", _grade_label(final_score))
    flow_score = score_result.get("Flow")
    flow_display = _fmt_score(flow_score)
    vol_score = score_result.get("Vol", 50)
    ichi_score = score_result.get("IchiSc", 0)

    if signals.get("near_52w_high"):
        st.warning(f"52주 신고가 부근: {signals.get('52w_high', 0):,.0f}")
    if signals.get("near_52w_low"):
        st.info(f"52주 신저가 부근: {signals.get('52w_low', 0):,.0f}")
    if signals.get("turnover_surge"):
        st.success(f"거래대금 급증: {signals.get('turnover_ratio', 0):.1f}배")

    last_close = float(score_df["Close"].iloc[-1])
    st.subheader("종합 점수")
    m0, m1, m2, m3, m4, m5 = st.columns(6)
    if rt and rt != last_close:
        m0.metric("현재가", f"{int(rt):,}", delta=f"{(rt - last_close) / last_close * 100:+.2f}%")
    else:
        m0.metric("종가", f"{int(last_close):,}")
    m1.metric("종합점수", f"{final_score} ({final_grade})")
    m2.metric("추세", tech.get("trend_score", score_result.get("Trend", 0)))
    m3.metric("모멘텀", tech.get("momentum_score", score_result.get("Momentum", 0)))
    m4.metric("수급", flow_display)
    m5.metric("일목/거래량", f"{ichi_score}/{vol_score}")

    with st.expander("유형별 점수 상세", expanded=True):
        detail_rows = [
            {"구분": "추세", "점수": tech.get("trend_score", score_result.get("Trend", 0)), "설명": "중장기 추세, 정배열, ADX, OBV, 피보나치 위치"},
            {"구분": "모멘텀", "점수": tech.get("momentum_score", score_result.get("Momentum", 0)), "설명": "5일선, RSI, MACD, 캔들 힘, 단기 과열 여부"},
            {"구분": "수급", "점수": flow_display, "설명": f"단기 {_fmt_score(flow_detail.get('short'))} / 중기 {_fmt_score(flow_detail.get('mid'))} / 지속성 {_fmt_score(flow_detail.get('persist'))}"},
            {"구분": "거래량", "점수": vol_score, "설명": f"20일 평균 거래대금 {_format_eok(score_result.get('AvgTurnover20'))}"},
            {"구분": "일목", "점수": ichi_score, "설명": "전환선-기준선, 구름 위치, 후행스팬 확인"},
            {"구분": "과열위험", "점수": tech.get("overheat_score", score_result.get("Overheat", 0)), "설명": "참고 경고값이며 종합점수에서는 차감하지 않음"},
            {"구분": "유동성위험", "점수": tech.get("liquidity_risk", score_result.get("LiqRisk", 0)), "설명": "참고 경고값이며 종합점수에서는 차감하지 않음"},
        ]
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

        st.markdown("**추세에 반영된 이유**")
        for item in tech.get("reasons_trend", []):
            st.markdown(f"- {item}")

        st.markdown("**모멘텀에 반영된 이유**")
        for item in tech.get("reasons_momentum", []):
            st.markdown(f"- {item}")

        st.caption(f"수급 상태: {flow_detail.get('status', 'N/A')}")

    warn_lines = []
    if tech.get("overheat_score", 0) >= 50:
        warn_lines.append(
            f"과열 경고: RSI/볼린저 기준 추격 주의 ({tech.get('overheat_score', 0)})"
        )
    if tech.get("liquidity_risk", 0) >= 40:
        warn_lines.append(
            f"유동성 경고: 20일 평균 거래대금 {_format_eok(score_result.get('AvgTurnover20'))} 수준 확인 필요 ({tech.get('liquidity_risk', 0)})"
        )
    if warn_lines:
        st.subheader("보조 경고")
        for line in warn_lines:
            st.warning(line)

    mtf = multi_tf_trend(score_df, fib)
    st.subheader("다중 타임프레임")
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

    st.subheader("요약")
    st.markdown(make_narrative(code, name, market, score_df, signals, zones, fib))

    fig = make_chart(full_df, f"{code} {name}", fib)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("지표 스냅샷")
        st.dataframe(snapshot.round(2), use_container_width=True)
    with right:
        st.subheader("시그널")
        st.json(signals)

    st.subheader("매수 / 매도 구간")
    st.json(zones)

    st.markdown("---")
    st.subheader("진입가 계산기")
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
