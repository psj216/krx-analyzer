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
LOOKBACK = 180

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
        return "관망 보유"
    return "주의"


def _format_turnover_eok(value):
    try:
        return f"{float(value) / 1e8:,.1f}억"
    except Exception:
        return "-"


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
    favs = st.session_state["favorites"]
    if favs:
        for i, fav in enumerate(favs):
            parts = [x.strip() for x in fav.split("|")]
            c1, c2 = st.columns([5, 1], gap="small")
            with c1:
                if st.button(parts[1], key=f"fav_{i}", use_container_width=True):
                    st.session_state["sel_override"] = fav
                    st.rerun()
            with c2:
                if st.button("삭제", key=f"del_{i}"):
                    st.session_state["favorites"].pop(i)
                    save_favorites(FAV_PATH, st.session_state["favorites"])
                    st.rerun()
    else:
        st.caption("즐겨찾기 없음")

st.markdown("---")
st.subheader("🔥 오늘의 점수 TOP10")

cur_regime = _get_kospi_regime()
regime_label = {"bull": "상승장", "sideways": "횡보장", "bear": "하락장"}.get(cur_regime, "횡보장")
st.info(f"현재 장 국면: **{regime_label}**")

col1, col2, col3 = st.columns(3)
with col1:
    uni = st.selectbox("대상 시장", ["ALL", "KOSPI", "KOSDAQ"], index=0)
with col2:
    scan_n = st.slider("코스닥 후보 수", 100, 1200, 1000, 50)
with col3:
    look_days = st.slider("점수 계산 Lookback", 100, 500, 200, 10)

if st.button("점수 계산 / 업데이트", use_container_width=True):
    try:
        prog = st.progress(0.0, text="점수 계산 중...")

        def _cb(done, total):
            prog.progress(done / max(total, 1), text=f"점수 계산 중... {done}/{total}")

        result = rank_top_scores(
            all_syms,
            universe=uni,
            limit=scan_n,
            lookback=look_days,
            regime=cur_regime,
            progress_cb=_cb,
        )
        prog.empty()
        st.session_state["top_rank_df"] = result
    except Exception as e:
        st.error(f"점수 계산 오류: {e}")

topdf = st.session_state.get("top_rank_df")
if isinstance(topdf, pd.DataFrame) and not topdf.empty:
    show = topdf.copy()
    show["Grade"] = show["Score"].apply(_grade_label)
    if "AvgTurnover20" in show.columns:
        show["AvgTurnover20(Eok)"] = show["AvgTurnover20"].apply(lambda x: round(float(x) / 1e8, 1))
    disp = [
        c
        for c in [
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
        if c in show.columns
    ]
    st.dataframe(show.head(10)[disp].reset_index(drop=True), use_container_width=True, height=360)
elif isinstance(topdf, pd.DataFrame):
    st.warning("결과 없음")
else:
    st.caption("버튼을 눌러 점수를 계산하세요.")

st.markdown("---")
st.subheader("⚡ 거래대금 급등 스캐너")
sc1, sc2 = st.columns(2)
with sc1:
    sc_limit = st.slider("스캔 종목 수", 50, 500, 200, 50, key="sc_lim")
with sc2:
    sc_threshold = st.slider("급등 기준(20일 평균 대비 배수)", 2.0, 10.0, 3.0, 0.5, key="sc_thr")
if st.button("급등 스캔 실행", use_container_width=True):
    prog2 = st.progress(0.0, text="거래대금 스캔 중...")

    def _cb2(done, total):
        prog2.progress(done / max(total, 1), text=f"스캔 중... {done}/{total}")

    st.session_state["surge_df"] = scan_turnover_surge(
        all_syms, limit=sc_limit, threshold=sc_threshold, progress_cb=_cb2
    )
    prog2.empty()

surge_df = st.session_state.get("surge_df")
if isinstance(surge_df, pd.DataFrame) and not surge_df.empty:
    st.dataframe(surge_df, use_container_width=True, height=300)
elif isinstance(surge_df, pd.DataFrame):
    st.info("조건에 맞는 종목 없음")
else:
    st.caption("버튼을 눌러 스캔하세요.")

st.markdown("---")
st.subheader("💼 포트폴리오 트래킹")
port = st.session_state["portfolio"]
with st.expander("종목 추가 / 삭제"):
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        p_code = st.text_input("종목코드", key="p_code")
    with pc2:
        p_name = st.text_input("종목명", key="p_name")
    with pc3:
        p_avg = st.number_input("평균단가", min_value=0.0, step=100.0, key="p_avg")
    with pc4:
        p_qty = st.number_input("수량", min_value=0, step=1, key="p_qty")
    if st.button("추가", key="p_add"):
        if p_code and p_avg > 0 and p_qty > 0:
            port[p_code] = {"name": p_name, "avg": float(p_avg), "qty": int(p_qty)}
            save_portfolio(PORTFOLIO_PATH, port)
            st.session_state["portfolio"] = port
            st.rerun()
    del_code = st.text_input("삭제할 코드", key="p_del")
    if st.button("삭제", key="p_del_btn"):
        if del_code in port:
            del port[del_code]
            save_portfolio(PORTFOLIO_PATH, port)
            st.session_state["portfolio"] = port
            st.rerun()

if port:
    rows_p = []
    for code, info in port.items():
        cur_p = get_realtime_price(code) or info["avg"]
        pnl = (cur_p - info["avg"]) * info["qty"]
        pnl_pct = (cur_p - info["avg"]) / info["avg"] * 100 if info["avg"] else 0
        rows_p.append(
            {
                "Code": code,
                "Name": info.get("name", ""),
                "Avg": f"{info['avg']:,.0f}",
                "Current": f"{cur_p:,.0f}",
                "Qty": info["qty"],
                "PnL": f"{pnl:+,.0f}",
                "PnL%": f"{pnl_pct:+.2f}%",
            }
        )
    st.dataframe(pd.DataFrame(rows_p), use_container_width=True)
else:
    st.caption("보유 종목 없음")

st.markdown("---")


def run_once():
    if not selection:
        st.info("왼쪽에서 종목을 선택하세요.")
        return

    code, name, market = [x.strip() for x in selection.split("|")]
    st.markdown(f"### {code} · {name} ({market})")

    fav_key = f"{code} | {name} | {market}"
    is_fav = fav_key in st.session_state["favorites"]
    cf1, cf2 = st.columns([1, 5])
    with cf1:
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

    df_chart = add_indicators(df.copy())
    score_df = df.tail(look_days).copy()
    score_result = compute_score(code, name, market, score_df, lookback=look_days, regime=cur_regime, include_flow=True)
    if not score_result:
        st.warning("현재 필터 조건에서 점수를 계산할 수 없습니다.")
        return

    score_df = add_indicators(score_df)
    fib = find_fib_levels(score_df, lookback=min(LOOKBACK, max(len(score_df) - 1, 1)))
    signals = basic_signals(score_df)
    zones = suggest_trade_zones(score_df, fib)
    snap = indicator_snapshot(score_df)
    tech = score_result["Technical"]
    flow_detail = score_result.get("FlowDetail", {})
    final_sc = score_result["Score"]
    final_grade = score_result.get("Grade", _grade_label(final_sc))
    flow_sc = score_result.get("Flow", 50)
    vol_sc = score_result.get("Vol", 50)
    ichi_sc = score_result.get("IchiSc", 0)

    if signals.get("near_52w_high"):
        st.warning(f"52주 신고가 부근: {signals.get('52w_high', 0):,.0f}")
    if signals.get("near_52w_low"):
        st.info(f"52주 신저가 부근: {signals.get('52w_low', 0):,.0f}")
    if signals.get("turnover_surge"):
        st.success(f"거래대금 급증: {signals.get('turnover_ratio', 0):.1f}배")

    last_close_score = float(score_df["Close"].iloc[-1])
    st.subheader("📊 종합 점수")
    s0, s1, s2, s3, s4, s5 = st.columns(6)
    if rt and rt != last_close_score:
        s0.metric("현재가", f"{int(rt):,}", delta=f"{(rt - last_close_score) / last_close_score * 100:+.2f}%")
    else:
        s0.metric("종가", f"{int(last_close_score):,}")
    s1.metric("종합점수", f"{final_sc} ({final_grade})")
    s2.metric("추세점수", tech["trend_score"])
    s3.metric("모멘텀점수", tech["momentum_score"])
    s4.metric("수급점수", flow_sc)
    s5.metric("일목/거래량", f"{ichi_sc}/{vol_sc}")

    with st.expander("유형별 점수 상세"):
        detail_df = pd.DataFrame(
            [
                {"유형": "추세", "점수": tech["trend_score"], "설명": "이동평균, ADX, OBV, 피보나치 위치"},
                {"유형": "모멘텀", "점수": tech["momentum_score"], "설명": "RSI, MACD, 캔들, 거래대금 반응"},
                {"유형": "수급", "점수": flow_sc, "설명": f"단기 {flow_detail.get('short', 50)} / 중기 {flow_detail.get('mid', 50)} / 지속성 {flow_detail.get('persist', 50)}"},
                {"유형": "거래량", "점수": vol_sc, "설명": f"20일 평균 거래대금 {_format_turnover_eok(score_result.get('AvgTurnover20'))}"},
                {"유형": "일목", "점수": ichi_sc, "설명": "전환선-기준선, 구름, 후행스팬"},
                {"유형": "과열위험", "점수": tech["overheat_score"], "설명": "높을수록 추격 리스크"},
                {"유형": "유동성위험", "점수": tech["liquidity_risk"], "설명": "높을수록 거래 부담"},
            ]
        )
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
        st.markdown("**추세 반영 포인트**")
        for item in tech.get("reasons_trend", []):
            st.markdown(f"- {item}")
        st.markdown("**모멘텀 반영 포인트**")
        for item in tech.get("reasons_momentum", []):
            st.markdown(f"- {item}")
        st.caption(f"수급 상태: {flow_detail.get('status', 'N/A')}")

    mtf = multi_tf_trend(score_df, fib)
    st.subheader("⏱️ 다중 타임프레임")
    mtf_rows = []
    for key, label in [("short", "단기(SMA20)"), ("mid", "중기(SMA60)"), ("long", "장기(SMA120)")]:
        bz = mtf[key].get("buy_zone")
        sz = mtf[key].get("sell_zone")
        mtf_rows.append(
            {
                "구간": label,
                "추세": mtf[key].get("trend", "N/A"),
                "매수 구간": f"{bz[0]:,.0f}~{bz[1]:,.0f}" if bz else "-",
                "매도 구간": f"{sz[0]:,.0f}~{sz[1]:,.0f}" if sz else "-",
            }
        )
    st.dataframe(pd.DataFrame(mtf_rows), use_container_width=True)

    st.subheader("요약")
    st.markdown(make_narrative(code, name, market, score_df, signals, zones, fib))

    fig = make_chart(df_chart, f"{code} {name}", fib)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("지표 스냅샷")
        st.dataframe(snap.round(2), use_container_width=True)
    with c2:
        st.subheader("시그널")
        st.json(signals)
    st.subheader("매수 / 매도 구간 (교육용)")
    st.json(zones)

    st.markdown("---")
    st.subheader("🧮 진입가 계산기")
    last_close = float(df["Close"].iloc[-1])
    entry = st.number_input("진입가", min_value=0.0, step=100.0, format="%.0f", value=last_close)
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        risk_pct = st.slider("손절폭(%)", 0.5, 10.0, 3.0, 0.5)
    with rc2:
        tp_pct = st.slider("익절폭(%)", 1.0, 20.0, 5.0, 0.5)
    with rc3:
        capital = st.number_input("총자본", min_value=0.0, step=100000.0, format="%.0f")
    with rc4:
        risk_pt = st.slider("트레이드 리스크(%)", 0.5, 5.0, 1.0, 0.5)
    if entry > 0:
        sl = entry * (1 - risk_pct / 100)
        tp = entry * (1 + tp_pct / 100)
        rr = (tp - entry) / max(entry - sl, 1e-9)
        st.write(f"손절가: **{sl:,.0f}** / 익절가: **{tp:,.0f}** / 손익비: **{rr:.2f}**")
        if capital > 0:
            max_loss = capital * (risk_pt / 100)
            size = int(max_loss / max(entry - sl, 1e-9))
            st.write(f"권장 수량: **{size:,}주** (최대 허용 손실 {max_loss:,.0f})")

    st.download_button("CSV 다운로드", df.to_csv().encode("utf-8"), file_name=f"{code}.csv", mime="text/csv")


run_once()
