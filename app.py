# KRX Auto Analyzer v0.4  — Streamlit UI
# 핵심 로직은 core/ 패키지에 분리됨

import warnings
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta
from pykrx import stock

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*use_container_width.*")

# ── core 임포트 ───────────────────────────────────────────────────────────
from core.utils       import (KR_TZ, load_favorites, save_favorites,
                               load_portfolio, save_portfolio,
                               market_regime, regime_weights,
                               find_fib_levels, suggest_trade_zones, multi_tf_trend)
from core.data        import (load_krx_symbols, get_index_info,
                               fetch_prices, get_realtime_price)
from core.indicators  import add_indicators
from core.signals     import basic_signals, ichimoku_signals
from core.scorer      import (score_technical, investor_flow_score,
                               volume_score, compute_score)
from core.ranking     import rank_top_scores, scan_turnover_surge, enrich_flow
from core.charts      import (indicator_snapshot, make_chart,
                               make_narrative, simple_backtest)
from core.advanced_backtests import (
    DEFAULT_PERIODS,
    TradeConfig,
    portfolio_backtest,
    relative_strength_metrics,
    walk_forward_report,
)

# ── 경로 상수 ─────────────────────────────────────────────────────────────
FAV_PATH       = Path("favorites.json")
PORTFOLIO_PATH = Path("portfolio.json")
LOOKBACK       = 180

st.set_page_config(page_title="KRX Auto Analyzer v0.4", layout="wide", page_icon="📈")

# ── 캐시 래퍼 (Streamlit 전용) ────────────────────────────────────────────
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
def _walk_forward_cached(universe, limit, pick_count, lookback, rebalance_interval,
                         hold_days, entry_mode, fee_bps, tax_bps, slippage_bps,
                         stop_loss_pct, take_profit_pct, trailing_stop_pct,
                         compare_flow):
    cfg = TradeConfig(
        hold_days=hold_days,
        entry_mode=entry_mode,
        fee_bps=fee_bps,
        tax_bps=tax_bps,
        slippage_bps=slippage_bps,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop_pct=trailing_stop_pct,
    )
    return walk_forward_report(
        _load_krx_symbols_cached(),
        universe=universe,
        limit=limit,
        pick_count=pick_count,
        lookback=lookback,
        rebalance_interval=rebalance_interval,
        trade_cfg=cfg,
        periods=DEFAULT_PERIODS,
        compare_flow=compare_flow,
    )

@st.cache_data(ttl=3600)
def _portfolio_bt_cached(start_date, end_date, universe, limit, lookback, rebalance_interval,
                         max_positions, max_weight, initial_capital, reentry_cooldown,
                         hold_days, entry_mode, fee_bps, tax_bps, slippage_bps,
                         stop_loss_pct, take_profit_pct, trailing_stop_pct):
    cfg = TradeConfig(
        hold_days=hold_days,
        entry_mode=entry_mode,
        fee_bps=fee_bps,
        tax_bps=tax_bps,
        slippage_bps=slippage_bps,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop_pct=trailing_stop_pct,
    )
    return portfolio_backtest(
        _load_krx_symbols_cached(),
        start=start_date,
        end=end_date,
        universe=universe,
        limit=limit,
        lookback=lookback,
        rebalance_interval=rebalance_interval,
        max_positions=max_positions,
        max_weight=max_weight,
        initial_capital=initial_capital,
        reentry_cooldown=reentry_cooldown,
        trade_cfg=cfg,
    )

@st.cache_data(ttl=3600)
def _get_kospi_regime():
    try:
        today = datetime.now(KR_TZ).strftime("%Y%m%d")
        start = (datetime.now(KR_TZ) - timedelta(days=100)).strftime("%Y%m%d")
        df    = stock.get_index_ohlcv_by_date(start, today, "1001")
        if df is not None and len(df) >= 60:
            df = df.rename(columns={"종가": "Close", "시가": "Open",
                                     "고가": "High",  "저가": "Low", "거래량": "Volume"})
            return market_regime(df)
    except Exception:
        pass
    return "sideways"

# ── UI ────────────────────────────────────────────────────────────────────
st.markdown("## 📈 KRX Auto Analyzer v0.4")

# ── 사이드바 ──────────────────────────────────────────────────────────────
with st.sidebar:
    if "favorites"   not in st.session_state: st.session_state["favorites"]   = load_favorites(FAV_PATH)
    if "portfolio"   not in st.session_state: st.session_state["portfolio"]   = load_portfolio(PORTFOLIO_PATH)
    if "sel_override" not in st.session_state: st.session_state["sel_override"] = None

    idx_info = _get_index_info_cached()
    for name in ["KOSPI", "KOSDAQ"]:
        ki = idx_info.get(name)
        if ki: st.metric(name, f"{ki['price']:,.2f}", delta=f"{ki['chg']:+.2f}%", delta_color="normal")
        else:  st.metric(name, "조회 중…")
    if st.button("지수 새로고침", key="idx_refresh"):
        _get_index_info_cached.clear(); st.rerun()

    st.markdown("---")
    st.subheader("종목 검색")
    all_syms = _load_krx_symbols_cached()
    st.caption(f"KRX {len(all_syms)}개 (스팩 제외)")

    query    = st.text_input("코드/종목명 검색")
    filtered = all_syms
    if query:
        q        = query.strip().upper()
        filtered = all_syms[
            all_syms["Code"].str.upper().str.contains(q, na=False) |
            all_syms["Name"].str.upper().str.contains(q, na=False)
        ].head(50)

    selection = st.selectbox("종목 선택", options=[""] + [
        f"{r.Code} | {r.Name} | {r.Market}" for _, r in filtered.iterrows()
    ])
    if st.session_state.get("sel_override"):
        selection = st.session_state["sel_override"]
        st.session_state["sel_override"] = None

    st.markdown("---")
    st.subheader("⭐ 즐겨찾기")
    favs = st.session_state["favorites"]
    if favs:
        for i, fav in enumerate(favs):
            parts  = [x.strip() for x in fav.split("|")]
            c1s, c2s = st.columns([5, 1], gap="small")
            with c1s:
                if st.button(parts[1], key=f"fav_{i}", use_container_width=True):
                    st.session_state["sel_override"] = fav; st.rerun()
            with c2s:
                if st.button("🗑", key=f"del_{i}"):
                    st.session_state["favorites"].pop(i)
                    save_favorites(FAV_PATH, st.session_state["favorites"]); st.rerun()
    else:
        st.caption("비어 있음.")

# ── TOP10 ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔥 오늘의 점수 TOP10")

cur_regime   = _get_kospi_regime()
regime_label = {"bull": "📈 상승장", "sideways": "📊 횡보장", "bear": "📉 하락장"}.get(cur_regime, "📊")
st.info(f"장 국면: **{regime_label}** (가중치 자동 조정)")

col_uni, col_n, col_days = st.columns(3)
with col_uni:  uni       = st.selectbox("대상", ["ALL", "KOSPI", "KOSDAQ"], index=0)
with col_n:    scan_n    = st.slider("스캔 종목 수", 50, 1000, 1000, 50)
with col_days: look_days = st.slider("지표 Lookback(일)", 100, 500, 200, 10)

if st.button("계산/업데이트", use_container_width=True):
    try:
        prog = st.progress(0.0, text="점수 계산 중…")

        def _cb(done, total):
            prog.progress(done / max(total, 1), text=f"점수 계산 중… {done}/{total}")

        result = rank_top_scores(
            all_syms, universe=uni, limit=scan_n,
            lookback=look_days, regime=cur_regime,
            progress_cb=_cb,
        )
        prog.empty()
        with st.spinner("수급 보완 중 (상위 50개)…"):
            result = enrich_flow(result, regime=cur_regime, top_n=50)
        st.session_state["top_rank_df"] = result
    except Exception as e:
        st.error(f"랭킹 오류: {e}")

topdf = st.session_state.get("top_rank_df")
if isinstance(topdf, pd.DataFrame) and not topdf.empty:
    def _grade(x):
        try: x = float(x)
        except: return "N/A"
        return ("매수 유망" if x >= 72 else "매수 관심" if x >= 60 else "관망/보유" if x >= 48 else "주의")
    show  = topdf.copy()
    show["Grade"] = show["Score"].apply(_grade)
    disp  = [c for c in ["Code","Name","Market","Score","Grade","Trend","Momentum","Breakout","Overheat","LiqRisk","AvgTurnover20","AvgTurnover5","Close"] if c in show.columns]
    st.dataframe(show.head(10)[disp].reset_index(drop=True), use_container_width=True, height=360)
elif isinstance(topdf, pd.DataFrame):
    st.warning("결과 없음.")
else:
    st.caption("버튼 눌러서 계산해라.")

# ── 거래대금 급등 스캐너 ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("⚡ 거래대금 급등 스캐너")
sc1, sc2 = st.columns(2)
with sc1: sc_limit     = st.slider("스캔 종목 수", 50, 500, 200, 50, key="sc_lim")
with sc2: sc_threshold = st.slider("급등 기준(20일평균 대비 배수)", 2.0, 10.0, 3.0, 0.5, key="sc_thr")
if st.button("급등 스캔 실행", use_container_width=True):
    prog2 = st.progress(0.0, text="거래대금 스캔 중…")

    def _cb2(done, total):
        prog2.progress(done / max(total, 1), text=f"스캔 중… {done}/{total}")

    st.session_state["surge_df"] = scan_turnover_surge(
        all_syms, limit=sc_limit, threshold=sc_threshold, progress_cb=_cb2
    )
    prog2.empty()

surge_df = st.session_state.get("surge_df")
if isinstance(surge_df, pd.DataFrame) and not surge_df.empty:
    st.dataframe(surge_df, use_container_width=True, height=300)
elif isinstance(surge_df, pd.DataFrame):
    st.info("조건에 맞는 종목 없음.")
else:
    st.caption("버튼 눌러서 스캔해라.")

# ── 포트폴리오 ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("💼 포트폴리오 트래킹")
port = st.session_state["portfolio"]
with st.expander("종목 추가/삭제"):
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1: p_code = st.text_input("종목코드", key="p_code")
    with pc2: p_name = st.text_input("종목명",   key="p_name")
    with pc3: p_avg  = st.number_input("평균단가", min_value=0.0, step=100.0, key="p_avg")
    with pc4: p_qty  = st.number_input("수량",    min_value=0,   step=1,     key="p_qty")
    if st.button("추가", key="p_add"):
        if p_code and p_avg > 0 and p_qty > 0:
            port[p_code] = {"name": p_name, "avg": float(p_avg), "qty": int(p_qty)}
            save_portfolio(PORTFOLIO_PATH, port)
            st.session_state["portfolio"] = port
            st.success(f"{p_code} 추가됨"); st.rerun()
    del_code = st.text_input("삭제할 코드", key="p_del")
    if st.button("삭제", key="p_del_btn"):
        if del_code in port:
            del port[del_code]
            save_portfolio(PORTFOLIO_PATH, port)
            st.session_state["portfolio"] = port; st.rerun()

if port:
    rows_p = []
    for code, info in port.items():
        cur_p   = get_realtime_price(code) or info["avg"]
        pnl     = (cur_p - info["avg"]) * info["qty"]
        pnl_pct = (cur_p - info["avg"]) / info["avg"] * 100 if info["avg"] else 0
        rows_p.append({
            "코드": code, "종목명": info.get("name", ""),
            "평균단가": f"{info['avg']:,.0f}", "현재가": f"{cur_p:,.0f}",
            "수량": info["qty"], "평가손익": f"{pnl:+,.0f}원", "수익률": f"{pnl_pct:+.2f}%",
        })
    st.dataframe(pd.DataFrame(rows_p), use_container_width=True)
else:
    st.caption("보유 종목을 추가해라.")

# ── 개별 종목 분석 ────────────────────────────────────────────────────────
st.markdown("---")

def run_once():
    if not selection:
        st.info("좌측에서 종목을 골라라.")
        return

    code, name, market = [x.strip() for x in selection.split("|")]
    st.markdown(f"### {code} — {name} ({market})")

    fav_key = f"{code} | {name} | {market}"
    is_fav  = fav_key in st.session_state["favorites"]
    cf1, cf2 = st.columns([1, 5])
    with cf1:
        if st.button("⭐ 제거" if is_fav else "⭐ 추가", key="fav_toggle", use_container_width=True):
            if is_fav:
                st.session_state["favorites"] = [f for f in st.session_state["favorites"] if f != fav_key]
            else:
                st.session_state["favorites"] = [fav_key] + st.session_state["favorites"]
            save_favorites(FAV_PATH, st.session_state["favorites"]); st.rerun()

    df = _fetch_prices_cached(code, market)
    rt = get_realtime_price(code)

    if df is None or df.empty:
        st.error("데이터 불러오기 실패."); return

    df_chart  = add_indicators(df.copy())
    score_df  = add_indicators(df.tail(look_days).copy())
    fib       = find_fib_levels(score_df, lookback=min(LOOKBACK, len(score_df) - 1))
    signals   = basic_signals(score_df)
    zones     = suggest_trade_zones(score_df, fib)
    snap      = indicator_snapshot(score_df)
    sc        = score_technical(score_df, signals, fib)
    ichi      = ichimoku_signals(score_df)
    ichi_sc   = (
        (10 if ichi.get("전환-기준") == "강세"    else 0) +
        (10 if ichi.get("구름")     == "상승구간" else 0) +
        (5  if ichi.get("후행스팬") == "상승 확인" else 0)
    )
    flow_sc, flow_raw, flow_detail = investor_flow_score(code)
    vol_sc  = volume_score(score_df)
    w       = regime_weights(cur_regime)
    if flow_sc is None:
        _fw   = w["flow"]
        _dnom = 1.0 - _fw
        _fr   = ((w["trend"]    / _dnom) * sc["trend_raw"]    +
                 (w["momentum"] / _dnom) * sc["momentum_raw"] +
                 (w["volume"]   / _dnom) * vol_sc              +
                 (w["ichi"]     / _dnom) * ichi_sc)
    else:
        _fr = (w["trend"]    * sc["trend_raw"]    +
               w["momentum"] * sc["momentum_raw"] +
               w["flow"]     * flow_sc             +
               w["volume"]   * vol_sc              +
               w["ichi"]     * ichi_sc)
    final_sc    = int(round(max(0.0, _fr - sc["overheat_score"] * 0.12 - sc["liquidity_risk"] * 0.08)))
    final_grade = ("매수 유망" if final_sc >= 72 else "매수 관심" if final_sc >= 60 else "관망/보유" if final_sc >= 48 else "주의")

    if flow_sc is None:
        st.warning("⚠️ 수급 데이터 미수신 — 수급 가중치를 나머지 지표에 재배분합니다.")
    if signals.get("near_52w_high"): st.warning(f"⚠️ 52주 신고가({signals.get('52w_high', 0):,.0f}원) 근접")
    if signals.get("near_52w_low"):  st.info(f"📌 52주 신저가({signals.get('52w_low', 0):,.0f}원) 근접")
    if signals.get("turnover_surge"): st.success(f"🔥 거래대금 급등 {signals.get('turnover_ratio', 0):.1f}배")

    last_close_score = float(score_df["Close"].iloc[-1])
    st.subheader("📊 종합 점수")
    s0, s1, s2, s3, s4, s5 = st.columns(6)
    if rt and rt != last_close_score:
        s0.metric("현재가(실시간)", f"{int(rt):,}원",
                  delta=f"{(rt - last_close_score) / last_close_score * 100:+.2f}%")
    else:
        s0.metric("종가", f"{int(last_close_score):,}원")
    s1.metric("종합점수", f"{final_sc} ({final_grade})")
    s2.metric("추세점수",   sc["trend_score"])
    s3.metric("모멘텀점수", sc["momentum_score"])
    s4.metric("돌파점수",   sc["breakout_score"])
    s5.metric("과열위험",   sc["overheat_score"])

    _fs  = flow_sc if flow_sc is not None else "N/A(미반영)"
    _fds = flow_detail.get("short")   if flow_detail.get("short")   is not None else "N/A"
    _fdm = flow_detail.get("mid")     if flow_detail.get("mid")     is not None else "N/A"
    _fdp = flow_detail.get("persist") if flow_detail.get("persist") is not None else "N/A"
    with st.expander("유형별 점수 상세"):
        st.markdown(f"""
| 유형 | 점수 | 비고 |
|---|---|---|
| 추세 | **{sc['trend_score']}** | SMA·OBV·ADX·일목 |
| 모멘텀 | **{sc['momentum_score']}** | RSI slope·MACD·거래량 맥락 |
| 돌파 | **{sc['breakout_score']}** | 장대양봉·장악형·볼밴·MACD 골든 |
| 과열 위험 ⚠ | **{sc['overheat_score']}** | 높을수록 추격 위험 |
| 저유동성 ⚠ | **{sc['liquidity_risk']}** | 높을수록 유동성 부족 |
| 수급 | **{_fs}** | 단기{_fds} / 중기{_fdm} / 지속{_fdp} |
| 거래량 | **{vol_sc}** | 오늘 vs 20일 평균 |
| 일목 | **{ichi_sc}** | 전환·구름·후행스팬 |
""")
        st.markdown("**추세:** " + " | ".join(sc.get("reasons_trend", [])))
        st.markdown("**모멘텀:** " + " | ".join(sc.get("reasons_momentum", [])))

    mtf = multi_tf_trend(score_df, fib)
    st.subheader("⏱️ 다중 타임프레임")
    mtf_rows = []
    for key, label in [("short", "단기(SMA20)"), ("mid", "중기(SMA60)"), ("long", "장기(SMA120)")]:
        row  = {"구간": label, "추세": mtf[key].get("trend", "N/A")}
        bz   = mtf[key].get("buy_zone"); sz = mtf[key].get("sell_zone")
        row["매수 후보"] = f"{bz[0]:,.0f}~{bz[1]:,.0f}" if bz else "-"
        row["매도 후보"] = f"{sz[0]:,.0f}~{sz[1]:,.0f}" if sz else "-"
        mtf_rows.append(row)
    st.dataframe(pd.DataFrame(mtf_rows), use_container_width=True)

    st.subheader("요약")
    st.markdown(make_narrative(code, name, market, score_df, signals, zones, fib))

    fig = make_chart(df_chart, f"{code} {name}", fib)
    st.plotly_chart(fig, use_container_width=True)

    c1s, c2s = st.columns([2, 1])
    with c1s: st.subheader("지표 스냅샷"); st.dataframe(snap.round(2), use_container_width=True)
    with c2s: st.subheader("시그널"); st.json(signals)
    st.subheader("매수/매도 구간 (교육용)"); st.json(zones)

    st.markdown("---")
    st.subheader("🧪 백테스트 간단 요약")
    bt1, bt2 = st.columns(2)
    with bt1: bt_signal = st.selectbox("시그널", ["golden_cross", "macd_bull_cross", "bb_breakout_up"], key="bt_sig")
    with bt2: bt_hold   = st.slider("보유일수", 1, 20, 5, 1, key="bt_hold")
    if st.button("백테스트 실행", key="bt_run", use_container_width=True):
        with st.spinner("계산 중…"):
            bt = simple_backtest(df, signal_col=bt_signal, hold_days=bt_hold)
        if bt.get("count", 0) == 0:
            st.info("시그널 발생 없음.")
        else:
            st.markdown(f"- 발생: **{bt['count']}회** / 승률: **{bt['win_rate']}%** / 평균수익: **{bt['avg_ret']}%** / 최대: {bt['max_ret']}% / 최소: {bt['min_ret']}%")
            st.dataframe(pd.DataFrame(bt["signals"]), use_container_width=True)

    st.markdown("---")
    st.subheader("🧮 진입가 계산기")
    last_close = float(df["Close"].iloc[-1])
    entry      = st.number_input("진입가(원)", min_value=0.0, step=100.0, format="%.0f", value=last_close)
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1: risk_pct = st.slider("손절폭(%)",       0.5,  10.0, 3.0, 0.5)
    with rc2: tp_pct   = st.slider("익절폭(%)",       1.0,  20.0, 5.0, 0.5)
    with rc3: capital  = st.number_input("총자본(원)", min_value=0.0, step=100000.0, format="%.0f")
    with rc4: risk_pt  = st.slider("트레이드 리스크(%)", 0.5, 5.0, 1.0, 0.5)
    if entry > 0:
        sl = entry * (1 - risk_pct / 100)
        tp = entry * (1 + tp_pct   / 100)
        rr = (tp - entry) / max(entry - sl, 1e-9)
        st.write(f"손절가: **{sl:,.0f}원** / 익절가: **{tp:,.0f}원** / 손익비: **{rr:.2f}**")
        if capital > 0:
            ml = capital * (risk_pt / 100)
            sz = int(ml / max(entry - sl, 1e-9))
            st.write(f"권장 수량: **{sz:,}주** (최대 손실 {ml:,.0f}원)")

    st.download_button("CSV 다운로드", df.to_csv().encode("utf-8"),
                       file_name=f"{code}.csv", mime="text/csv")


run_once()

st.markdown("---")
st.subheader("📈 워크포워드 리포트")
wf1, wf2, wf3, wf4 = st.columns(4)
with wf1:
    wf_universe = st.selectbox("워크포워드 대상", ["ALL", "KOSPI", "KOSDAQ"], index=0, key="wf_universe")
with wf2:
    wf_limit = st.slider("코스닥 후보 수", 200, 1200, 1000, 100, key="wf_limit")
with wf3:
    wf_pick_count = st.slider("TOP picks", 3, 20, 10, 1, key="wf_pick_count")
with wf4:
    wf_rebalance = st.slider("리밸런스 간격(거래일)", 5, 40, 20, 5, key="wf_rebalance")

wf5, wf6, wf7, wf8 = st.columns(4)
with wf5:
    wf_hold = st.slider("보유일수", 5, 40, 20, 1, key="wf_hold")
with wf6:
    wf_stop = st.slider("손절(%)", 2.0, 15.0, 7.0, 0.5, key="wf_stop")
with wf7:
    wf_take = st.slider("익절(%)", 4.0, 30.0, 12.0, 0.5, key="wf_take")
with wf8:
    wf_trail = st.slider("트레일링(%)", 3.0, 20.0, 8.0, 0.5, key="wf_trail")

wf9, wf10, wf11, wf12 = st.columns(4)
with wf9:
    wf_entry_mode = st.selectbox("진입 방식", ["open", "vwap"], index=0, key="wf_entry_mode")
with wf10:
    wf_fee = st.number_input("수수료(bps)", min_value=0.0, max_value=50.0, value=5.0, step=1.0, key="wf_fee")
with wf11:
    wf_tax = st.number_input("세금(bps)", min_value=0.0, max_value=100.0, value=15.0, step=1.0, key="wf_tax")
with wf12:
    wf_slip = st.number_input("슬리피지(bps)", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="wf_slip")

wf_compare_flow = st.checkbox("수급 포함/미포함 비교 시도", value=False, key="wf_compare_flow")
if st.button("워크포워드 실행", use_container_width=True, key="wf_run"):
    with st.spinner("워크포워드 리포트 계산 중..."):
        wf_result = _walk_forward_cached(
            wf_universe, wf_limit, wf_pick_count, look_days, wf_rebalance,
            wf_hold, wf_entry_mode, wf_fee, wf_tax, wf_slip,
            wf_stop, wf_take, wf_trail, wf_compare_flow,
        )
    st.session_state["wf_result"] = wf_result

wf_result = st.session_state.get("wf_result")
if isinstance(wf_result, dict):
    st.caption(f"후보군 수: {wf_result.get('candidate_count', 0)} / 제외 카운트: {wf_result.get('rejected', {})}")
    wf_summary = wf_result.get("summary")
    if isinstance(wf_summary, pd.DataFrame) and not wf_summary.empty:
        st.dataframe(wf_summary, use_container_width=True)
        for label, _, _ in DEFAULT_PERIODS:
            detail = (wf_result.get("details") or {}).get(label)
            if not detail:
                continue
            with st.expander(label):
                cmp = detail.get("compare", {})
                cwa, cwb, cwc = st.columns(3)
                cwa.metric("과열 필터 전 평균수익", cmp.get("overheat_before_avg_ret"))
                cwb.metric("과열 필터 후 평균수익", cmp.get("overheat_after_avg_ret"))
                cwc.metric("수급 포함-미포함", cmp.get("flow_included_minus_excluded_avg_ret"))
                if isinstance(detail.get("regime_perf"), pd.DataFrame) and not detail["regime_perf"].empty:
                    st.markdown("Regime Performance")
                    st.dataframe(detail["regime_perf"], use_container_width=True)
                if isinstance(detail.get("score_buckets"), pd.DataFrame) and not detail["score_buckets"].empty:
                    st.markdown("Score Bucket Expectancy")
                    st.dataframe(detail["score_buckets"], use_container_width=True)
                if isinstance(detail.get("trades"), pd.DataFrame) and not detail["trades"].empty:
                    st.markdown("Recent Trades")
                    st.dataframe(detail["trades"].head(50), use_container_width=True)

st.markdown("---")
st.subheader("🧺 포트폴리오 단위 백테스트")
pf1, pf2, pf3, pf4 = st.columns(4)
with pf1:
    pf_start = st.date_input("시작일", value=pd.Timestamp("2024-01-02"), key="pf_start")
with pf2:
    pf_end = st.date_input("종료일", value=pd.Timestamp(datetime.now(KR_TZ).date()), key="pf_end")
with pf3:
    pf_universe = st.selectbox("대상", ["ALL", "KOSPI", "KOSDAQ"], index=0, key="pf_universe")
with pf4:
    pf_limit = st.slider("코스닥 후보 수 ", 200, 1200, 1000, 100, key="pf_limit")

pf5, pf6, pf7, pf8 = st.columns(4)
with pf5:
    pf_rebalance = st.slider("리밸런스(거래일)", 5, 40, 20, 5, key="pf_rebalance")
with pf6:
    pf_max_positions = st.slider("동시 보유 수", 1, 10, 3, 1, key="pf_max_positions")
with pf7:
    pf_max_weight = st.slider("종목당 최대 비중", 0.1, 0.5, 0.3, 0.05, key="pf_max_weight")
with pf8:
    pf_reentry = st.slider("재진입 제한(거래일)", 0, 30, 10, 1, key="pf_reentry")

pf9, pf10, pf11, pf12 = st.columns(4)
with pf9:
    pf_capital = st.number_input("초기 자본", min_value=1000000.0, value=100000000.0, step=1000000.0, format="%.0f", key="pf_capital")
with pf10:
    pf_hold = st.slider("최대 보유일수", 5, 40, 20, 1, key="pf_hold")
with pf11:
    pf_stop = st.slider("포트폴리오 손절(개별, %)", 2.0, 15.0, 7.0, 0.5, key="pf_stop")
with pf12:
    pf_take = st.slider("포트폴리오 익절(개별, %)", 4.0, 30.0, 12.0, 0.5, key="pf_take")

pf13, pf14, pf15, pf16 = st.columns(4)
with pf13:
    pf_trail = st.slider("포트폴리오 트레일링(%)", 3.0, 20.0, 8.0, 0.5, key="pf_trail")
with pf14:
    pf_entry_mode = st.selectbox("포트폴리오 진입 방식", ["open", "vwap"], index=0, key="pf_entry_mode")
with pf15:
    pf_fee = st.number_input("포트폴리오 수수료(bps)", min_value=0.0, max_value=50.0, value=5.0, step=1.0, key="pf_fee")
with pf16:
    pf_slip = st.number_input("포트폴리오 슬리피지(bps)", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="pf_slip")

if st.button("포트폴리오 백테스트 실행", use_container_width=True, key="pf_run"):
    with st.spinner("포트폴리오 백테스트 계산 중..."):
        pf_result = _portfolio_bt_cached(
            str(pd.Timestamp(pf_start).date()), str(pd.Timestamp(pf_end).date()),
            pf_universe, pf_limit, look_days, pf_rebalance,
            pf_max_positions, pf_max_weight, pf_capital, pf_reentry,
            pf_hold, pf_entry_mode, pf_fee, 15.0, pf_slip,
            pf_stop, pf_take, pf_trail,
        )
    st.session_state["pf_result"] = pf_result

pf_result = st.session_state.get("pf_result")
if isinstance(pf_result, dict):
    pf_summary = pf_result.get("summary") or {}
    pfa, pfb, pfc, pfd, pfe, pff = st.columns(6)
    pfa.metric("총수익", f"{pf_summary.get('TotalReturn%', 0):.2f}%")
    pfb.metric("시장대비", f"{pf_summary.get('ExcessReturn%', 0):.2f}%")
    pfc.metric("MDD", f"{pf_summary.get('MDD%', 0):.2f}%")
    pfd.metric("승률", f"{pf_summary.get('WinRate%', 0):.1f}%")
    pfe.metric("거래수", f"{pf_summary.get('TradeCount', 0)}")
    pff.metric("평균 현금비중", f"{pf_summary.get('AvgCashPct', 0):.1f}%")
    st.caption(f"후보군 수: {pf_result.get('candidate_count', 0)} / 제외 카운트: {pf_result.get('rejected', {})}")
    if isinstance(pf_result.get("equity_curve"), pd.DataFrame) and not pf_result["equity_curve"].empty:
        st.line_chart(pf_result["equity_curve"].set_index("Date")[["Equity", "Cash"]])
        st.dataframe(pf_result["equity_curve"].tail(50), use_container_width=True)
    if isinstance(pf_result.get("trades"), pd.DataFrame) and not pf_result["trades"].empty:
        st.dataframe(pf_result["trades"].head(100), use_container_width=True)
