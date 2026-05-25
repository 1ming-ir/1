import pandas as pd
import streamlit as st

from coach import api_key, reply, report
from market_data import convert_cycle, read_kbar_file
from strategy_lab import METHODS, NOTES, best_table, compare, holding, tune


DATA_FILE = "data/stock_KBar_2330_2022_2024.csv.gz"
NUMERIC_COLS = ["net_profit", "avg_trade", "max_dd", "profit_factor", "profit_dd", "stability", "expectancy", "rating"]


st.set_page_config(page_title="2330 技術策略研究室", layout="wide")


@st.cache_data
def source_data() -> pd.DataFrame:
    return read_kbar_file(DATA_FILE)


def display_table(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for col in out.columns:
        if col == "win_rate":
            out[col] = out[col].map(lambda v: f"{float(v):.2%}")
        elif col in NUMERIC_COLS:
            out[col] = out[col].map(lambda v: round(float(v), 4))
    return out


raw = source_data()

st.title("2330 技術策略研究室")
st.caption("以台積電 KBar 資料比較不同技術指標策略，並透過風險調整指標與參數搜尋觀察策略表現。")

with st.sidebar:
    st.subheader("資料與交易條件")
    start_end = st.date_input(
        "資料區間",
        value=(raw["time"].min().date(), raw["time"].max().date()),
        min_value=raw["time"].min().date(),
        max_value=raw["time"].max().date(),
    )
    cycle = st.radio("KBar 週期", ["5min", "15min", "30min", "60min", "1D"], index=1)
    units = st.number_input("單次交易口數 / 張數", min_value=1, max_value=100, value=1)
    stop_points = st.number_input("停損點數", min_value=0.0, value=0.0, step=1.0)
    selected = st.multiselect("納入比較的策略", list(METHODS.keys()), default=list(METHODS.keys()))
    run_tuning = st.toggle("同時進行參數搜尋", value=False)
    temperature = st.slider("AI 回覆溫度", 0.0, 1.0, 0.3, 0.1)

if len(start_end) == 2:
    begin = pd.to_datetime(start_end[0])
    finish = pd.to_datetime(start_end[1]) + pd.Timedelta(days=1)
    use_raw = raw[(raw["time"] >= begin) & (raw["time"] < finish)].copy()
else:
    use_raw = raw.copy()

market = convert_cycle(use_raw, cycle)
if market.empty:
    st.error("目前設定的日期區間沒有資料。")
    st.stop()

base = {"units": int(units), "stop_points": float(stop_points)}
results = {name: METHODS[name](market, **base) for name in selected}
if not results:
    st.warning("請至少選擇一個策略。")
    st.stop()

board = compare(results)
hold = holding(market, units=int(units))
board_with_hold = pd.concat([board, pd.DataFrame([{"strategy": "單純買進持有", **hold.stats}])], ignore_index=True)

tuned = {}
tuned_summary = pd.DataFrame()
if run_tuning:
    tuned = {name: tune(name, market, base) for name in selected}
    tuned_summary = best_table(tuned)

top = board.iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("目前最佳策略", top["strategy"])
col2.metric("最佳策略淨損益", f"{top['net_profit']:.2f}")
col3.metric("最佳策略最大回撤", f"{top['max_dd']:.2f}")
col4.metric("資料筆數", f"{len(market):,}")

overview, strategies, optimizer, assistant, downloads = st.tabs(["總覽", "策略邏輯", "參數搜尋", "AI 分析", "匯出"])

with overview:
    st.subheader("價格走勢")
    st.line_chart(market.set_index("time")["close"], height=280)

    st.subheader("策略績效排行榜")
    st.dataframe(display_table(board_with_hold), use_container_width=True)

    curve = pd.DataFrame({"time": market["time"]})
    for name, sim in results.items():
        if not sim.curve.empty:
            curve = curve.merge(sim.curve.rename(columns={"equity": name}), on="time", how="left")
    st.subheader("權益曲線比較")
    st.line_chart(curve.set_index("time").ffill().fillna(0), height=320)

with strategies:
    st.subheader("策略訊號設計")
    st.dataframe(pd.DataFrame({"策略": list(NOTES.keys()), "設計重點": list(NOTES.values())}), use_container_width=True, hide_index=True)

    st.subheader("交易明細")
    tabs = st.tabs(list(results.keys()))
    for tab, name in zip(tabs, results.keys()):
        sim = results[name]
        with tab:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("淨損益", f"{sim.stats['net_profit']:.2f}")
            m2.metric("交易次數", sim.stats["trades"])
            m3.metric("勝率", f"{sim.stats['win_rate']:.2%}")
            m4.metric("報酬回撤比", f"{sim.stats['profit_dd']:.2f}")
            if sim.trades.empty:
                st.write("此策略在目前區間沒有完成交易。")
            else:
                st.dataframe(sim.trades.tail(100), use_container_width=True)

with optimizer:
    if not run_tuning:
        st.info("請在左側打開「同時進行參數搜尋」以檢視最佳化結果。")
    else:
        st.subheader("各策略最佳設定")
        st.dataframe(display_table(tuned_summary), use_container_width=True)
        tabs = st.tabs(list(tuned.keys()))
        for tab, name in zip(tabs, tuned.keys()):
            with tab:
                st.dataframe(display_table(tuned[name].head(20)), use_container_width=True)

with assistant:
    key = api_key(st)
    st.subheader("策略評析")
    st.markdown(report(board, tuned_summary, key, temperature))

    st.divider()
    st.subheader("針對目前結果提問")
    if "qa_log_other" not in st.session_state:
        st.session_state.qa_log_other = []
    if st.button("清空對話"):
        st.session_state.qa_log_other = []
    for message in st.session_state.qa_log_other:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("例如：哪個策略的風險較低？參數搜尋結果代表什麼？")
    if question:
        st.session_state.qa_log_other.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("整理分析中..."):
                answer = reply(board, tuned_summary, st.session_state.qa_log_other, question, key, temperature)
            st.markdown(answer)
        st.session_state.qa_log_other.append({"role": "assistant", "content": answer})

with downloads:
    st.subheader("資料匯出")
    all_trades = []
    for name, sim in results.items():
        if not sim.trades.empty:
            item = sim.trades.copy()
            item.insert(0, "strategy", name)
            all_trades.append(item)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    c1, c2, c3 = st.columns(3)
    c1.download_button("下載績效表", board_with_hold.to_csv(index=False, encoding="utf-8-sig"), "scoreboard.csv", "text/csv")
    c2.download_button("下載交易紀錄", trades.to_csv(index=False, encoding="utf-8-sig"), "trades.csv", "text/csv")
    c3.download_button("下載K棒資料", market.to_csv(index=False, encoding="utf-8-sig"), "kbars.csv", "text/csv")
    if run_tuning and not tuned_summary.empty:
        st.download_button("下載最佳參數彙整", tuned_summary.to_csv(index=False, encoding="utf-8-sig"), "best_settings.csv", "text/csv")
