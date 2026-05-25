from itertools import product

import pandas as pd

from portfolio import Simulation, blank_signal, simulate
from ta_tools import bollinger, down_cross, kdj, ma, macd, rsi, up_cross


def trend_ma(data: pd.DataFrame, fast_ma=5, slow_ma=20, units=1, stop_points=0) -> Simulation:
    d = data.copy()
    d["fast_ma"] = ma(d["close"], fast_ma)
    d["slow_ma"] = ma(d["close"], slow_ma)
    out = simulate(d, up_cross(d["fast_ma"], d["slow_ma"]), down_cross(d["fast_ma"], d["slow_ma"]), units=units, stop_points=stop_points)
    out.detail = d
    return out


def rsi_momentum(data: pd.DataFrame, short_rsi=5, long_rsi=10, units=1, stop_points=0) -> Simulation:
    d = data.copy()
    d["rsi_fast"] = rsi(d["close"], short_rsi)
    d["rsi_slow"] = rsi(d["close"], long_rsi)
    out = simulate(d, up_cross(d["rsi_fast"], d["rsi_slow"]), down_cross(d["rsi_fast"], d["rsi_slow"]), units=units, stop_points=stop_points)
    out.detail = d
    return out


def rsi_rebound(data: pd.DataFrame, period=5, low_line=20, high_line=80, units=1, stop_points=0) -> Simulation:
    d = data.copy()
    d["rsi"] = rsi(d["close"], period)
    lower = pd.Series(low_line, index=d.index)
    upper = pd.Series(high_line, index=d.index)
    out = simulate(
        d,
        up_cross(d["rsi"], lower),
        down_cross(d["rsi"], upper),
        up_cross(d["rsi"], upper),
        down_cross(d["rsi"], lower),
        units=units,
        stop_points=stop_points,
    )
    out.detail = d
    return out


def band_breakout(data: pd.DataFrame, window=20, width=2.0, units=1, stop_points=0) -> Simulation:
    d = data.copy()
    d["upper"], d["mid"], d["lower"] = bollinger(d["close"], window, width)
    out = simulate(
        d,
        up_cross(d["close"], d["upper"]),
        down_cross(d["close"], d["lower"]),
        down_cross(d["close"], d["mid"]),
        up_cross(d["close"], d["mid"]),
        units=units,
        stop_points=stop_points,
    )
    out.detail = d
    return out


def macd_cross(data: pd.DataFrame, fast=12, slow=26, signal=9, use_zero=True, units=1, stop_points=0) -> Simulation:
    d = data.copy()
    d["dif"], d["dea"], d["bar"] = macd(d["close"], fast, slow, signal)
    buy = up_cross(d["dif"], d["dea"])
    sell = down_cross(d["dif"], d["dea"])
    if use_zero:
        buy = buy & (d["dif"] > 0) & (d["dea"] > 0)
        sell = sell & (d["dif"] < 0) & (d["dea"] < 0)
    out = simulate(d, buy, sell, units=units, stop_points=stop_points)
    out.detail = d
    return out


def kdj_turn(data: pd.DataFrame, lookback=9, k_span=3, d_span=3, filter_zone=True, low_line=20, high_line=80, units=1, stop_points=0) -> Simulation:
    d = data.copy()
    d["k"], d["d"], d["j"] = kdj(d["high"], d["low"], d["close"], lookback, k_span, d_span)
    buy = up_cross(d["k"], d["d"])
    sell = down_cross(d["k"], d["d"])
    if filter_zone:
        buy = buy & (d["k"] < low_line)
        sell = sell & (d["k"] > high_line)
    out = simulate(d, buy, sell, units=units, stop_points=stop_points)
    out.detail = d
    return out


METHODS = {
    "均線趨勢": trend_ma,
    "RSI 動能": rsi_momentum,
    "RSI 反轉": rsi_rebound,
    "布林突破": band_breakout,
    "MACD 交叉": macd_cross,
    "KDJ 轉折": kdj_turn,
}


NOTES = {
    "均線趨勢": "以短均線與長均線的交叉辨識趨勢方向，適合捕捉中短期趨勢轉換。",
    "RSI 動能": "比較短週期與長週期 RSI，短週期 RSI 較強時視為動能轉強。",
    "RSI 反轉": "利用 RSI 超買超賣區間判斷反彈與回落，偏向逆勢交易。",
    "布林突破": "用布林通道上下軌判斷價格突破，並以中軌作為離場參考。",
    "MACD 交叉": "DIF 與 DEA 黃金交叉偏多、死亡交叉偏空，並加入零軸方向過濾。",
    "KDJ 轉折": "K、D 線交叉搭配超買超賣區間，觀察短線動能轉折。",
}


SEARCH_SPACE = {
    "均線趨勢": {"fast_ma": [3, 5, 8, 10], "slow_ma": [15, 20, 30, 40]},
    "RSI 動能": {"short_rsi": [3, 5, 7], "long_rsi": [10, 14, 21]},
    "RSI 反轉": {"period": [5, 7, 10, 14], "low_line": [20, 25, 30], "high_line": [70, 75, 80]},
    "布林突破": {"window": [10, 20, 30, 60], "width": [1.5, 2.0, 2.5]},
    "MACD 交叉": {"fast": [8, 12, 15], "slow": [20, 26, 35], "signal": [7, 9, 12]},
    "KDJ 轉折": {"lookback": [5, 9, 14], "k_span": [3, 5], "d_span": [3, 5]},
}


def compare(results: dict[str, Simulation]) -> pd.DataFrame:
    rows = []
    for name, sim in results.items():
        row = {"strategy": name}
        row.update(sim.stats)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rating", "net_profit"], ascending=False)


def tune(name: str, data: pd.DataFrame, base: dict, limit: int = 100) -> pd.DataFrame:
    grid = SEARCH_SPACE[name]
    keys = list(grid.keys())
    rows = []
    runs = 0
    for values in product(*[grid[k] for k in keys]):
        params = dict(base)
        params.update(dict(zip(keys, values)))
        if name == "均線趨勢" and params["fast_ma"] >= params["slow_ma"]:
            continue
        if name == "RSI 動能" and params["short_rsi"] >= params["long_rsi"]:
            continue
        if name == "MACD 交叉" and params["fast"] >= params["slow"]:
            continue
        sim = METHODS[name](data, **params)
        row = {"strategy": name, **{k: params[k] for k in keys}, **sim.stats}
        rows.append(row)
        runs += 1
        if runs >= limit:
            break
    return pd.DataFrame(rows).sort_values(["rating", "net_profit"], ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def best_table(tuned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    metrics = {"net_profit", "trades", "win_rate", "avg_trade", "max_dd", "profit_factor", "profit_dd", "stability", "expectancy", "rating"}
    rows = []
    for name, df in tuned.items():
        if df.empty:
            continue
        best = df.iloc[0].to_dict()
        params = {k: v for k, v in best.items() if k not in metrics and k != "strategy"}
        rows.append({"strategy": name, "best_setting": str(params), **{k: best[k] for k in metrics}})
    return pd.DataFrame(rows).sort_values(["rating", "net_profit"], ascending=False) if rows else pd.DataFrame()


def holding(data: pd.DataFrame, units=1) -> Simulation:
    buy = pd.Series([True] + [False] * (len(data) - 1), index=data.index)
    return simulate(data, buy, blank_signal(data), units=units, allow_short=False)
