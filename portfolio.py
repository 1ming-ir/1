from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Simulation:
    trades: pd.DataFrame
    curve: pd.DataFrame
    stats: dict
    detail: pd.DataFrame


def simulate(
    data: pd.DataFrame,
    long_signal: pd.Series,
    short_signal: pd.Series,
    long_exit: pd.Series | None = None,
    short_exit: pd.Series | None = None,
    units: int = 1,
    stop_points: float = 0,
    allow_short: bool = True,
) -> Simulation:
    long_exit = short_signal if long_exit is None else long_exit
    short_exit = long_signal if short_exit is None else short_exit
    pos = 0
    entry_price = 0.0
    entry_time = None
    side = ""
    records = []
    curve = []
    realized = 0.0

    for i in range(len(data) - 1):
        now = data.iloc[i]
        nxt = data.iloc[i + 1]
        close = float(now["close"])

        if pos > 0:
            stop_hit = stop_points > 0 and close <= entry_price - stop_points
            if bool(long_exit.iloc[i]) or stop_hit:
                px = close if stop_hit else float(nxt["open"])
                pnl = (px - entry_price) * units
                realized += pnl
                records.append(_record(side, entry_time, entry_price, now["time"] if stop_hit else nxt["time"], px, units, pnl))
                pos = 0
        elif pos < 0:
            stop_hit = stop_points > 0 and close >= entry_price + stop_points
            if bool(short_exit.iloc[i]) or stop_hit:
                px = close if stop_hit else float(nxt["open"])
                pnl = (entry_price - px) * units
                realized += pnl
                records.append(_record(side, entry_time, entry_price, now["time"] if stop_hit else nxt["time"], px, units, pnl))
                pos = 0

        if pos == 0:
            if bool(long_signal.iloc[i]):
                pos = units
                entry_price = float(nxt["open"])
                entry_time = nxt["time"]
                side = "Buy"
            elif allow_short and bool(short_signal.iloc[i]):
                pos = -units
                entry_price = float(nxt["open"])
                entry_time = nxt["time"]
                side = "Sell"

        floating = 0.0
        if pos > 0:
            floating = (close - entry_price) * units
        elif pos < 0:
            floating = (entry_price - close) * units
        curve.append({"time": now["time"], "equity": realized + floating})

    if pos and len(data):
        last = data.iloc[-1]
        px = float(last["close"])
        pnl = ((px - entry_price) if pos > 0 else (entry_price - px)) * units
        realized += pnl
        records.append(_record(side, entry_time, entry_price, last["time"], px, units, pnl))
        curve.append({"time": last["time"], "equity": realized})

    trades = pd.DataFrame(records)
    equity = pd.DataFrame(curve)
    return Simulation(trades, equity, scorecard(trades, equity), data)


def _record(side, entry_time, entry_price, exit_time, exit_price, units, pnl):
    return {
        "side": side,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "quantity": units,
        "profit": pnl,
        "return_pct": pnl / (entry_price * units) if entry_price and units else 0,
    }


def scorecard(trades: pd.DataFrame, curve: pd.DataFrame) -> dict:
    if trades.empty:
        return {k: 0.0 for k in ["net_profit", "trades", "win_rate", "avg_trade", "max_dd", "profit_factor", "profit_dd", "stability", "expectancy", "rating"]}

    pnl = trades["profit"].astype(float)
    returns = trades["return_pct"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    net = float(pnl.sum())
    win_rate = float((pnl > 0).mean())
    if curve.empty:
        max_dd = 0.0
    else:
        max_dd = float((curve["equity"].cummax() - curve["equity"]).max())
    gross_win = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    pf = gross_win / gross_loss if gross_loss else (gross_win if gross_win else 0)
    profit_dd = net / max_dd if max_dd else (net if net > 0 else 0)
    stability = float(returns.mean() / returns.std(ddof=0)) if len(returns) > 1 and returns.std(ddof=0) else 0
    avg_win = float(wins.mean()) if len(wins) else 0
    avg_loss = abs(float(losses.mean())) if len(losses) else 0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    rating = net - 0.5 * max_dd + 100 * (win_rate - 0.5) + 10 * profit_dd
    return {
        "net_profit": net,
        "trades": int(len(trades)),
        "win_rate": win_rate,
        "avg_trade": float(pnl.mean()),
        "max_dd": max_dd,
        "profit_factor": pf,
        "profit_dd": float(profit_dd),
        "stability": stability,
        "expectancy": expectancy,
        "rating": float(rating),
    }


def blank_signal(data: pd.DataFrame) -> pd.Series:
    return pd.Series(np.zeros(len(data), dtype=bool), index=data.index)
