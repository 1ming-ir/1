import numpy as np
import pandas as pd


def ma(values: pd.Series, period: int) -> pd.Series:
    return values.rolling(period, min_periods=period).mean()


def ema(values: pd.Series, period: int) -> pd.Series:
    return values.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def bollinger(close: pd.Series, period: int, width: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = ma(close, period)
    spread = close.rolling(period, min_periods=period).std()
    return mid + width * spread, mid, mid - width * spread


def macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    dif = ema(close, fast) - ema(close, slow)
    dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return dif, dea, dif - dea


def kdj(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int, k_span: int, d_span: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    low_min = low.rolling(lookback, min_periods=lookback).min()
    high_max = high.rolling(lookback, min_periods=lookback).max()
    rsv = ((close - low_min) / (high_max - low_min).replace(0, np.nan) * 100).fillna(50)
    k = rsv.ewm(alpha=1 / k_span, adjust=False).mean()
    d = k.ewm(alpha=1 / d_span, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def up_cross(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left.shift(1) <= right.shift(1)) & (left > right)


def down_cross(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left.shift(1) >= right.shift(1)) & (left < right)
