import pandas as pd


def read_kbar_file(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, compression="infer")
    data["time"] = pd.to_datetime(data["time"])
    data = data.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["open", "high", "low", "close"])


def convert_cycle(data: pd.DataFrame, cycle: str) -> pd.DataFrame:
    if cycle == "1min":
        return data.copy()

    price = data.set_index("time")
    out = price.resample(cycle).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
            "product": "last",
        }
    )
    return out.dropna(subset=["open", "high", "low", "close"]).reset_index()
