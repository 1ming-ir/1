import json
import os

import pandas as pd
import requests


URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"


def api_key(st_module=None) -> str | None:
    if st_module is not None:
        try:
            key = st_module.secrets.get("DEEPSEEK_API_KEY")
            if key:
                return key
        except Exception:
            pass
    return os.getenv("DEEPSEEK_API_KEY")


def local_summary(board: pd.DataFrame, tuned: pd.DataFrame | None = None) -> str:
    if board.empty:
        return "目前沒有可分析的策略結果。"
    ranked = board.sort_values(["rating", "net_profit"], ascending=False).reset_index(drop=True)
    top = ranked.iloc[0]
    low_risk = ranked.sort_values(["max_dd", "net_profit"], ascending=[True, False]).iloc[0]
    text = [
        "### 回測摘要",
        f"- 目前綜合評分最高的是 **{top['strategy']}**，淨損益為 `{top['net_profit']:.2f}`，最大回撤為 `{top['max_dd']:.2f}`。",
        f"- 回撤相對較低的是 **{low_risk['strategy']}**，報酬回撤比為 `{low_risk['profit_dd']:.2f}`。",
        "- 評分同時考慮損益、勝率、最大回撤與報酬回撤比，適合用來做策略間的初步比較。",
    ]
    if tuned is not None and not tuned.empty:
        best = tuned.iloc[0]
        text.append(f"- 參數搜尋後，目前最佳設定來自 **{best['strategy']}**，參數為 `{best['best_setting']}`。")
    text.append("- 回測結果只代表歷史區間表現，不能直接視為未來獲利保證。")
    return "\n\n".join(text)


def ask_model(messages: list[dict], key: str, temperature: float) -> str:
    payload = {"model": MODEL, "temperature": temperature, "messages": messages}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    res = requests.post(URL, headers=headers, data=json.dumps(payload), timeout=40)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]


def report(board: pd.DataFrame, tuned: pd.DataFrame | None, key: str | None, temperature: float) -> str:
    if not key:
        return local_summary(board, tuned)
    extra = "" if tuned is None or tuned.empty else "\n\n最佳化摘要：\n" + tuned.to_json(orient="records", force_ascii=False)
    messages = [
        {"role": "system", "content": "你是台股程式交易課程助教。請用繁體中文客觀分析，不要保證未來獲利。"},
        {
            "role": "user",
            "content": "請根據策略回測表與最佳化摘要，整理報酬、風險、穩定性與改進方向，輸出 5 點重點。\n\n"
            + board.to_json(orient="records", force_ascii=False)
            + extra,
        },
    ]
    try:
        return ask_model(messages, key, temperature)
    except Exception as exc:
        return local_summary(board, tuned) + f"\n\nAI 服務暫時無法使用，錯誤摘要：`{exc}`"


def reply(board: pd.DataFrame, tuned: pd.DataFrame | None, history: list[dict], question: str, key: str | None, temperature: float) -> str:
    if not key:
        return "目前尚未設定 DeepSeek API key，因此無法啟用即時問答。"
    context = "策略回測表：\n" + board.to_json(orient="records", force_ascii=False)
    if tuned is not None and not tuned.empty:
        context += "\n\n最佳化摘要：\n" + tuned.to_json(orient="records", force_ascii=False)
    messages = [
        {"role": "system", "content": "你是台股策略回測助教。根據資料回答，保持簡潔客觀。"},
        {"role": "user", "content": context},
    ]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": question})
    try:
        return ask_model(messages, key, temperature)
    except Exception as exc:
        return f"AI 暫時無法回覆，錯誤摘要：`{exc}`"
