import os


def get_tushare_token(explicit: str | None = None) -> str:
    token = (explicit or os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN") or "").strip()
    if token:
        return token
    raise RuntimeError(
        "Missing Tushare token. Set env var TUSHARE_TOKEN (recommended) or TS_TOKEN."
    )

