import os

try:
    from scripts.data._shared.runtime import load_env_file
except Exception:
    load_env_file = None


def get_tushare_token(explicit: str | None = None) -> str:
    if load_env_file is not None:
        load_env_file()
    token = (explicit or os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN") or "").strip()
    if token:
        return token
    raise RuntimeError(
        "Missing Tushare token. Set env var TUSHARE_TOKEN (recommended) or TS_TOKEN."
    )
