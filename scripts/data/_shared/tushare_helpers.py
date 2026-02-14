from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pandas as pd
import tushare as ts

from scripts.data._shared.runtime import disable_proxy
from scripts.data.macro.tushare_auth import get_tushare_token


def get_pro(token: Optional[str] = None, disable_proxy_flag: bool = True):
    if disable_proxy_flag:
        disable_proxy()
    return ts.pro_api(get_tushare_token(token))


def request_with_retry(
    pro,
    api_name: str,
    params: Dict[str, Any],
    max_retries: int = 3,
    sleep_seconds: int = 40,
    backoff_seconds: int = 30,
) -> Optional[pd.DataFrame]:
    for attempt in range(max_retries):
        try:
            api_func = getattr(pro, api_name)
            df = api_func(**params)
            time.sleep(sleep_seconds)
            return df
        except Exception as exc:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * backoff_seconds
                print(f"  请求失败 (尝试 {attempt + 1}/{max_retries}): {exc}")
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"  请求失败，已达到最大重试次数: {exc}")
                return None


class PagedDownloader:
    def __init__(
        self,
        pro,
        api_name: str,
        limit: int = 3000,
        sleep_seconds: int = 40,
        max_retries: int = 3,
        backoff_seconds: int = 30,
    ) -> None:
        self.pro = pro
        self.api_name = api_name
        self.limit = limit
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def fetch_pages(self, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        offset = 0
        page = 1
        frames = []
        while True:
            page_params = dict(params)
            page_params["offset"] = offset
            print(f"  第{page}页获取 (offset={offset})...", end=" ")
            df = request_with_retry(
                self.pro,
                self.api_name,
                page_params,
                max_retries=self.max_retries,
                sleep_seconds=self.sleep_seconds,
                backoff_seconds=self.backoff_seconds,
            )
            if df is not None and not df.empty:
                print(f"成功获取 {len(df)} 条")
                frames.append(df)
                offset += len(df)
                page += 1
                if len(df) < self.limit:
                    break
            else:
                print("无数据或获取失败")
                break
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)
