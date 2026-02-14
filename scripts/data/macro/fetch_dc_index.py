#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
获取东方财富概念指数历史数据 (dc_index)
薄封装：逻辑统一放在 tushare_tasks
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.tushare_tasks import run_dc_index


def main():
    print("=" * 80)
    print("获取东方财富概念指数历史数据")
    print("=" * 80)

    run_dc_index(start_date="20200101")

    print("\n" + "=" * 80)
    print("概念指数数据获取完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
