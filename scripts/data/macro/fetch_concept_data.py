#!/usr/bin/env python3
# -u
# -*- coding: utf-8 -*-

"""
获取东方财富概念板块数据
薄封装：逻辑统一放在 tushare_tasks
"""

import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.tushare_tasks import run_concept_data_full


def main():
    run_concept_data_full(start_date="20200101", end_date=datetime.now().strftime("%Y%m%d"))


if __name__ == "__main__":
    main()
