#!/usr/bin/env python3
# -u
# -*- coding: utf-8 -*-

"""
获取Tushare概念板块数据
使用dc_index、dc_member、dc_daily接口
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
