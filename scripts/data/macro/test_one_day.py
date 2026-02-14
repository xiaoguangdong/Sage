#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.tushare_test_runner import test_one_day_one_industry


if __name__ == "__main__":
    test_one_day_one_industry()
