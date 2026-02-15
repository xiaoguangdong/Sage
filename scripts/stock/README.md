# Stock scripts

## 月度重训与最新周信号导出

```bash
./venv/bin/python scripts/stock/run_stock_selector_monthly.py \
  --as-of-date 20260213 \
  --train-lookback-days 900 \
  --top-n 10 \
  --allow-rule-fallback
```

- 默认读取：`sage_app/config/strategy_governance.yaml` 中 `seed_balance_strategy` 配置。
- 默认数据根目录：`data/tushare/`（可用 `--data-dir` 覆盖）。
- 默认输出目录：`data/signals/stock_selector/monthly/`
  - `weekly_signals_<date>.parquet`
  - `feature_importance_<date>.parquet`
  - `training_summary_<date>.json`
  - `models/stock_selector_<model_type>_<date>.*`
