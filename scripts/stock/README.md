# Stock scripts

## 月度重训与最新周信号导出

```bash
python scripts/run_job.py stock_monthly -- \
  --as-of-date 20260213 \
  --train-lookback-days 900 \
  --top-n 10 \
  --allow-rule-fallback

./venv/bin/python scripts/stock/run_stock_selector_monthly.py \
  --as-of-date 20260213 \
  --train-lookback-days 900 \
  --top-n 10 \
  --allow-rule-fallback
```

- 默认读取：`config/app/strategy_governance.yaml` 中 `seed_balance_strategy` 配置。
- 默认数据根目录：`data/tushare/`（可用 `--data-dir` 覆盖）。
- 默认输出目录：`data/signals/stock_selector/monthly/`
  - `weekly_signals_<date>.parquet`
  - `feature_importance_<date>.parquet`
  - `training_summary_<date>.json`
  - `models/stock_selector_<model_type>_<date>.*`

## 仅更新周信号（使用最近一次训练模型）

```bash
python scripts/run_job.py stock_weekly -- --top-n 10

./venv/bin/python scripts/stock/run_stock_selector_weekly_signal.py \
  --top-n 10
```

## APScheduler 调度（月度重训 + 周度信号）

```bash
python scripts/run_job.py stock_scheduler -- --mode cron

./venv/bin/python sage_app/pipelines/stock_selector_scheduler.py --mode cron
```

- 月度任务：每月第1~7个自然日的工作日触发，并在任务内校验“是否当月首个交易日”。
- 周度任务：默认每周五运行，生成最新周信号。
