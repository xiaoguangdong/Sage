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
  --valid-days 120 \
  --eval-top-n 10 \
  --top-n 10 \
  --allow-rule-fallback
```

- 默认读取：`config/app/strategy_governance.yaml` 中 `seed_balance_strategy` 配置。
- 默认数据根目录：`data/tushare/`（可用 `--data-dir` 覆盖）。
- 默认输出目录：`data/signals/stock_selector/monthly/`
  - `weekly_signals_<date>.parquet`
  - `feature_importance_<date>.parquet`
  - `training_summary_<date>.json`
  - `validation_predictions_<date>.parquet`（启用留出验证时）
  - `models/stock_selector_<model_type>_<date>.*`

## 仅更新周信号（使用最近一次训练模型）

```bash
python scripts/run_job.py stock_weekly -- --top-n 10

./venv/bin/python scripts/stock/run_stock_selector_weekly_signal.py \
  --top-n 10
```

## 选股模型基准对比（Rule / LGBM / XGB）

```bash
python scripts/run_job.py stock_benchmark -- \
  --as-of-date 20260213 \
  --train-lookback-days 900 \
  --valid-days 120 \
  --eval-top-n 10 \
  --models rule,lgbm,xgb

./venv/bin/python scripts/stock/run_stock_model_benchmark.py \
  --as-of-date 20260213 \
  --train-lookback-days 900 \
  --valid-days 120 \
  --eval-top-n 10 \
  --models rule,lgbm,xgb
```

- 默认输出目录：`data/backtest/stock_selector/benchmark/`
  - `stock_model_benchmark_<run_id>.json`
  - `stock_model_benchmark_<run_id>.parquet`
  - `stock_model_benchmark_latest.json`

## APScheduler 调度（月度重训 + 周度信号）

```bash
python scripts/run_job.py stock_scheduler -- --mode cron

./venv/bin/python sage_app/pipelines/stock_selector_scheduler.py --mode cron
```

- 月度任务：每月第1~7个自然日的工作日触发，并在任务内校验“是否当月首个交易日”。
- 周度任务：默认每周五运行，生成最新周信号。

## 券商执行入口（平安证券预留）

```bash
python scripts/run_job.py broker_submit -- --broker pingan

./venv/bin/python scripts/stock/run_broker_execution.py \
  --broker pingan \
  --top-n 10
```

- 默认模式是 `dry-run`，输出提交回执：
  - `data/signals/portfolio/broker_submit_<YYYYMMDD_HHMMSS>.json`
- 使用 `--submit` 会尝试实盘提交；当前 PingAn 适配器为预留 stub，会显式报未实现。

## 统一信号契约（执行层单入口）

- 选股治理脚本会额外输出统一契约：
  - `data/signals/stock_selector/contracts/stock_signal_contract_<trade_date>.parquet`
  - `data/signals/stock_selector/contracts/stock_signal_contract_latest.parquet`
- 周度主流程会基于 `stock_signal_contract` + `industry_signal_snapshot_latest.parquet` 生成执行信号：
  - `data/signals/stock_selector/contracts/execution_signals_<trade_date>.parquet`
  - `run_weekly.py` 会先按当日 `trade_date` 自动重建行业信号快照（回看窗口见 `config/app/strategy_governance.yaml` 的 `industry_signals`）
- 组合构建与仓位风控仅消费上述执行信号，不再直接读取单策略中间表。
