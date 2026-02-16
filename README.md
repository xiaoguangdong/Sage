# Sage

量化与宏观数据相关脚本与模型实验仓库。

## 快速开始

1. 创建/激活虚拟环境（示例）：
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `venv/`、`venv310/` 仅保留兼容用途，不作为默认开发环境
2. 设置数据源 token（不要写进代码/仓库）：
   - `export TUSHARE_TOKEN=...`
3. 宏观数据脚本入口：
   - `./scripts/data/macro/fetch_all_macro_data.sh`
   - `./scripts/data/macro/check_macro_data.sh`
4. 任务编排统一入口：
   - `python scripts/run_job.py weekly_pipeline`
   - `python scripts/run_job.py stock_monthly -- --top-n 10`
   - `python scripts/run_job.py stock_benchmark -- --models rule,lgbm,xgb`
   - `python scripts/run_job.py broker_submit -- --broker pingan`（默认 dry-run）

## 代码结构与职责

- `sage_core/`：核心算法库（趋势/宏观/选股/风控/回测等）
- `sage_app/`：非核心管线（数据接入/调度/入口脚本）
- `scripts/`：脚本入口（data/models/strategy/backtest/monitoring/legacy）
- `scripts/run_job.py`：统一任务编排入口（weekly/stock/macro/scheduler）
- `config/`：统一配置目录
  - `config/base.yaml`：全局运行时配置（数据根目录、日志、下载策略）
  - `config/tushare_tasks.yaml`：Tushare 下载任务定义
  - `config/app/`：应用层策略配置（趋势/选股/治理/风控）
  - `config/app/broker.yaml`：券商执行入口配置（仅结构，敏感信息走环境变量）

## 数据不丢失机制（推荐启用）

本仓库默认不把 `data/` 提交到 Git（见 `.gitignore`），建议使用外置盘进行镜像备份：
- 默认目标：`/Volumes/SPEED/BizData/Stock/Sage/data/`

两种触发方式（二选一或都启用）：
1. **Git 提交触发**（提交/切换分支/拉取后自动同步一次）：
   - `./scripts/backup/install_git_hooks.sh`
2. **数据变更触发（macOS）**：监控 `./data` 有变化就同步：
   - `./scripts/backup/install_launchd_watch.sh`
