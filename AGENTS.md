# Sage / 量化交易智能平台 - Agent 约定

本仓库为 A 股量化交易平台，采用"宏观定方向 → 趋势择时 → 选股排序 → 执行过滤"四层架构。Agent 在这里工作的目标是：**少猜、可复现、可回滚**，并尽量把约定固化为脚本与文件规范。

## 1) 项目结构与职责

```
Sage/
├── sage_core/           # 核心算法库（趋势/宏观/选股/风控/回测）
│   ├── trend/           # 趋势状态识别
│   ├── stock_selection/ # 选股与排序（rank_model / stock_scoring）
│   ├── execution/       # 买卖点与执行过滤（broker_adapter / signal_contract）
│   ├── industry/        # 行业与宏观模型
│   ├── governance/      # Champion/Challenger 策略治理
│   ├── portfolio/       # 组合构建与风控
│   └── backtest/        # 回测引擎
├── sage_app/            # 非核心管线（数据接入/调度/入口脚本）
│   ├── data/            # 数据加载与提供者
│   └── pipelines/       # 周度/日度调度器
├── scripts/             # 脚本入口（统一任务编排）
│   ├── run_job.py       # 统一任务入口（核心）
│   ├── data/            # 数据下载/清洗/对齐
│   ├── models/          # 模型训练/预测
│   ├── stock/           # 选股与基准对比
│   ├── strategy/        # 信号融合/策略编排
│   └── monitoring/      # 监控与报告
├── config/              # 统一配置目录
│   ├── base.yaml        # 全局运行时配置
│   ├── tushare_tasks.yaml # Tushare 下载任务定义
│   └── app/             # 应用层策略配置
│       ├── trend_model.yaml   # 趋势模型配置
│       ├── rank_model.yaml    # 选股排序模型配置
│       ├── broker.yaml        # 券商执行配置
│       └── strategy_governance.yaml
└── data/                # 数据目录（不进 Git）
```

## 2) 统一任务入口（推荐使用）

**核心入口：`scripts/run_job.py`**

```bash
# 周度流水线（趋势+选股+信号融合）
python scripts/run_job.py weekly_pipeline

# 选股任务
python scripts/run_job.py stock_monthly -- --top-n 15
python scripts/run_job.py stock_benchmark -- --models rule,lgbm,xgb

# 趋势信号调度
python scripts/run_job.py trend_scheduler

# 券商执行（默认 dry-run）
python scripts/run_job.py broker_submit -- --broker pingan
```

**可用任务列表：**
| 任务名 | 说明 |
|--------|------|
| `weekly_pipeline` | 周度流水线（趋势+选股+信号融合） |
| `stock_monthly` | 月度选股 |
| `stock_benchmark` | Rule/LGBM/XGB 基准对比 |
| `stock_governance` | 策略治理评估 |
| `trend_scheduler` | 趋势信号调度 |
| `stock_scheduler` | 选股调度器 |
| `broker_submit` | 券商执行提交 |
| `ths_daily_monthly_full` | 同花顺板块指数全量更新 |

## 3) 数据目录与路径约定

**数据根目录（优先从 `config/base.yaml` 读取）：**
- 主目录：`data/`（项目内）
- 备份目录：`/Volumes/SPEED/BizData/Stock/Sage/data/`（外置盘）

**数据布局：**
```
data/
├── tushare/           # Tushare 原始数据
│   ├── macro/         # 宏观数据
│   ├── northbound/    # 北向资金
│   ├── sectors/       # 行业指数
│   └── concepts/      # 概念指数（ths_daily/ths_index）
├── raw/               # 其他原始数据（akshare/policy）
├── processed/         # 处理后数据（特征/标签/信号）
├── features/          # 特征库
├── labels/            # 标签库
├── signals/           # 信号库
├── states/            # 模型状态
└── backtest/          # 回测结果
```

## 4) 配置分层规范

```
config/
├── base.yaml                 # 全局运行时（数据根目录、日志、下载策略）
├── tushare_tasks.yaml        # Tushare 任务定义
├── sw_nbs_mapping.yaml       # 申万L1 → NBS 行业映射
├── sw_nbs_mapping_l2.yaml    # 申万L2 行业映射
└── app/
    ├── trend_model.yaml      # 趋势模型（rule/lgbm/hmm）
    ├── rank_model.yaml       # 选股排序（lgbm/xgb）
    ├── broker.yaml           # 券商执行（敏感信息走环境变量）
    ├── risk_control.yaml     # 风控配置
    └── strategy_governance.yaml # 策略治理
```

## 5) 运行方式与环境

**Python 版本：**
- `venv/`：Python 3.13.x（主开发环境）
- `venv310/`：Python 3.10.x（兼容性用途）

**运行脚本：**
```bash
# 推荐：使用统一入口
python scripts/run_job.py <task_name>

# 直接调用脚本
./venv/bin/python scripts/data/xxx.py
```

**测试：**
```bash
pytest  # 自动扫描 sage_core/tests 和 sage_app/tests
```

## 6) Tushare / 账号密钥（强约束）

- **禁止**在代码、文档、日志里硬编码 token/密码
- Tushare token 统一从环境变量读取：
  - `TUSHARE_TOKEN`（推荐）或 `TS_TOKEN`
- 券商 API 密钥走环境变量：
  - `PINGAN_API_KEY` / `PINGAN_API_SECRET` / `PINGAN_API_TOKEN`
- 本地私密配置文件（如 `config/joinquant.json`）不提交版本库

## 7) 速率限制与长任务习惯

**Tushare 约束：**
- IP 数量限制（最多 2 IP）
- 请求间隔：脚本中已按 40–60s sleep

**长任务规范：**
- 输出重定向到日志：`logs/data/` 或 `logs/jobs/`
- 后台执行：`nohup python scripts/... &`
- 追踪日志：`tail -f logs/xxx.log`
- 并发抓取要保守，不要擅自提高并发或缩短 sleep

## 8) 数据脚本使用指南

**Tushare 统一下载器：**
```bash
# 同花顺板块指数
python scripts/data/tushare_downloader.py --task ths_index
python scripts/data/tushare_downloader.py --task ths_daily --start-date 20200101 --end-date 20251231 --resume
```

**政策信号管道：**
```bash
# 政策数据拉取
python scripts/data/tushare_downloader.py --task tushare_anns --start-date 20200101 --end-date 20251231 --resume
python scripts/data/policy/fetch_gov_policy.py

# 信号生成
python scripts/data/policy/policy_signal_pipeline.py
python scripts/data/policy/policy_signal_enhanced.py
```

**概念→行业映射：**
```bash
python scripts/data/concepts/build_concept_industry_mapping.py --min-ratio 0.05
```

## 9) 变更原则（写代码时默认遵守）

- 先读现有脚本/文档再改
- 只做与任务相关的最小改动
- 破坏性操作（`rm -rf`、`pkill -9`、清空数据目录）**先问一句**
- 输出文件可重跑：断点续传、增量更新
- 搜索/批处理默认排除 `venv/`、`venv310/`、`data/`、`logs/`

## 10) 数据备份机制

- 代码进 Git；`data/` 不进 Git（见 `.gitignore`）
- `data/` 镜像备份到外置盘：`/Volumes/SPEED/BizData/Stock/Sage/data/`
- 触发方式：
  - Git hooks：`./scripts/backup/install_git_hooks.sh`
  - macOS launchd：`./scripts/backup/install_launchd_watch.sh`

## 11) 重要文档索引

**架构设计：**
- `docs/2.1 Sage股票智能交易平台总架构设计文档.md`
- `docs/项目总体架构设计设想.md`

**模块设计：**
- `docs/2.2 Sage股票智能交易平台趋势状态模块设计文档.md`
- `docs/2.13 Sage股票智能交易平台选股排序模型模块设计文档.md`
- `docs/2.14 Sage股票智能交易平台买卖点过滤模块设计文档.md`
- `docs/2.11 Sage股票智能交易平台宏观模型模块设计文档.md`

**运行记录：**
- `docs/2.8.Sage股票智能交易平台趋势状态模块运行情况总结.md`
- `docs/2.9.Sage股票智能交易平台趋势状态模块回测情况总结.md`
- `docs/2.6 Sage股票智能交易平台问题记录.md`

**数据相关：**
- `docs/申万-NBS行业映射系统说明.md`
- `docs/概念板块数据源对比总结.md`
- `scripts/data/README.md`

## 12) 常见任务速查

| 场景 | 命令 |
|------|------|
| 周度流水线 | `python scripts/run_job.py weekly_pipeline` |
| 选股基准对比 | `python scripts/run_job.py stock_benchmark -- --models rule,lgbm,xgb` |
| 趋势信号更新 | `python scripts/run_job.py trend_scheduler` |
| 同花顺板块更新 | `python scripts/run_job.py ths_daily_monthly_full -- --start-date 20200101` |
| 政策信号生成 | `python scripts/data/policy/policy_signal_pipeline.py` |
| 行业映射构建 | `python scripts/data/concepts/build_concept_industry_mapping.py` |
| 运行测试 | `pytest` |