# Data scripts

数据下载、清洗、对齐相关脚本放在这里。

## 运行前准备

- 设置 Tushare Token 环境变量：`export TUSHARE_TOKEN=xxxx`
- 或在项目根目录创建 `.env`：`TUSHARE_TOKEN=xxxx`
- 日志默认输出到：`logs/data/YYYYMMDD_NNN_<script>.log`
- 数据根目录优先从 `config/base.yaml -> data.roots.primary` 读取（如 `/Volumes/SPEED/BizData/Stock/sage_primary`）
- 可用环境变量覆盖：`SAGE_DATA_ROOT_PRIMARY` / `SAGE_DATA_ROOT_SECONDARY`

## 数据导入数据库（PostgreSQL 单库）

脚本：`scripts/data/import_to_postgres.py`

```bash
# 导入单表
python scripts/data/import_to_postgres.py --task daily_kline

# 导入全部（建议先 dry-run 评估体量）
python scripts/data/import_to_postgres.py --task all --dry-run
```

默认读取环境变量：
- `SAGE_DB_HOST` / `SAGE_DB_PORT` / `SAGE_DB_NAME` / `SAGE_DB_USER` / `SAGE_DB_PASSWORD`

## Tushare 统一脚本（理想版统一入口）

脚本：`scripts/data/tushare_downloader.py`

配置：`config/tushare_tasks.yaml`

```bash
python scripts/data/tushare_downloader.py --task ths_index

python scripts/data/tushare_downloader.py --task ths_daily \
  --start-date 20200101 --end-date 20251231 \
  --resume --sleep-seconds 40
```

输出分片目录：
- `data/tushare/`（默认）

默认输出目录（不传 `--output-root`）：`data/tushare/`

### 其他示例（YYYYMMDD）

```bash
# 东方财富概念指数（dc_index）
python scripts/data/tushare_downloader.py --task dc_index \
  --start-date 20200101 --end-date 20251231 --resume
```

> 说明：理想版统一入口以 `config/tushare_tasks.yaml` 为唯一任务清单，
> 后续新增接口只需追加配置，不再新增脚本。

## Akshare 统一脚本

脚本：`scripts/data/akshare_suite.py`

```bash
# 概念列表
python scripts/data/akshare_suite.py --action concept_list \
  --output-dir /tmp/sage_data

# 概念成分（支持断点续传）
python scripts/data/akshare_suite.py --action concept_components \
  --output-dir /tmp/sage_data --resume --max-items 50

# 个股历史（日线，YYYYMMDD）
python scripts/data/akshare_suite.py --action stock_hist \
  --start-date 20240101 --end-date 20240201 \
  --stock-list-csv data/tushare/filtered_stocks_list.csv \
  --output-dir /tmp/sage_data --resume
```

输出目录：
- `/tmp/sage_data/akshare/concepts/`
- `/tmp/sage_data/akshare/concepts/components/`
- `/tmp/sage_data/akshare/stock_hist/`

默认输出目录（不传 `--output-dir`）：`data/raw/akshare/`

## NBS 工业品对齐（宏观）

脚本：`scripts/data/macro/align_nbs_industrial_data.py`

```bash
# 默认读取 config/base.yaml 的数据根目录
python scripts/data/macro/align_nbs_industrial_data.py --output-dir data/processed

# 指定宏观数据目录（可用于临时数据）
python scripts/data/macro/align_nbs_industrial_data.py \
  --data-dir /tmp/sage_data/tushare/macro \
  --output-dir /tmp/sage_data/processed
```

输出（示例）：
- `nbs_industrial_aligned.parquet`（行业级汇总特征）
- `nbs_output_product_level.parquet`（产品级 + 映射结果）
- `nbs_output_mapping_summary.csv`（覆盖率摘要）
- `nbs_output_mapping_by_industry.csv`（行业统计）
- `nbs_output_mapping_unmatched.csv`（未映射产品清单）
- `nbs_fai_mapping_summary.csv` / `nbs_fai_mapping_unmatched.csv`（固定资产投资映射审计）
- `nbs_price_mapping_summary.csv` / `nbs_price_mapping_unmatched.csv`（价格指数映射审计）

说明：
- 固定资产投资/价格指数会尝试使用 `config/sw_nbs_mapping.yaml` 将 NBS 行业名映射到申万一级。

## 政策信号管道（MVP）

### 1) 政策数据拉取（Tushare 公告/研报）

脚本：`scripts/data/tushare_downloader.py`

说明：
- 需要 `.env` 中配置 `TUSHARE_TOKEN`
- 自动断点续传（`--resume`）
- 会强制关闭代理环境变量（避免代理导致的访问问题）

示例：
```bash
python scripts/data/tushare_downloader.py --task tushare_anns \
  --start-date 20200101 --end-date 20251231 --resume

python scripts/data/tushare_downloader.py --task tushare_reports \
  --start-date 20200101 --end-date 20251231 --resume
```

默认输出：
- `data/tushare/policy/tushare_anns.parquet`
- `data/tushare/policy/tushare_reports.parquet`

### 2) 政策数据拉取（政府网站 RSS/Atom）

脚本：`scripts/data/policy/fetch_gov_policy.py`

说明：
- 先在 `config/policy_sources.yaml` 填写官方 RSS/Atom 地址
- 支持 `type: html`（用于没有 RSS 的官网列表页），可填写 `base_url` 处理相对链接
- 输出汇总到 `gov_notices.parquet`

示例：
```bash
python scripts/data/policy/fetch_gov_policy.py
```

排查解析失败可加：
```bash
python scripts/data/policy/fetch_gov_policy.py --dump-html
```

默认输出：
- `data/raw/policy/gov_notices.parquet`
- `data/raw/policy/gov_notices_summary.json`

### 2.5) 研报/个股摘要（仅供个股模型，非政策信号输入）

脚本：`scripts/data/policy/fetch_10jqka_reports.py`

示例：
```bash
python3 scripts/data/policy/fetch_10jqka_reports.py --symbol 002988 --dump-html
python3 scripts/data/policy/fetch_10jqka_reports.py --symbol 002988 --section forecast --dump-html
```

默认输出（不进入 `policy_signal_pipeline`）：
- `data/raw/policy/10jqka_reports.parquet`
- `data/raw/policy/10jqka_forecast.parquet`

### 2.6) 行业研报（东方财富）

脚本：`scripts/data/policy/fetch_eastmoney_industry_reports.py`

说明：
- 行业级研报（替代 Tushare 行业研报权限）
- 可按日期、评级、行业代码过滤

示例：
```bash
python3 scripts/data/policy/fetch_eastmoney_industry_reports.py \
  --begin-date 2024-01-01 --end-date 2024-12-31 \
  --page-size 50 --sleep-seconds 1.0 --resume
```

默认输出（进入 `policy_signal_pipeline`）：
- `data/raw/policy/eastmoney_industry_reports.parquet`

### 2.6.1) 同花顺板块指数（ths_index / ths_daily）

脚本：`scripts/data/tushare_downloader.py`

说明：
- `ths_index`：板块指数列表（单次全量）
- `ths_daily`：板块指数行情（按月分页）

示例：
```bash
python3 scripts/data/tushare_downloader.py --task ths_index
python3 scripts/data/tushare_downloader.py --task ths_daily --start-date 20230101 --end-date 20251231 --resume
python3 scripts/data/run_ths_daily_monthly_full.py --start-date 20200101 --end-date 20260213
# 或统一入口
python3 scripts/run_job.py ths_daily_monthly_full -- --start-date 20200101 --end-date 20260213
```

输出：
- `data/tushare/concepts/ths_index.parquet`
- `data/tushare/concepts/ths_daily.parquet`
- `data/processed/concepts/ths_daily_completeness_report.json`

### 3) 政策信号管道

脚本：`scripts/data/policy/policy_signal_pipeline.py`

输入文件（放在 `data/raw/policy/` 或 `data/tushare/policy/`）：
- `tushare_anns.parquet` / `tushare_anns.csv`（公告）
- `gov_notices.parquet` / `gov_notices.csv`（政府网站）
- `tushare_reports.parquet` / `tushare_reports.csv`（研报）
- `eastmoney_industry_reports.parquet` / `eastmoney_industry_reports.csv`（行业研报）

输出：
- `data/processed/policy/policy_signals.parquet`
- `data/processed/policy/policy_signals_summary.json`
- `data/processed/policy/policy_source_health.parquet`

说明：
- `policy_signals.parquet` 已包含 `confidence` 字段。
- 管道会先做去重（同源+同日+标题/内容指纹），再聚合行业信号。
- 来源权重会叠加 `source_stability_score`（基于14/30日活跃度）。

示例：
```bash
python scripts/data/policy/policy_signal_pipeline.py
```

### 4) 政策信号增强（行业研报 + 申万行业动量）

脚本：`scripts/data/policy/policy_signal_enhanced.py`

说明：
- 行业研报：使用东方财富行业研报（近7/30日研报数量、评级变动）
- 行业动量：使用申万行业指数日线（20/60日动量）

示例：
```bash
python3 scripts/data/policy/policy_signal_enhanced.py
```

## 概念→行业映射（数据与映射层）

脚本：`scripts/data/concepts/build_concept_industry_mapping.py`

说明：
- 基于概念成分股 + 申万L1成分股，生成概念→行业覆盖率与主行业

示例：
```bash
python3 scripts/data/concepts/build_concept_industry_mapping.py --min-ratio 0.05 --strict-ratio 0.2
```

输出：
- `data/processed/concepts/concept_industry_coverage.parquet`
- `data/processed/concepts/concept_industry_primary.parquet`
- `data/processed/concepts/concept_industry_primary_high_conf.parquet`
- `data/processed/concepts/concept_industry_unmapped.parquet`

输出：
- `data/processed/policy/policy_signals_enhanced.parquet`
- `data/processed/policy/policy_signals_enhanced_summary.json`

## Legacy 脚本

历史脚本已移动至 `scripts/legacy/data/`，仅保留参考：
- `baostock_downloader.py`
- `fetch_akshare_concepts.py`
- `fetch_efinance_concepts.py`

如需恢复为主流程，请先按新的统一脚本结构改造。

示例调用在 `main()` 中已注释（按需打开）。
