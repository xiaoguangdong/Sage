# Data scripts

数据下载、清洗、对齐相关脚本放在这里。

## 运行前准备

- 设置 Tushare Token 环境变量：`export TUSHARE_TOKEN=xxxx`
- 或在项目根目录创建 `.env`：`TUSHARE_TOKEN=xxxx`
- 日志默认输出到：`logs/data/YYYYMMDD_NNN_<script>.log`
- 数据根目录优先从 `config/base.yaml -> data.roots.primary` 读取（如 `/Volumes/SPEED/BizData/Stock/sage_primary`）
- 可用环境变量覆盖：`SAGE_DATA_ROOT_PRIMARY` / `SAGE_DATA_ROOT_SECONDARY`

## Tushare 统一脚本

脚本：`scripts/data/tushare_suite.py`

```bash
python scripts/data/tushare_suite.py --action daily_basic \
  --start-date 20200101 --end-date 20241231 \
  --output-dir /tmp/sage_data --resume

python scripts/data/tushare_suite.py --action margin \
  --start-date 20200101 --end-date 20241231 \
  --output-dir /tmp/sage_data --resume
```

输出分片目录：
- `/tmp/sage_data/tushare/daily_basic/parts/`
- `/tmp/sage_data/tushare/margin/parts/`

默认输出目录（不传 `--output-dir`）：`data/raw/tushare/`

### 其他示例（YYYYMMDD）

```bash
# 日线K线
python scripts/data/tushare_suite.py --action daily_kline \
  --start-date 20240101 --end-date 20240105 \
  --output-dir /tmp/sage_data

# 指数OHLC
python scripts/data/tushare_suite.py --action index_ohlc \
  --start-date 20240101 --end-date 20240105 \
  --output-dir /tmp/sage_data

# 沪深300成分
python scripts/data/tushare_suite.py --action hs300_constituents \
  --start-year 2020 --end-year 2021 \
  --output-dir /tmp/sage_data

# 沪深300资金流
python scripts/data/tushare_suite.py --action hs300_moneyflow \
  --start-date 20240101 --end-date 20240105 \
  --output-dir /tmp/sage_data

# 申万行业分类
python scripts/data/tushare_suite.py --action sw_industry_classify \
  --output-dir /tmp/sage_data

# 申万行业日线
python scripts/data/tushare_suite.py --action sw_industry_daily \
  --start-date 20240101 --end-date 20240105 \
  --output-dir /tmp/sage_data

# 期权日线
python scripts/data/tushare_suite.py --action opt_daily \
  --start-date 20240101 --end-date 20240105 \
  --output-dir /tmp/sage_data

# 财务指标
python scripts/data/tushare_suite.py --action fina_indicator \
  --start-date 20240101 --end-date 20241231 \
  --stock-list-csv data/raw/tushare/filtered_stocks_list.csv \
  --output-dir /tmp/sage_data

# 财务指标VIP
python scripts/data/tushare_suite.py --action fina_indicator_vip \
  --start-year 2020 --end-year 2021 \
  --output-dir /tmp/sage_data

# 行业/概念列表
python scripts/data/tushare_suite.py --action tushare_sectors \
  --output-dir /tmp/sage_data

# 概念更新（Tushare）
python scripts/data/tushare_suite.py --action concept_update_tushare \
  --mode update --start-date 20240924 --end-date 20241231 \
  --min-stock-count 10 --output-dir /tmp/sage_data

# 概念更新（东财）
python scripts/data/tushare_suite.py --action concept_update_eastmoney \
  --mode update --start-date 20240924 --end-date 20241231 \
  --output-dir /tmp/sage_data
```

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
  --stock-list-csv data/raw/tushare/filtered_stocks_list.csv \
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

说明：
- 固定资产投资/价格指数会尝试使用 `config/sw_nbs_mapping.yaml` 将 NBS 行业名映射到申万一级。

## Legacy 脚本

历史脚本已移动至 `scripts/legacy/data/`，仅保留参考：
- `baostock_downloader.py`
- `fetch_akshare_concepts.py`
- `fetch_efinance_concepts.py`

如需恢复为主流程，请先按新的统一脚本结构改造。

示例调用在 `main()` 中已注释（按需打开）。
