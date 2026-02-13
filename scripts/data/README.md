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

## Baostock 下载器

`scripts/data/baostock_downloader.py`

示例调用在 `main()` 中已注释（按需打开）。
