# Data scripts

数据下载、清洗、对齐相关脚本放在这里。

## 运行前准备

- 设置 Tushare Token 环境变量：`export TUSHARE_TOKEN=xxxx`
- 或在项目根目录创建 `.env`：`TUSHARE_TOKEN=xxxx`
- 日志默认输出到：`logs/data/YYYYMMDD_NNN_<script>.log`

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

### 其他示例（YYYYMMDD）

- 日线K线：`--action daily_kline --start-date 20240101 --end-date 20240105`
- 指数OHLC：`--action index_ohlc --start-date 20240101 --end-date 20240105`
- 沪深300成分：`--action hs300_constituents --start-year 2020 --end-year 2021`
- 沪深300资金流：`--action hs300_moneyflow --start-date 20240101 --end-date 20240105`
- 申万行业分类：`--action sw_industry_classify`
- 申万行业日线：`--action sw_industry_daily --start-date 20240101 --end-date 20240105`
- 期权日线：`--action opt_daily --start-date 20240101 --end-date 20240105`
- 财务指标：`--action fina_indicator --start-date 20240101 --end-date 20241231`
- 财务指标VIP：`--action fina_indicator_vip --start-year 2020 --end-year 2021`
- 行业/概念列表：`--action tushare_sectors`
- 概念更新（Tushare）：`--action concept_update_tushare --mode init|update|calculate`
- 概念更新（东财）：`--action concept_update_eastmoney --mode init|update|calculate`

## Baostock 下载器

`scripts/data/baostock_downloader.py`

示例调用在 `main()` 中已注释（按需打开）。
