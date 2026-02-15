# 宏观数据规范文档

## 一、数据目录结构

```
data/tushare/
├── macro/                           # 宏观数据
│   ├── nbs_ppi_industry_*.csv      # NBS分行业PPI（月度）
│   ├── nbs_fai_industry_*.csv       # NBS分行业固定资产投资（月度）
│   ├── nbs_output_*.csv             # NBS主要工业品产量（月度）
│   ├── tushare_cpi.parquet         # Tushare CPI（月度）
│   ├── tushare_ppi.parquet         # Tushare PPI（月度）
│   ├── tushare_pmi.parquet         # Tushare PMI（月度）
│   ├── yield_10y.parquet           # Tushare 10年期收益率（日度）
│   ├── yield_2y.parquet            # Tushare 2年期收益率（日度）
│   └── tushare_shibor.parquet      # Tushare SHIBOR（日度）
├── northbound/                      # 北向资金数据
│   ├── northbound_flow.parquet          # 北向资金日度流向（日度）
│   └── northbound_hold.parquet          # 北向资金持仓（日度）
└── sectors/                        # 行业数据
    ├── sw_daily_all.parquet          # 申万行业日K线
    └── all_concept_details.csv       # 概念成分股数据
```

## 二、数据文件规范

### 2.1 NBS数据文件

#### 文件命名
- `nbs_ppi_industry_YYYYMM.csv`
- `nbs_fai_industry_YYYYMM.csv`
- `nbs_output_YYYYMM.csv`

#### 数据格式

**分行业PPI (nbs_ppi_industry_YYYYMM.csv)**
```csv
industry,industry_code,ppi_yoy,year,month,date
煤炭开采和洗选业,C01,-2.5,2026,1,2026-01-01
石油和天然气开采业,C02,-5.8,2026,1,2026-01-01
...
```

**分行业固定资产投资 (nbs_fai_industry_YYYYMM.csv)**
```csv
industry,industry_code,fai_yoy,year,month,date
制造业,C13,8.2,2026,1,2026-01-01
医药制造业,C14,15.8,2026,1,2026-01-01
...
```

**主要工业品产量 (nbs_output_YYYYMM.csv)**
```csv
product,output_yoy,year,month,date
集成电路（亿块）,8.5,2026,1,2026-01-01
新能源汽车（万辆）,35.2,2026,1,2026-01-01
...
```

### 2.2 Tushare宏观数据文件

#### 文件命名
- `tushare_cpi.parquet`
- `tushare_ppi.parquet`
- `tushare_pmi.parquet`
- `yield_10y.parquet`
- `yield_2y.parquet`
- `tushare_shibor.parquet`

#### 数据格式

**CPI数据 (tushare_cpi.parquet)**
```
month,nt_yoy,nt_mom,ct_yoy,ct_mom,adj_ct_yoy
202601,0.5,0.2,0.8,0.3,0.6
202602,0.6,0.1,0.9,0.4,0.7
...
```

**PPI数据 (tushare_ppi.parquet)**
```
month,ppi_yoy,ppi_mom,ppi_cyc_yoy,ppi_cyc_mom
202601,-2.5,-0.3,-2.0,-0.2
202602,-2.3,-0.2,-1.8,-0.1
...
```

**PMI数据 (tushare_pmi.parquet)**
```
month,markit_pmi,cfl_pmi,manufacture_pmi,non_manufacture_pmi
202601,50.5,51.2,49.8,51.8
202602,51.0,51.5,50.2,52.0
...
```

**10年期收益率 (yield_10y.parquet)**
```
date,yield_10y
2026-01-01,2.5
2026-01-02,2.52
...
```

### 2.3 北向资金数据文件

#### 文件命名
- `northbound_flow.parquet`
- `northbound_hold.parquet`
- `industry_northbound_flow.parquet`（行业配置聚合，含 `is_proxy` 代理延展标记）

#### 数据格式

**日度流向 (northbound_flow.parquet)**
```
trade_date,net_amount_in,net_amount_out,net_amount,net_pct_chg
2026-01-01,500000000,200000000,300000000,0.05
2026-01-02,600000000,150000000,450000000,0.08
...
```

**持仓数据 (northbound_hold.parquet)**
```
trade_date,ts_code,name,hold_amount,hold_ratio,exchange_code
2026-01-01,600519.SH,贵州茅台,1000000000,15.5,SH
2026-01-01,000858.SZ,五粮液,800000000,12.3,SZ
...
```

**行业配置聚合 (industry_northbound_flow.parquet)**
```
industry_code,industry_name,trade_date,vol,ratio,industry_ratio,is_proxy
801010,农林牧渔,2026-02-10,,0.021,0.021,true
...
```
说明：
- 原始持仓可用日期之后，按最新行业权重结构做“代理延展”（`is_proxy=true`），用于维持行业信号时效；
- 代理数据默认降置信度使用，不替代真实持仓数据。

## 三、统一下载入口（理想版）

统一入口：`scripts/data/tushare_downloader.py`  
任务清单：`config/tushare_tasks.yaml`

示例：
```bash
python3 scripts/data/tushare_downloader.py --task cn_cpi --start-date 202001 --end-date $(date +%Y%m) --resume
python3 scripts/data/tushare_downloader.py --task cn_ppi --start-date 202001 --end-date $(date +%Y%m) --resume
python3 scripts/data/tushare_downloader.py --task cn_pmi --start-date 202001 --end-date $(date +%Y%m) --resume
python3 scripts/data/tushare_downloader.py --task yield_10y --resume
python3 scripts/data/tushare_downloader.py --task yield_2y --resume
python3 scripts/data/tushare_downloader.py --task northbound_flow --start-date 20200101 --end-date $(date +%Y%m%d) --resume
python3 scripts/data/tushare_downloader.py --task northbound_hold --start-date 20200101 --end-date $(date +%Y%m%d) --resume
python3 scripts/data/macro/aggregate_northbound_industry_flow.py --tushare-root data/tushare
```

## 四、数据更新时间表

| 数据类型 | 更新频率 | 更新时间 | 数据来源 |
|---------|---------|---------|---------|
| NBS-PPI | 月度 | 每月9-10日 | 国家统计局 |
| NBS-FAI | 月度 | 每月15日左右 | 国家统计局 |
| NBS-产量 | 月度 | 每月15日左右 | 国家统计局 |
| Tushare-CPI | 月度 | 每月9-10日 | Tushare API |
| Tushare-PPI | 月度 | 每月9-10日 | Tushare API |
| Tushare-PMI | 月度 | 每月1日 | Tushare API |
| 10Y收益率 | 日度 | 每日收盘后 | Tushare API |
| 北向资金流向 | 日度 | 每日收盘后 | Tushare API |
| 北向资金持仓 | 周度 | 每周五收盘后 | Tushare API |

## 五、数据质量检查

### 4.1 检查脚本
```bash
./scripts/data/macro/check_macro_data.sh
```

### 4.2 检查项
- [ ] 数据文件是否存在
- [ ] 数据文件大小正常
- [ ] 数据时间戳是最新的
- [ ] 数据没有缺失值
- [ ] 数据格式正确

## 六、数据备份

### 6.1 备份策略
- 每月自动备份到 `data/backup/macro/` 目录
- 保留最近12个月的数据
- 超过12个月的数据压缩存储

### 6.2 恢复策略
- 从备份目录恢复历史数据
- 使用增量更新补全数据

## 七、常见问题

### Q1: 数据更新失败
**A**: 检查网络连接，检查Tushare token是否有效，检查cron服务是否运行

### Q2: 数据缺失
**A**: 查看日志文件 `logs/macro/fetch_*.log`，检查是否有错误信息

### Q3: 数据格式错误
**A**: 检查数据源是否更新了格式，相应调整解析代码
