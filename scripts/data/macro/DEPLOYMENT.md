# 宏观数据获取系统部署说明

## 一、已创建的文件

### 1. 数据获取脚本
- `scripts/data/macro/fetch_nbs_data.py` - 国家统计局数据获取
- `scripts/data/tushare_downloader.py` - Tushare统一下载入口
- `scripts/data/macro/fetch_all_macro_data.sh` - 一键获取所有宏观数据
- `scripts/data/macro/check_macro_data.sh` - 数据完整性检查

### 2. 定时任务配置
- `scripts/data/macro/cron_macro_schedule.conf` - Cron定时任务配置

### 3. 文档
- `scripts/data/macro/README.md` - 宏观数据规范文档

## 二、数据目录结构

```
data/tushare/
├── macro/                           # 宏观数据
│   ├── nbs_ppi_industry_202512.csv  ✅ 已测试（模拟数据）
│   ├── nbs_fai_industry_202512.csv   ✅ 已测试（模拟数据）
│   └── nbs_output_202512.csv        ✅ 已测试（模拟数据）
└── northbound/                      # 北向资金数据（待IP限制解除后获取）
```

## 三、使用方法

### 3.1 手动获取数据

**获取NBS数据（上个月）**
```bash
python3 scripts/data/macro/fetch_nbs_data.py
```

**获取Tushare宏观数据（CPI/PPI/PMI/国债）**
```bash
python3 scripts/data/tushare_downloader.py --task cn_cpi --start-date 202001 --end-date $(date +%Y%m) --resume
python3 scripts/data/tushare_downloader.py --task cn_ppi --start-date 202001 --end-date $(date +%Y%m) --resume
python3 scripts/data/tushare_downloader.py --task cn_pmi --start-date 202001 --end-date $(date +%Y%m) --resume
python3 scripts/data/tushare_downloader.py --task yield_10y --resume
python3 scripts/data/tushare_downloader.py --task yield_2y --resume
```

**获取北向资金数据（最近1个月）**
```bash
python3 scripts/data/tushare_downloader.py --task northbound_flow --start-date 20200101 --end-date $(date +%Y%m%d) --resume
python3 scripts/data/tushare_downloader.py --task northbound_hold --start-date 20200101 --end-date $(date +%Y%m%d) --resume
```

**一键获取所有数据**
```bash
./scripts/data/macro/fetch_all_macro_data.sh
```

### 3.2 设置定时任务

**查看配置**
```bash
cat scripts/data/macro/cron_macro_schedule.conf
```

**添加到crontab**
```bash
crontab scripts/data/macro/cron_macro_schedule.conf
```

**编辑crontab**
```bash
crontab -e
```

### 3.3 检查数据完整性
```bash
./scripts/data/macro/check_macro_data.sh
```

## 四、定时任务说明

### 4.1 每月1日凌晨2点
- 更新国家统计局数据（PMI等）
- 命令：`python3 scripts/data/macro/fetch_nbs_data.py`

### 4.2 每月10日凌晨2点
- 更新Tushare月度数据（CPI、PPI等）
- 命令：`python3 scripts/data/tushare_downloader.py --task cn_cpi ...`

### 4.3 每月16日凌晨2点
- 更新国家统计局行业数据（固定资产投资、分行业PPI等）
- 命令：`python3 scripts/data/macro/fetch_nbs_data.py`

### 4.4 每周五凌晨2点
- 更新北向资金持仓数据
- 命令：`python3 scripts/data/tushare_downloader.py --task northbound_hold ...`

### 4.5 每日凌晨2点15分
- 更新日度数据（10Y收益率、北向资金流向）
- 命令：`python3 scripts/data/tushare_downloader.py --task yield_10y ...`

## 五、当前状态

### ✅ 已完成
- NBS数据获取脚本（模拟数据，待替换为真实爬虫）
- Tushare数据获取脚本（待IP限制解除）
- 北向资金数据获取脚本（待IP限制解除）
- 定时任务配置文件
- 数据完整性检查脚本
- 数据规范文档

### ⚠️ 待完成
- 等待Tushare IP限制解除（30分钟-1小时）
- 替换NBS模拟数据为真实爬虫
- 测试所有数据获取功能
- 验证定时任务配置

## 六、注意事项

### 6.1 Tushare IP限制
- 当前IP限制：117.35.134.71
- 最大IP数量：2个
- 解除时间：30分钟-1小时
- 建议：每天10:30后运行批量下载

### 6.2 NBS数据爬虫
- 当前使用模拟数据
- 需要根据实际网页结构编写爬虫
- 建议每月手动下载Excel并导入

### 6.3 数据更新
- NBS数据：每月9-15日分批发布
- Tushare数据：每月1-10日更新
- 定时任务已配置，自动更新

## 七、下一步

1. 等待IP限制解除
2. 测试Tushare数据获取
3. 实现NBS真实爬虫
4. 验证所有数据格式
5. 开始实现宏观模型

---

**创建时间**: 2026-02-11
**脚本位置**: `scripts/data/macro/`
**数据目录**: `data/tushare/macro/`
