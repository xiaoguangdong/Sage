# 概念成分股数据更新系统（理想版统一入口）

## 概述

这个系统用于定期更新概念板块的成分股数据。当前统一由 `tushare_downloader.py` 入口完成，后续评分与表现计算再独立接入。

## 功能特点

1. **定期更新**: 每周更新概念列表和成分股数据
2. **超时处理**: 设置30秒超时，避免长时间等待
3. **失败重试**: 自动重试失败的请求（最多3次）
4. **数据版本管理**: 保存带时间戳的历史数据
5. **表现计算**: 自动计算概念的多维度表现指标
6. **评分排名**: 综合涨幅、回撤、持续性等指标进行评分

## 使用方法

### 方法1：使用Python脚本

```bash
# 激活虚拟环境
source venv/bin/activate

# 初始化/更新：获取概念列表与成分
python scripts/data/tushare_downloader.py --task tushare_concept_list
python scripts/data/tushare_downloader.py --task tushare_concept_detail
```

### 方法2：使用Shell脚本

```bash
# 初始化
./scripts/data/update_concept_data.sh init

# 周度更新
./scripts/data/update_concept_data.sh update

# 只计算表现
./scripts/data/update_concept_data.sh calc
```

## 命令参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务名（`tushare_concept_list`/`tushare_concept_detail`） | - |

## 工作流程

### 概念更新流程

1. 拉取概念列表（`tushare_concept_list`）
2. 按概念逐个拉取成分股（`tushare_concept_detail`）

> 概念表现/评分计算将作为独立模块接入，不在下载器内处理。

## 评分指标

系统综合以下6个维度进行评分：

| 指标 | 权重 | 说明 |
|------|------|------|
| 涨幅分 | 25% | 累计涨幅（越高越好） |
| 回撤分 | 15% | 最大回撤（越小越好） |
| 波动率分 | 10% | 波动率（越小越好） |
| 持续分 | 20% | 上涨周数比例 |
| 上榜分 | 20% | 周度Top5上榜次数 |
| 成分股分 | 10% | 成分股数量（越多越好） |

## 数据文件

### 输入文件

- `data/raw/tushare/daily/daily_YYYY.parquet` - 个股日线数据（按年存储）
- `data/raw/tushare/sectors/all_concept_details_base.csv` - 基准概念成分股数据

### 输出文件

- `data/raw/tushare/sectors/concepts.parquet` - 概念列表
- `data/raw/tushare/sectors/concept_detail.parquet` - 概念成分
- `logs/data/YYYYMMDD_NNN_tushare_downloader.log` - 更新日志

## 日志记录

所有操作都会记录到 `logs/data/YYYYMMDD_NNN_tushare_downloader.log` 文件中，包括：

- 获取的概念数量
- 成功/失败/超时的概念列表
- 计算过程中的关键信息

## 注意事项

1. **IP限制**: Tushare有IP数量限制（最大2个），如果遇到IP限制，请等待30分钟-1小时后重试
2. **频率限制**: 脚本已设置0.3秒间隔，避免触发频率限制
3. **超时设置**: 每个概念获取超时时间为30秒
4. **数据更新**: 建议每周五收盘后运行更新脚本

## 定时任务设置

可以使用cron定时任务实现每周自动更新：

```bash
# 编辑crontab
crontab -e

# 添加以下内容（每周五下午3点执行）
0 15 * * 5 cd /Users/dongxg/SourceCode/Sage && ./scripts/data/update_concept_data.sh update >> logs/concept_update_cron.log 2>&1
```

## 故障排查

### 问题1：IP数量超限

**现象**: `Exception: 您的IP数量超限，最大数量为2个！`

**解决**:
- 等待30分钟-1小时后重试
- 检查是否有其他程序同时访问Tushare

### 问题2：概念获取超时

**现象**: 日志中出现大量"概念 XXX 获取超时"

**解决**:
- 检查网络连接
- 增加超时时间（修改脚本中的`TIMEOUT`参数）
- 减少并发请求

### 问题3：个股数据文件不存在

**现象**: `个股数据文件不存在: data/raw/tushare/daily/daily_2024.parquet`

**解决**:
- 确认个股数据已下载
- 检查文件路径是否正确
- 如果需要其他年份的数据，先下载对应年份的个股数据

## 示例输出

```
================================================================
综合评分 Top 10:
====================================================================
排名   概念名称                 综合评分       涨幅分      回撤分      持续分      上榜分      成分股分    
----------------------------------------------------------------------------------------------------
1    通用航空                     58.4   100.0    50.6    60.0    26.7    62.0
2    集成电路概念                   57.2    95.3    15.2    66.7    26.7   100.0
3    郭台铭概念                    55.3    73.9    65.4    60.0    26.7    34.0
...
```

## 后续改进

1. **时间切片**: 实现按时间切片获取历史成分股数据
2. **增量更新**: 只更新有变化的概念成分股
3. **通知机制**: 更新完成后发送邮件/消息通知
4. **可视化**: 生成概念表现的可视化图表
