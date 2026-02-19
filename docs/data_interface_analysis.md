# 数据接口需求分析文档

## 概述

本文档详细分析了`market_tendcy_qa.md`中提到的所有指标，并按数据类型和来源进行分类，列出了需要调用的供应商接口。

---

## 一、指标分类汇总

### 1.1 技术面指标（可自行计算）

所有技术指标都可以基于OHLCV数据自行计算，无需外部API调用。

| 指标类别 | 具体指标 | 计算方法 | 数据需求 |
|---------|---------|---------|---------|
| **移动平均线** | MA5, MA10, MA20, MA60, MA120, MA250 | 简单移动平均 | close |
| **MACD指标** | DIF, DEA, MACD, MACD_Hist | EMA计算 | close |
| **RSI指标** | RSI14, RSI30 | 相对强弱指数 | close |
| **布林带** | 上轨、下轨、中轨、宽度 | 均线±2倍标准差 | close |
| **ATR** | ATR14, ATR20 | 平均真实波幅 | high, low, close |
| **波动率** | 20日波动率 | 收益率标准差 | close |
| **动量指标** | 动量5日、20日、60日 | 价格变化率 | close |
| **成交量指标** | 成交量均线、量比 | 成交量统计 | volume |
| **价格位置** | 相对均线位置、相对高低点位置 | 价格比率 | close, high, low |

**数据来源**: Baostock (query_history_k_data_plus)
**成本**: 免费

---

### 1.2 基本面指标（需要财务数据）

基本面指标需要获取公司的财务报表数据。

#### 1.2.1 盈利能力指标（15分）

| 指标 | 英文 | 计算公式 | 数据来源 | Tushare接口 |
|------|------|---------|---------|------------|
| 净资产收益率 | ROE | 净利润 / 股东权益 | 利润表、资产负债表 | fina_indicator (roe) |
| 投入资本回报率 | ROIC | EBIT / 投入资本 | 利润表、资产负债表 | fina_indicator (roe) |
| 盈利稳定性 | - | 过去8季度ROE标准差 | 利润表（历史） | fina_indicator (历史数据) |

**Tushare接口**:
- `fina_indicator` - 财务指标接口
- `income` - 利润表
- `balancesheet` - 资产负债表

#### 1.2.2 成长性指标（15分）

| 指标 | 英文 | 计算公式 | 数据来源 | Tushare接口 |
|------|------|---------|---------|------------|
| 营收增长率 | Revenue Growth | (本期营收-上期营收)/上期营收 | 利润表 | fina_indicator (or_yoy) |
| 利润增长率 | EPS Growth | (本期EPS-上期EPS)/上期EPS | 利润表 | fina_indicator (eps_yoy) |
| 经营现金流增长率 | Cash Flow Growth | 现金流增长率 | 现金流量表 | cashflow |
| 毛利率 | Gross Margin | (营收-成本)/营收 | 利润表 | fina_indicator (grossprofit_margin) |

**Tushare接口**:
- `fina_indicator` - 财务指标（增长率）
- `cashflow` - 现金流量表

#### 1.2.3 估值指标（10分）

| 指标 | 英文 | 数据来源 | Tushare接口 | Baostock字段 |
|------|------|---------|------------|--------------|
| 市盈率 | PE (TTM) | 日线数据 | daily_basic | peTTM |
| 市净率 | PB (MRQ) | 日线数据 | daily_basic | pbMRQ |
| 市销率 | PS (TTM) | 日线数据 | daily_basic | psTTM |
| 市现率 | PCF (TTM) | 日线数据 | daily_basic | pcfNcfTTM |
| PEG | PEG | PE/增长率 | 计算 | - |

**数据来源**:
- Tushare: `daily_basic` (更权威)
- Baostock: `query_history_k_data_plus` (免费)

#### 1.2.4 财务质量指标（10分）

| 指标 | 英文 | 计算公式 | 数据来源 | Tushare接口 |
|------|------|---------|---------|------------|
| 资产负债率 | Debt Ratio | 总负债/总资产 | 资产负债表 | fina_indicator (debt_to_assets) |
| 现金流质量 | CF Quality | 经营现金流/净利润 | 利润表、现金流量表 | fina_indicator (cf_to_netprofit) |
| 存货周转率 | Inventory Turnover | 营业成本/平均存货 | 利润表、资产负债表 | fina_indicator (inv_turn) |
| 应收账款周转率 | AR Turnover | 营业收入/平均应收账款 | 利润表、资产负债表 | fina_indicator (ar_turn) |
| 分红率 | Dividend Ratio | 每股股利/每股收益 | 分红数据、利润表 | fina_indicator (dividend_ratio) |

**Tushare接口**:
- `fina_indicator` - 财务指标
- `balancesheet` - 资产负债表
- `dividend` - 分红数据

---

### 1.3 市场广度指标（需全市场数据）

| 指标 | 计算公式 | 数据来源 | 获取方式 |
|------|---------|---------|---------|
| 上涨家数比例 | 上涨股票数 / 总股票数 | 全市场行情 | 遍历所有股票统计 |
| 成交量比率 | 上涨股票成交量 / 总成交量 | 全市场行情 | 遍历所有股票统计 |
| 涨停家数 | 涨停股票数量 | 涨跌停数据 | Tushare: limit_list |
| 跌停家数 | 跌停股票数量 | 涨跌停数据 | Tushare: limit_list |
| 换手率分位数 | 换手率在历史中的位置 | 换手率历史数据 | 计算 |

**数据来源**:
- **首选**: 自行计算（基于Baostock全市场数据）
- **备选**: Tushare `limit_list` (需要积分)

---

### 1.4 情绪指标（需要特殊数据）

| 指标 | 英文 | 数据来源 | Tushare接口 |
|------|------|---------|------------|
| 融资余额变化率 | Margin Change | 融资融券数据 | margin |
| 北向资金净流入 | Northbound Flow | 沪深港通数据 | moneyflow_hsgt |
| 涨停家数/跌停家数 | Limit Up/Down | 涨跌停数据 | limit_list |
| 散户参与度指标 | - | 需要数据建模 | 暂无 |

**Tushare接口**:
- `margin` - 融资融券数据（需要积分）
- `moneyflow_hsgt` - 沪深港通资金流向（需要积分）
- `limit_list` - 涨跌停列表（需要积分）

---

### 1.5 资金流向指标

| 指标 | 英文 | 数据来源 | Tushare接口 |
|------|------|---------|------------|
| 主力资金净流入 | Main Fund Inflow | 大单数据 | moneyflow |
| 北向资金持股比例 | Northbound Holding | 沪深港通持股 | hk_hold |
| 大单净买入 | Large Buy | 大单数据 | moneyflow |

**Tushare接口**:
- `moneyflow` - 资金流向（大单、中单、小单）（需要积分）
- `hk_hold` - 沪深港通持股（需要积分）

---

### 1.6 公司治理指标

| 指标 | 数据来源 | Tushare接口 |
|------|---------|------------|
| 实际控制人信息 | 公司基本信息 | stock_basic |
| 股权质押比例 | 质押数据 | pledge |
| 股东结构 | 股东信息 | shareholder, top10_holders |
| 管理层稳定性 | 高管信息 | managers |
| 审计意见 | 审计报告 | fina_audit |
| 监管处罚 | 处罚记录 | 无公开接口 |

**Tushare接口**:
- `stock_basic` - 股票基本信息
- `pledge` - 股权质押
- `shareholder` - 股东信息
- `top10_holders` - 前十大股东
- `managers` - 高管信息
- `fina_audit` - 审计意见

---

### 1.7 行业和指数数据

| 指标 | 数据来源 | Tushare接口 | Baostock |
|------|---------|------------|----------|
| 沪深300指数OHLCV | 指数数据 | index_daily | query_history_k_data_plus |
| 行业指数OHLCV | 指数数据 | index_daily | query_history_k_data_plus |
| 行业分类 | 行业信息 | index_classify | 无 |
| 指数成分股 | 成分股列表 | index_member | 无 |

**数据来源**:
- **首选**: Baostock (免费)
- **备选**: Tushare `index_daily`, `index_classify` (需要积分)

---

## 二、供应商接口清单

### 2.1 Baostock（免费推荐）

#### 优势
- ✅ 完全免费
- ✅ 无需注册
- ✅ 历史数据完整（从1990年起）
- ✅ 数据质量较高
- ✅ 支持股票和指数数据

#### 可用接口

| 接口 | 功能 | 是否必需 |
|------|------|---------|
| `query_history_k_data_plus` | 股票日线OHLCV | ✅ 必需 |
| `query_history_k_data_plus` | 指数日线OHLCV | ✅ 必需 |
| `query_stock_basic` | 证券基本信息 | ⚠️ 可选 |

#### 数据字段

**股票日线数据**:
- 日期、代码、开盘、最高、最低、收盘、前收盘
- 成交量、成交额、复权因子
- 换手率、涨跌幅
- **估值指标**: peTTM, pbMRQ, psTTM, pcfNcfTTM

**指数日线数据**:
- 日期、代码、开盘、最高、最低、收盘、前收盘
- 成交量、成交额、涨跌幅

#### 使用示例

```python
import baostock as bs

# 登录
bs.login()

# 获取股票数据
rs = bs.query_history_k_data_plus(
    "sh.600000",
    "date,code,open,high,low,close,preclose,volume,amount,peTTM,pbMRQ",
    start_date="2020-01-01",
    end_date="2025-12-31",
    frequency="d",
    adjustflag="2"  # 后复权
)

# 获取指数数据
rs = bs.query_history_k_data_plus(
    "sh.000300",
    "date,code,open,high,low,close,volume,amount",
    start_date="2020-01-01",
    end_date="2025-12-31",
    frequency="d"
)

# 登出
bs.logout()
```

---

### 2.2 Tushare Pro（需要积分）

#### 优势
- ✅ 数据最权威
- ✅ 指标最全面
- ✅ 财务数据完整
- ✅ 更新及时

#### 缺点
- ❌ 需要积分（大部分接口需要付费）
- ❌ 需要注册

#### 可用接口

##### 2.2.1 基础行情接口

| 接口 | 功能 | 数据字段 | 必需性 |
|------|------|---------|--------|
| `daily` | 日线行情 | open, high, low, close, volume, amount, pct_chg | ⚠️ 可选 |
| `adj_factor` | 复权因子 | adj_factor | ❌ 可选 |
| `weekly` | 周线行情 | 周线OHLCV | ❌ 可选 |
| `monthly` | 月线行情 | 月线OHLCV | ❌ 可选 |

##### 2.2.2 财务数据接口

| 接口 | 功能 | 数据字段 | 必需性 |
|------|------|---------|--------|
| `fina_indicator` | 财务指标 | roe, roic, debt_to_assets, grossprofit_margin | ✅ 推荐 |
| `income` | 利润表 | 营业收入、净利润、每股收益 | ⚠️ 可选 |
| `balancesheet` | 资产负债表 | 总资产、总负债、股东权益 | ⚠️ 可选 |
| `cashflow` | 现金流量表 | 经营现金流 | ⚠️ 可选 |
| `fina_audit` | 审计意见 | 审计意见类型 | ❌ 可选 |
| `dividend` | 分红数据 | 分红率、分红金额 | ❌ 可选 |

##### 2.2.3 估值数据接口

| 接口 | 功能 | 数据字段 | 必需性 |
|------|------|---------|--------|
| `daily_basic` | 日线基本面指标 | pe_ttm, pb_mrq, ps_ttm, pcf_ncf_ttm | ⚠️ 可选 |
| `valuation_indicator` | 估值指标分位数 | 估值分位数 | ❌ 可选 |

##### 2.2.4 市场广度接口

| 接口 | 功能 | 数据字段 | 必需性 |
|------|------|---------|--------|
| `limit_list` | 涨跌停列表 | 涨跌停股票 | ⚠️ 可选 |
| `limit_list_d` | 涨跌停详情 | 详细涨跌停信息 | ❌ 可选 |

##### 2.2.5 资金流向接口

| 接口 | 功能 | 数据字段 | 必需性 |
|------|------|---------|--------|
| `moneyflow_hsgt` | 沪深港通资金流向 | ggt_ss, ggt_sz, north_money | ⚠️ 可选 |
| `moneyflow_hsgt_hist` | 北向资金历史 | 历史北向资金 | ❌ 可选 |
| `margin` | 融资融券数据 | 融资余额、融券余额 | ⚠️ 可选 |
| `margin_detail` | 融资融券明细 | 详细融资融券数据 | ❌ 可选 |
| `moneyflow` | 资金流向（大单） | 大单净流入、中单净流入、小单净流入 | ❌ 可选 |

##### 2.2.6 板块数据接口

| 接口 | 功能 | 数据字段 | 必需性 |
|------|------|---------|--------|
| `index_basic` | 指数基本信息 | 指数代码、名称 | ⚠️ 可选 |
| `index_daily` | 指数日线行情 | 指数OHLCV | ⚠️ 可选 |
| `index_classify` | 行业分类 | 申万、中信行业分类 | ❌ 可选 |
| `index_member` | 指数成分股 | 成分股列表 | ❌ 可选 |

##### 2.2.7 公司信息接口

| 接口 | 功能 | 数据字段 | 必需性 |
|------|------|---------|--------|
| `stock_basic` | 股票基本信息 | 行业、上市日期、ST标记 | ⚠️ 可选 |
| `pledge` | 股权质押 | 质押比例、质押数量 | ❌ 可选 |
| `shareholder` | 股东信息 | 股东人数、股东户数 | ❌ 可选 |
| `hold_control` | 控制股东 | 实际控制人信息 | ❌ 可选 |
| `top10_holders` | 前十大股东 | 前十大股东持股 | ❌ 可选 |
| `managers` | 高管信息 | 高管姓名、职位 | ❌ 可选 |
| `st` | ST标记历史 | ST标记历史 | ❌ 可选 |

#### 使用示例

```python
import tushare as ts

# 初始化
ts.set_token("your_token")
pro = ts.pro_api()

# 获取财务指标
df = pro.fina_indicator(
    ts_code="600000.SH",
    start_date="20240101",
    end_date="20241231"
)

# 获取估值数据
df = pro.daily_basic(
    ts_code="600000.SH",
    start_date="20240101",
    end_date="20241231"
)

# 获取北向资金
df = pro.moneyflow_hsgt(
    start_date="20240101",
    end_date="20241231"
)
```

---

### 2.3 新浪/腾讯（免费实时数据）

#### 优势
- ✅ 完全免费
- ✅ 实时数据
- ✅ 无需注册

#### 可用接口

| 接口 | 功能 | 数据字段 | 必需性 |
|------|------|---------|--------|
| `https://qt.gtimg.cn/q=` | 实时行情 | 实时价格、成交量 | ⚠️ 可选 |

#### 使用示例

```python
import requests

# 获取沪深300实时行情
url = "https://qt.gtimg.cn/q=sh000300"
response = requests.get(url)
print(response.text)

# 批量获取
url = "https://qt.gtimg.cn/q=sh000300,sz000001"
response = requests.get(url)
print(response.text)
```

---

## 三、推荐数据获取方案

### 3.1 最小化方案（免费，覆盖90%功能）

**数据来源**: Baostock

**可获取数据**:
- ✅ 股票日线OHLCV数据
- ✅ 指数日线OHLCV数据
- ✅ 估值指标（PE_TTM, PB_MRQ）
- ✅ 所有技术指标（自行计算）
- ✅ 市场广度（自行计算）

**无法获取数据**:
- ❌ 详细财务指标（ROE, ROIC等）
- ❌ 北向资金
- ❌ 融资融券
- ❌ 资金流向

**实施步骤**:
1. 使用Baostock获取股票和指数日线数据
2. 自行计算所有技术指标
3. 自行统计市场广度（涨跌家数）
4. 使用估值指标替代部分财务指标

**成本**: 免费

---

### 3.2 标准方案（需要Tushare积分，覆盖100%功能）

**数据来源**: Baostock + Tushare Pro

**数据分配**:

| 数据类型 | 首选来源 | 备选来源 |
|---------|---------|---------|
| 股票OHLCV | Baostock | Tushare |
| 指数OHLCV | Baostock | Tushare |
| 估值指标 | Baostock | Tushare |
| 财务指标 | Tushare | - |
| 北向资金 | Tushare | - |
| 融资融券 | Tushare | - |
| 市场广度 | 自行计算 | Tushare |

**可获取数据**:
- ✅ 所有技术指标
- ✅ 完整基本面指标
- ✅ 市场广度指标
- ✅ 情绪指标
- ✅ 资金流向
- ✅ 公司治理指标

**成本**: Tushare积分（约500积分/年）

---

### 3.3 高级方案（付费，功能最全）

**数据来源**: Baostock + Tushare Pro + Wind/Choice

**可获取数据**:
- ✅ 所有标准方案数据
- ✅ 更高频数据（分钟线）
- ✅ 更详细的财务数据
- ✅ 行业研究报告
- ✅ 宏观经济数据

**成本**: Tushare积分 + Wind/Choice费用（数万元/年）

---

## 四、实施建议

### 4.1 第一阶段（当前）：技术面+基础基本面

**目标**: 完成技术面评分和基础趋势判断

**数据需求**:
- Baostock: 股票和指数日线数据
- 自行计算: 所有技术指标
- Baostock: 估值指标（peTTM, pbMRQ）

**实施内容**:
1. ✅ 技术面评分系统（30分）
2. ✅ 趋势判断系统
3. ✅ 买卖信号检测
4. ⚠️ 基础基本面评分（使用估值指标代理，降低权重）

**完成度**: 70%

---

### 4.2 第二阶段：完善基本面评分

**目标**: 集成真实财务数据

**数据需求**:
- Tushare: `fina_indicator` - 财务指标
- Tushare: `daily_basic` - 估值数据

**实施内容**:
1. ✅ 盈利能力评分（15分）
2. ✅ 成长性评分（15分）
3. ✅ 财务质量评分（10分）

**完成度**: 90%

---

### 4.3 第三阶段：完善情绪和市场广度

**目标**: 集成情绪指标和资金流向

**数据需求**:
- Tushare: `moneyflow_hsgt` - 北向资金
- Tushare: `margin` - 融资融券
- 自行计算: 市场广度

**实施内容**:
1. ✅ 市场广度指标
2. ✅ 情绪指标
3. ✅ 资金流向指标
4. ✅ 风险质量评分完善

**完成度**: 100%

---

## 五、接口使用优先级

### 5.1 优先级1（必须）

| 数据类型 | 推荐来源 | 接口 | 原因 |
|---------|---------|------|------|
| 股票日线OHLCV | Baostock | query_history_k_data_plus | 免费、完整 |
| 指数日线OHLCV | Baostock | query_history_k_data_plus | 免费、完整 |

### 5.2 优先级2（重要）

| 数据类型 | 推荐来源 | 接口 | 原因 |
|---------|---------|------|------|
| 财务指标 | Tushare | fina_indicator | 基本面评分必需 |
| 估值数据 | Tushare | daily_basic | 更权威 |
| 市场广度 | 自行计算 | - | 基于Baostock数据 |

### 5.3 优先级3（可选）

| 数据类型 | 推荐来源 | 接口 | 原因 |
|---------|---------|------|------|
| 北向资金 | Tushare | moneyflow_hsgt | 情绪指标 |
| 融资融券 | Tushare | margin | 情绪指标 |
| 资金流向 | Tushare | moneyflow | 主力资金 |
| 行业指数 | Tushare | index_daily | 行业轮动 |

---

## 六、总结

### 6.1 核心结论

1. **技术指标可自行计算**，无需外部API
2. **基础行情数据使用Baostock**（免费、完整）
3. **财务数据使用Tushare Pro**（最完整，但需要积分）
4. **市场广度可自行计算**（基于Baostock全市场数据）
5. **实时行情可使用新浪/腾讯**（免费、实时）

### 6.2 最小化方案

- Baostock: 股票和指数日线数据
- 自行计算: 所有技术指标和市场广度指标
- 结果: 可完成70%的评分功能

### 6.3 完整方案

- Baostock: 基础行情数据
- Tushare Pro: 财务数据、估值数据、北向资金
- 自行计算: 技术指标、市场广度
- 结果: 可完成100%的评分和判断功能

### 6.4 实施建议

**第一阶段**: 使用Baostock + 自行计算，完成技术面和基础趋势判断
**第二阶段**: 集成Tushare Pro财务数据，完善基本面评分
**第三阶段**: 集成Tushare Pro市场广度数据，完善情绪指标

---

## 七、附录：数据字段映射

### 7.1 Baostock字段映射

| Baostock字段 | 说明 | 评分用途 |
|-------------|------|---------|
| date | 日期 | - |
| code | 股票代码 | - |
| open | 开盘价 | 技术指标 |
| high | 最高价 | 技术指标 |
| low | 最低价 | 技术指标 |
| close | 收盘价 | 技术指标 |
| preclose | 前收盘价 | 计算涨跌幅 |
| volume | 成交量 | 技术指标 |
| amount | 成交额 | 技术指标 |
| adjustflag | 复权类型 | - |
| turn | 换手率 | 技术指标 |
| tradestatus | 交易状态 | 过滤停牌 |
| pctChg | 涨跌幅 | 技术指标 |
| peTTM | 市盈率TTM | 估值评分 |
| pbMRQ | 市净率MRQ | 估值评分 |
| psTTM | 市销率TTM | 估值评分 |
| pcfNcfTTM | 市现率TTM | 估值评分 |

### 7.2 Tushare字段映射

| Tushare接口 | 字段 | 说明 | 评分用途 |
|------------|------|------|---------|
| fina_indicator | roe | 净资产收益率 | 盈利能力 |
| fina_indicator | roe_dt | 净资产收益率季度同比 | 盈利能力 |
| fina_indicator | roe_waa | 加权平均净资产收益率 | 盈利能力 |
| fina_indicator | roa | 总资产净利率 | 盈利能力 |
| fina_indicator | npta | 净利润/总资产 | 盈利能力 |
| fina_indicator | debt_to_assets | 资产负债率 | 财务质量 |
| fina_indicator | assets_turn | 总资产周转率 | 财务质量 |
| fina_indicator | ca_turn | 流动资产周转率 | 财务质量 |
| fina_indicator | fa_turn | 固定资产周转率 | 财务质量 |
| fina_indicator | inv_turn | 存货周转率 | 财务质量 |
| fina_indicator | ar_turn | 应收账款周转率 | 财务质量 |
| fina_indicator | grossprofit_margin | 销售毛利率 | 财务质量 |
| fina_indicator | netprofit_margin | 销售净利率 | 财务质量 |
| fina_indicator | current_ratio | 流动比率 | 财务质量 |
| fina_indicator | quick_ratio | 速动比率 | 财务质量 |
| fina_indicator | cf_to_sales | 销售现金比率 | 财务质量 |
| fina_indicator | cf_to_netprofit | 净利润现金含量 | 财务质量 |
| fina_indicator | or_yoy | 营业收入同比增长率 | 成长性 |
| fina_indicator | ebitda_yoy | EBITDA同比增长率 | 成长性 |
| fina_indicator | ocf_yoy | 经营活动现金流同比增长率 | 成长性 |
| fina_indicator | roe_yoy | 净资产收益率同比增长率 | 成长性 |
| fina_indicator | bps_yoy | 每股净资产同比增长率 | 成长性 |
| fina_indicator | eps_yoy | 基本每股收益同比增长率 | 成长性 |
| daily_basic | pe_ttm | 市盈率TTM | 估值 |
| daily_basic | pe | 市盈率 | 估值 |
| daily_basic | pb_mrq | 市净率 | 估值 |
| daily_basic | ps_ttm | 市销率 | 估值 |
| daily_basic | pcf_ncf_ttm | 市现率 | 估值 |
| moneyflow_hsgt | ggt_ss | 沪股通当日买入净额 | 资金流向 |
| moneyflow_hsgt | ggt_sz | 深股通当日买入净额 | 资金流向 |
| moneyflow_hsgt | north_money | 北向资金净流入 | 资金流向 |
| margin | rzche | 融资余额 | 融资融券 |
| margin | rqche | 融券余额 | 融资融券 |

---

**文档版本**: v1.0
**更新日期**: 2026-02-08
**作者**: iFlow CLI
