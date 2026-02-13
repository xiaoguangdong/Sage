# A股量化交易预测系统

## 项目简介

这是一个基于机器学习的A股量化交易预测系统，采用三模型架构：
1. **趋势状态模型** - 基于沪深300指数判断市场趋势（RISK_ON/震荡/RISK_OFF）
2. **选股排序模型** - 使用LightGBM对股票进行排序
3. **买卖点过滤模型** - 使用Logistic Regression识别买卖时机

## 系统架构

```
ml_stock_forecast/
├── config/              # 配置文件
│   ├── trend_model.yaml
│   ├── rank_model.yaml
│   └── entry_model.yaml
├── data/                # 数据目录
│   ├── raw/             # 原始数据
│   └── processed/       # 处理后数据
├── features/            # 特征工程
│   ├── price_features.py
│   └── market_features.py
├── models/              # 模型
│   ├── trend_model.py
│   ├── rank_model.py
│   └── entry_model.py
├── portfolio/           # 组合管理
│   ├── construction.py
│   └── risk_control.py
├── backtest/            # 回测
│   └── walk_forward.py
├── utils/               # 工具
│   └── data_loader.py
├── tests/               # 测试
└── run_weekly.py        # 主入口
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
cd ml_stock_forecast
python -m unittest discover tests -v
```

### 每周运行

```bash
cd ml_stock_forecast
python run_weekly.py weekly
```

### 运行回测

```bash
cd ml_stock_forecast
python run_weekly.py backtest
```

## 数据准备

### Baostock数据

1. 从Baostock下载股票数据
2. 将数据保存到 `data/raw/` 目录
3. 文件格式：`{stock_code}.parquet`

### 数据格式要求

```python
{
    'date': datetime,    # 交易日期
    'code': str,         # 股票代码
    'open': float,       # 开盘价
    'high': float,       # 最高价
    'low': float,        # 最低价
    'close': float,      # 收盘价
    'volume': int,       # 成交量
    'turnover': float,   # 换手率
    'amount': float      # 成交额
}
```

## 模型说明

### 趋势状态模型（规则版）

基于沪深300指数的20个硬核指标：
- 均线系统：MA20、MA60、MA120
- 波动率：4周、12周、20周波动率
- MACD：DIF、DEA、MACD柱
- 价格位置：相对MA位置、相对区间位置
- 动量：动量指标、趋势强度
- 成交量：成交量变化率

### 选股排序模型（LightGBM）

使用20个因子：
- 动量因子：4周、12周、20周收益率
- 流动性因子：换手率均值、标准差
- 稳定性因子：波动率、最大回撤
- 技术因子：RSI、MACD、布林带

参数限制：
- max_depth ≤ 4
- num_leaves ≤ 16
- 强正则化

### 买卖点过滤模型（Logistic Regression）

使用20个特征：
- 价格位置特征：相对MA位置、相对区间位置
- 成交量分析特征：成交量均值、量价相关性
- MACD特征：MACD柱、金叉信号
- K线形态特征：十字星、锤子线、射击之星
- 波动率特征：5天、10天波动率
- 趋势强度特征：20天趋势强度

触发规则：
- 模型概率 > 0.6
- 价格 > MA20
- 波动率 < 5%

## Walk-forward回测

- 训练窗口：2.5年（约130周）
- 测试窗口：半年（约26周）
- 滚动步长：半年（约26周）

## 风险控制

- 止损：-8%
- 最大回撤：-15%
- 单只股票最大仓位：5%
- 单个行业最大暴露：30%

## 配置说明

### trend_model.yaml

```yaml
trend_model:
  enabled: true
  model_type: rule  # rule / lgbm / hmm
  use_all_features: false
```

### rank_model.yaml

```yaml
rank_model:
  enabled: true
  model_type: lgbm
  enable_ic_weight: false  # Phase 1先不实现

lgbm_params:
  objective: 'lambdarank'
  num_leaves: 16
  max_depth: 4
  learning_rate: 0.05

walk_forward:
  train_weeks: 130
  test_weeks: 26
  roll_step: 26
```

### entry_model.yaml

```yaml
entry_model:
  enabled: true
  model_type: lr  # Logistic Regression
  use_all_features: true

rules:
  prob_threshold: 0.6
  ma_period: 20
  vol_threshold: 0.05

label:
  horizon: 10
  return_threshold: 0.05  # 5%
  max_drawdown: -0.03  # -3%
```

## 开发进度

- [x] 创建目录结构
- [x] 开发data模块
- [x] 开发features模块
- [x] 开发models模块
- [x] 开发portfolio模块
- [x] 开发backtest模块
- [x] 开发utils模块
- [x] 开发run_weekly.py主入口
- [x] 编写测试类
- [ ] 检查Baostock数据缺失值
- [ ] 检查Baostock数据异常值
- [ ] 下载Tushare的指数数据
- [ ] 对比Baostock和Tushare的价格数据

## 注意事项

1. **LightGBM依赖**：如果需要使用排序模型，需要安装LightGBM：
   ```bash
   pip install lightgbm
   ```

2. **数据质量**：确保数据格式正确，没有缺失值

3. **防过拟合**：
   - 模型越简单越好
   - 数据先于模型
   - 禁止全样本fit
   - 使用Walk-forward回测

4. **沪深300关系**：所有趋势模型指标基于沪深300指数（000300.SH）

## License

MIT License