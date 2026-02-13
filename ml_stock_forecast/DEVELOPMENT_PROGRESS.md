# A股量化交易预测系统 - 开发进度报告

## 已完成的工作

### 1. 项目结构搭建 ✅

```
sage_core/
├── backtest/
├── data/
├── features/
├── models/
├── portfolio/
└── utils/

ml_stock_forecast/
├── non_core/            # 非核心模块（数据接入/调度）
│   ├── config/
│   ├── data/
│   └── pipelines/
├── tests/               # 测试目录
│   ├── __init__.py
│   ├── test_data_loader.py  # 数据加载器测试
│   ├── test_trend_model.py  # 趋势模型测试
│   └── test_models.py       # 排序和买卖点模型测试
└── README.md            # 项目说明文档
```

### 2. 核心模块开发 ✅

#### 2.1 数据模块 (sage_core/data/ + non_core/data/)

**data_loader.py**
- `DataLoader` 类：负责加载Baostock数据
- `check_data_quality()` 方法：检查数据质量（缺失值、重复值、异常值等）
- 支持单个股票和批量股票数据加载

**universe.py**
- `Universe` 类：股票池筛选
- 支持ST股票排除、停牌股票排除
- 支持换手率、市值过滤

#### 2.2 特征工程模块 (sage_core/features/)

**price_features.py**
- `PriceFeatures` 类：价格特征提取器
- 动量特征：mom_4w, mom_12w, mom_20w, relative_strength
- 流动性特征：turnover_4w, turnover_12w, volume_trend
- 稳定性特征：vol_4w, vol_12w, max_dd_4w, max_dd_12w
- 技术指标：MA斜率、RSI

**market_features.py**
- `MarketFeatures` 类：市场特征提取器（基于沪深300指数）
- 8个基础特征：MA20/60/120、4周波动率、12周波动率、MACD指标、价格位置
- 10个硬核特征：均线多头排列、波动率分位、MACD金叉、价格创新高、量价背离等

#### 2.3 模型模块 (sage_core/models/)

**trend_model.py**
- `TrendModelRule` 类：规则版趋势模型
- 基于沪深300指数的20个指标
- 输出三种状态：RISK_ON (2)、震荡 (1)、RISK_OFF (0)
- `TrendModelLGBM` 类：LightGBM版本（Phase 2）
- `TrendModelHMM` 类：HMM版本（Phase 3）
- `create_trend_model()` 工厂函数

**rank_model.py**
- `RankModelLGBM` 类：LightGBM排序模型
- 使用20个因子：动量、流动性、稳定性、技术指标
- LightGBM参数：max_depth=4, num_leaves=16, objective='lambdarank'
- `prepare_features()` 方法：准备排序特征
- `create_ranking_label()` 方法：创建排序标签（未来N天收益率排名）
- `train()` 方法：训练模型
- `predict()` 方法：预测股票排序
- `get_feature_importance()` 方法：获取特征重要性

**entry_model.py**
- `EntryModelLR` 类：Logistic回归买卖点模型
- 使用20个特征：价格位置、成交量分析、MACD、K线形态、波动率、趋势强度
- 触发规则：prob > 0.6 且 价格 > MA20 且 波动率 < 5%
- `prepare_features()` 方法：准备买卖点特征
- `create_entry_label()` 方法：创建买卖点标签（安全窗口识别）
- `train()` 方法：训练模型
- `predict()` 方法：预测买卖点
- `get_model_params()` 方法：获取模型参数

#### 2.4 组合管理模块 (sage_core/portfolio/)

**construction.py**
- `PortfolioConstruction` 类：组合构建器
- `construct_portfolio()` 方法：根据趋势状态构建组合
- `_construct_equal_weight_portfolio()` 方法：等权重组合
- `_construct_low_risk_portfolio()` 方法：低风险组合（RISK_OFF，30%仓位）
- `_construct_balanced_portfolio()` 方法：中等风险组合（震荡，60%仓位）
- `_construct_high_risk_portfolio()` 方法：高风险组合（RISK_ON，90%仓位）
- `apply_sector_limits()` 方法：应用行业限制
- `calculate_portfolio_returns()` 方法：计算组合收益

**risk_control.py**
- `RiskControl` 类：风险控制器
- `check_entry_signal()` 方法：检查买入信号是否应该被过滤
- `check_exit_signal()` 方法：检查是否需要止损或止盈
- `check_portfolio_risk()` 方法：检查组合风险
- `check_drawdown()` 方法：检查回撤是否超过限制
- `adjust_weights()` 方法：调整组合权重以满足风险限制
- `get_risk_metrics()` 方法：计算风险指标（波动率、回撤、VaR、CVaR）

#### 2.5 回测模块 (backtest/)

**walk_forward.py**
- `WalkForwardBacktest` 类：Walk-forward回测器
- `prepare_data_splits()` 方法：准备时间划分（2.5年训练 + 半年测试）
- `train_model()` 方法：训练模型
- `run_backtest()` 方法：运行完整回测
- `_run_test_window()` 方法：运行单个测试窗口
- `_calculate_metrics()` 方法：计算回测指标（总收益、年化收益、波动率、夏普比率、最大回撤、胜率、盈亏比）

#### 2.6 工具模块 (utils/)

**data_loader.py**
- `DataLoader` 类：数据加载器
- `load_baostock_data()` 方法：加载单个股票数据
- `load_all_baostock_data()` 方法：加载所有股票数据
- `check_data_quality()` 方法：检查数据质量

### 3. 主入口文件 ✅

**run_weekly.py**
- `load_config()` 函数：加载配置文件
- `load_data()` 函数：加载数据
- `filter_universe()` 函数：过滤股票池
- `calculate_features()` 函数：计算特征
- `run_weekly_workflow()` 函数：运行每周工作流
- `run_backtest_workflow()` 函数：运行回测工作流
- `main()` 函数：主函数，支持命令行参数（weekly/backtest）

### 4. 配置文件 ✅

**trend_model.yaml**
- 趋势模型配置
- 数据源配置（沪深300指数、Tushare）
- 规则参数（MA20/60/120）

**rank_model.yaml**
- 排序模型配置
- LightGBM参数（max_depth=4, num_leaves=16）
- Walk-forward参数（train_weeks=130, test_weeks=26, roll_step=26）
- IC动态权重开关（Phase 1先不实现）

**entry_model.yaml**
- 买卖点模型配置
- 触发规则（prob_threshold=0.6, ma_period=20, vol_threshold=0.05）
- 标签参数（horizon=10, return_threshold=0.05, max_drawdown=-0.03）

### 5. 测试代码 ✅

**test_data_loader.py**
- 测试数据加载器的基本功能
- 测试数据质量检查
- 测试股票池筛选

**test_trend_model.py**
- 测试趋势模型创建
- 测试RISK_ON状态预测
- 测试RISK_OFF状态预测
- 测试震荡状态预测
- 测试缺少必要列的错误处理

**test_models.py**
- 测试买卖点模型创建
- 测试买卖点模型训练和预测
- 测试特征准备
- 测试标签创建

**测试结果**：所有13个测试全部通过 ✅

### 6. 文档 ✅

**README.md**
- 项目简介
- 系统架构
- 快速开始指南
- 数据准备说明
- 模型说明
- Walk-forward回测说明
- 风险控制说明
- 配置说明
- 开发进度
- 注意事项

## 待完成的任务

### 1. 数据准备 ⏳

- [ ] 检查Baostock数据缺失值
- [ ] 检查Baostock数据异常值
- [ ] 下载Tushare的指数数据（沪深300、中证500）
- [ ] 对比Baostock和Tushare的价格数据

### 2. 功能增强 ⏳

- [ ] Phase 2：实现IC动态权重
- [ ] Phase 2：实现LightGBM趋势模型
- [ ] Phase 3：实现HMM趋势模型
- [ ] Phase 3：完整系统整合

### 3. 性能优化 ⏳

- [ ] 特征计算优化（向量化、并行化）
- [ ] 模型训练优化（增量学习、在线学习）
- [ ] 回测性能优化（向量化、缓存）

### 4. 监控和日志 ⏳

- [ ] 模型性能监控
- [ ] 交易日志记录
- [ ] 告警系统

## 技术栈

- **语言**：Python 3.14+
- **数据处理**：pandas, numpy
- **机器学习**：scikit-learn, lightgbm（可选）
- **配置管理**：pyyaml
- **测试框架**：unittest
- **日志**：logging

## 设计原则

1. **防过拟合**：模型越简单越好，数据先于模型，禁止全样本fit
2. **沪深300关系**：所有趋势模型指标基于沪深300指数（000300.SH）
3. **三模型架构**：趋势状态 → 选股排序 → 买卖点过滤
4. **Walk-forward回测**：2.5年训练，半年测试，每半年滚动
5. **风险控制**：止损、回撤、仓位限制、行业限制

## 注意事项

1. **LightGBM依赖**：如果需要使用排序模型，需要安装LightGBM
2. **数据质量**：确保数据格式正确，没有缺失值
3. **防过拟合**：
   - 模型越简单越好
   - 数据先于模型
   - 禁止全样本fit
   - 使用Walk-forward回测
4. **沪深300关系**：所有趋势模型指标基于沪深300指数（000300.SH）

## 总结

已成功构建了A股量化交易预测系统的核心框架，包括：
- ✅ 完整的项目结构和配置文件
- ✅ 数据加载、特征工程、模型训练、组合构建、风险控制、回测等核心模块
- ✅ 三模型架构：趋势状态模型、选股排序模型、买卖点过滤模型
- ✅ Walk-forward回测框架
- ✅ 完整的测试代码（13个测试全部通过）
- ✅ 详细的项目文档

系统已经可以运行基本的每周工作流和回测，下一步需要：
1. 准备高质量的数据
2. 进行参数调优和模型验证
3. 实现Phase 2和Phase 3的功能增强
4. 优化性能和监控系统
