# Agent任务清单

## 待完成任务

### Phase 0：数据准备

- [ ] 检查Baostock数据缺失值
- [ ] 检查Baostock数据异常值
- [ ] 下载Tushare的指数数据（沪深300、中证500）
- [ ] 对比Baostock和Tushare的价格数据
- [ ] 生成数据质量报告
- [ ] EDA分析（数据质量、价格分析、特征分析、标签分析）

### 代码完善

- [ ] 实现run_weekly.py中的训练逻辑（2处TODO）
- [ ] 实现portfolio/risk_control.py中的行业暴露调整逻辑
- [ ] 实现portfolio/construction.py中的权重调整逻辑
- [ ] 完善TrendModelHMM类的实现（刚才已经开始但未完成）

### Phase 2：优化迭代

- [ ] 趋势模型：完善LightGBM版本（已添加打标和训练逻辑）
- [ ] 趋势模型：完善HMM版本（需要完成replace操作）
- [ ] IC动态权重：启用IC动态权重
- [ ] 评估指标：ICIR、Top-K命中率

### Phase 3：完整系统整合

- [ ] 创建ml_stock_forecast/strategies/entry_strategy.py - 入场策略
- [ ] 创建ml_stock_forecast/strategies/portfolio_construction.py - 组合构建
- [ ] 创建ml_stock_forecast/strategies/risk_control.py - 风险控制

### 数据处理优化

- [ ] 创建标签生成脚本（ml_stock_forecast/scripts/label_generator.py）
- [ ] 创建EDA分析脚本（ml_stock_forecast/scripts/eda_analysis.py）
- [ ] 实现增量数据处理流程
- [ ] 优化数据加载性能

### 测试和验证

- [ ] 添加LightGBM模型的测试（需要安装lightgbm）
- [ ] 添加HMM模型的测试（需要安装hmmlearn）
- [ ] 添加回测功能的测试
- [ ] 添加特征工程的测试

### 文档

- [ ] 更新README.md，补充完整的使用说明
- [ ] 创建API文档
- [ ] 创建部署文档

## 已完成的工作

### 核心模块

- [x] 创建ml_stock_forecast目录结构
- [x] 开发data模块（data_loader.py, universe.py）
- [x] 开发features模块（price_features.py, market_features.py）
- [x] 开发models模块（trend_model.py, rank_model.py, entry_model.py）
- [x] 开发portfolio模块（construction.py, risk_control.py）
- [x] 开发backtest模块（walk_forward.py）
- [x] 开发utils模块（data_loader.py）
- [x] 开发run_weekly.py主入口
- [x] 开发config配置文件（trend_model.yaml, rank_model.yaml, entry_model.yaml）
- [x] 编写测试类（13个测试全部通过）

### 趋势模型打标

- [x] 添加TrendModelRule.create_labels()方法
- [x] 实现TrendModelLGBM的完整训练和预测逻辑
- [ ] 实现TrendModelHMM的完整训练和预测逻辑（进行中）

### 文档

- [x] 创建README.md
- [x] 创建DEVELOPMENT_PROGRESS.md
- [x] 创建agent.md（本文件）

## 下一步计划

1. **立即执行**：检查Baostock数据缺失值和异常值
2. **短期目标**：完成EDA分析，生成数据质量报告
3. **中期目标**：完善代码中的TODO项，实现Phase 2功能
4. **长期目标**：完整系统整合，实盘部署

## 注意事项

- LightGBM和hmmlearn是可选依赖，如果需要使用需要安装
- 所有趋势模型都基于沪深300指数（000300.SH）
- Walk-forward回测：2.5年训练，半年测试，每半年滚动
- 防过拟合：模型越简单越好，禁止全样本fit