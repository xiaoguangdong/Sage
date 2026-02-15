# sage_core 目录说明

`sage_core` 按业务域拆分，主目录如下：

- `industry/`：行业与宏观相关模型（行业分类、周期识别、宏观信号）
- `trend/`：趋势状态识别
- `stock_selection/`：选股与排序（单模型、多策略、评分）
- `execution/`：买卖点与执行过滤
- `governance/`：Champion/Challenger 策略治理
- `portfolio/`：组合构建与风控
- `backtest/`：回测引擎

兼容层：

- `models/` 目录保留为向后兼容导入入口，内部已转发到以上新目录。
