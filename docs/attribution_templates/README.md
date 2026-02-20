# 归因输入模板说明

本目录提供归因分析所需 CSV 模板。你只需要把数据填进去即可。

## 1) Brinson 归因

文件：
- `portfolio_template.csv`（组合）
- `benchmark_template.csv`（基准）

字段：
- `trade_date`：日期（建议与回测周期一致，例如周度）
- `industry_l1`：行业（申万一级）
- `weight`：该行业在组合/基准中的权重（建议同日加总为 1）
- `return`：该行业在该期的收益

用途：分解超额收益为「配置效应/选股效应/交互效应」。

## 2) 因子归因

文件：
- `factor_exposure_template.csv`

字段：
- `trade_date`：日期
- `ts_code`：股票代码
- `weight`：组合内权重（建议同日加总为 1）
- `return`：该股票在该期收益
- `factor_*`：任意数量的因子暴露列（例如 `factor_value`/`factor_momentum`）

用途：估计每个因子的「因子收益」并计算组合的因子贡献。
