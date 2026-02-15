# `sage_core.models` 说明

`sage_core/models/` 仅用于兼容历史导入路径，不再承载新实现。

## 新代码入口

- 趋势：`sage_core/trend/`
- 行业/宏观：`sage_core/industry/`
- 选股：`sage_core/stock_selection/`
- 执行：`sage_core/execution/`
- 治理：`sage_core/governance/`

## 迁移原则

- 新功能只放业务域目录，不新增 `sage_core/models/*.py`。
- 老脚本可继续通过 `sage_core.models.*` 访问，后续按需逐步替换。
