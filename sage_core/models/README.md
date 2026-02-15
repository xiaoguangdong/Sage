# `sage_core.models` 说明

`sage_core/models/` 作为兼容层保留，并按业务域重构为以下目录：

- `sage_core/models/industry/`
- `sage_core/models/trend/`
- `sage_core/models/stock_selection/`
- `sage_core/models/execution/`
- `sage_core/models/governance/`

根目录中的历史模块（如 `trend_model.py`、`rank_model.py`）仅作为向后兼容入口，实际转发到对应业务域子目录。

## 新代码入口

- 趋势：`sage_core/trend/`
- 行业/宏观：`sage_core/industry/`
- 选股：`sage_core/stock_selection/`
- 执行：`sage_core/execution/`
- 治理：`sage_core/governance/`
