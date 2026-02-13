# Scripts (modular layout)

建议按模块存放脚本，逐步从历史脚本迁移到以下目录：

- `scripts/data/` 数据下载、清洗、对齐
- `scripts/models/` 模型训练/预测
- `scripts/strategy/` 信号融合、策略编排
- `scripts/backtest/` 回测运行
- `scripts/monitoring/` 监控与报告

历史脚本（如 `scripts/macro/`）先保留，后续逐步迁移。

