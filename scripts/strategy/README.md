# Strategy scripts

信号融合、策略编排与组合构建脚本放在这里。

## 政策信号回测（MVP）

脚本：`scripts/strategy/policy_backtest.py`

依赖：
- `data/processed/policy/policy_signals.parquet`
- `data/tushare/daily/daily_*.parquet`
- `data/tushare/constituents/hs300_constituents_all.parquet`

示例：
```bash
python scripts/strategy/policy_backtest.py --top-industries 5 --cost-rate 0.005
```
