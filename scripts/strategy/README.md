# Strategy scripts

信号融合、策略编排与组合构建脚本放在这里。

## 行业信号正式链路

```bash
# 1) 概念热度 -> 行业偏置
python scripts/strategy/build_industry_concept_bias.py --top-k 10

# 2) 行业信号统一契约（policy/concept/northbound）
python scripts/strategy/build_industry_signal_contract.py
```

输出：
- `data/signals/industry/industry_concept_bias.parquet`
- `data/signals/industry/industry_signal_contract.parquet`
- `data/signals/industry/industry_signal_snapshot_latest.parquet`

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
