# Monitoring scripts

监控、告警、日报/周报生成脚本放在这里。

## 日志归档

根目录散落日志可用下述命令整理到 `logs/<module>/`：

```bash
python scripts/monitoring/organize_logs.py --dry-run
python scripts/monitoring/organize_logs.py
```

## 行业信号质量检查

检查行业信号覆盖率与新鲜度阈值（返回码非0代表未通过）：

```bash
python scripts/monitoring/check_industry_signal_quality.py
python scripts/monitoring/check_industry_signal_quality.py \
  --northbound-min-rows 20 \
  --northbound-max-stale-days 7 \
  --concept-min-coverage 0.95
```

输出：`data/signals/industry/industry_signal_quality_report.json`
