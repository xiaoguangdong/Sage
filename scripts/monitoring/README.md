# Monitoring scripts

监控、告警、日报/周报生成脚本放在这里。

## 日志归档

根目录散落日志可用下述命令整理到 `logs/<module>/`：

```bash
python scripts/monitoring/organize_logs.py --dry-run
python scripts/monitoring/organize_logs.py
```
