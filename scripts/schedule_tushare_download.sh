#!/bin/bash
# Tushare批量下载定时任务脚本
# 每天在指定时间运行下载任务

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/data"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 运行下载脚本
cd "$PROJECT_DIR"
nohup venv/bin/python scripts/data/tushare_suite.py --action daily_basic --resume > "$LOG_DIR/tushare_daily_basic.log" 2>&1 &
nohup venv/bin/python scripts/data/tushare_suite.py --action margin --resume > "$LOG_DIR/tushare_margin.log" 2>&1 &

echo "Tushare下载任务已启动"
echo "日志文件: $LOG_DIR/tushare_daily_basic.log, $LOG_DIR/tushare_margin.log"
