#!/bin/bash
# Tushare批量下载定时任务脚本
# 每天在指定时间运行下载任务

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/data/tushare"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 清空状态文件
rm -f "$LOG_DIR/download_state_optimized.json"

# 运行下载脚本
cd "$PROJECT_DIR"
nohup venv/bin/python scripts/data/batch_download_tushare_optimized.py > "$LOG_DIR/download_batch.log" 2>&1 &

echo "Tushare下载任务已启动，PID: $!"
echo "日志文件: $LOG_DIR/download_batch.log"
