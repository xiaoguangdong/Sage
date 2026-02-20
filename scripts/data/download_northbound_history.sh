#!/bin/bash
# 补充北向持股历史数据（2020-2025年缺口）
# 创建时间: 2026-02-19

LOG_DIR="logs/data"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/northbound_history_${TIMESTAMP}.log"

echo "开始补充北向持股历史数据: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# northbound_hold - 补充2020-2025年数据
echo "==========================================" | tee -a "$LOG_FILE"
echo "northbound_hold - 北向持股历史数据" | tee -a "$LOG_FILE"
echo "参数: --start-date 20200101 --end-date 20251231" | tee -a "$LOG_FILE"
echo "开始: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
python -m scripts.data.tushare_downloader \
    --task northbound_hold \
    --start-date 20200101 \
    --end-date 20251231 \
    --sleep-seconds 40 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ northbound_hold 完成: $(date)" | tee -a "$LOG_FILE"
else
    echo "⚠️ northbound_hold 失败" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

echo "==========================================" | tee -a "$LOG_FILE"
echo "北向持股历史数据补充完成: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
