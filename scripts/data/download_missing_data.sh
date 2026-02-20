#!/bin/bash
# 下载缺失的核心数据
# 创建时间: 2026-02-19

LOG_DIR="logs/data"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/download_missing_${TIMESTAMP}.log"

echo "开始下载缺失数据: $(date)" | tee -a "$LOG_FILE"
echo "共 5 个任务" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 1. sw_industry_daily - 申万行业日线（2016-2026）
echo "==========================================" | tee -a "$LOG_FILE"
echo "[1/5] sw_industry_daily - 申万行业日线" | tee -a "$LOG_FILE"
echo "参数: --start-date 20160101 --end-date 20261231" | tee -a "$LOG_FILE"
echo "开始: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
python -m scripts.data.tushare_downloader \
    --task sw_industry_daily \
    --start-date 20160101 \
    --end-date 20261231 \
    --sleep-seconds 40 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ sw_industry_daily 完成: $(date)" | tee -a "$LOG_FILE"
else
    echo "⚠️ sw_industry_daily 失败" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# 2. forecast - 业绩预告（2016-2026）
echo "==========================================" | tee -a "$LOG_FILE"
echo "[2/5] forecast - 业绩预告" | tee -a "$LOG_FILE"
echo "参数: --start-date 20160101 --end-date 20261231" | tee -a "$LOG_FILE"
echo "开始: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
python -m scripts.data.tushare_downloader \
    --task forecast \
    --start-date 20160101 \
    --end-date 20261231 \
    --sleep-seconds 40 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ forecast 完成: $(date)" | tee -a "$LOG_FILE"
else
    echo "⚠️ forecast 失败" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# 3. express - 业绩快报（2016-2026）
echo "==========================================" | tee -a "$LOG_FILE"
echo "[3/5] express - 业绩快报" | tee -a "$LOG_FILE"
echo "参数: --start-date 20160101 --end-date 20261231" | tee -a "$LOG_FILE"
echo "开始: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
python -m scripts.data.tushare_downloader \
    --task express \
    --start-date 20160101 \
    --end-date 20261231 \
    --sleep-seconds 40 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ express 完成: $(date)" | tee -a "$LOG_FILE"
else
    echo "⚠️ express 失败" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# 4. margin_detail - 融资融券明细（2016-2026）
echo "==========================================" | tee -a "$LOG_FILE"
echo "[4/5] margin_detail - 融资融券明细" | tee -a "$LOG_FILE"
echo "参数: --start-date 20160101 --end-date 20261231" | tee -a "$LOG_FILE"
echo "开始: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
python -m scripts.data.tushare_downloader \
    --task margin_detail \
    --start-date 20160101 \
    --end-date 20261231 \
    --sleep-seconds 40 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ margin_detail 完成: $(date)" | tee -a "$LOG_FILE"
else
    echo "⚠️ margin_detail 失败" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# 5. daily_kline - 补充2016年初数据
echo "==========================================" | tee -a "$LOG_FILE"
echo "[5/5] daily_kline - 补充2016年初数据" | tee -a "$LOG_FILE"
echo "参数: --start-date 20160101 --end-date 20161231" | tee -a "$LOG_FILE"
echo "开始: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
python -m scripts.data.tushare_downloader \
    --task daily_kline \
    --start-date 20160101 \
    --end-date 20161231 \
    --sleep-seconds 40 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ daily_kline 完成: $(date)" | tee -a "$LOG_FILE"
else
    echo "⚠️ daily_kline 失败" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

echo "==========================================" | tee -a "$LOG_FILE"
echo "全部下载完成: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
