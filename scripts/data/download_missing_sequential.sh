#!/bin/bash
# 顺序下载缺失数据（一次只运行一个任务）
# 创建时间: 2026-02-19

LOG_DIR="logs/data"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/sequential_download_${TIMESTAMP}.log"

echo "开始顺序下载缺失数据: $(date)" | tee -a "$LOG_FILE"
echo "共 6 个任务，预计耗时: 8-12小时" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 1. sw_valuation - 申万估值（2016-2019，约48个月）
echo "==========================================" | tee -a "$LOG_FILE"
echo "[1/6] sw_valuation - 申万估值 2016-2019" | tee -a "$LOG_FILE"
echo "开始: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
python -m scripts.data.tushare_downloader \
    --task sw_valuation \
    --start-date 20160101 \
    --end-date 20191231 \
    --sleep-seconds 40 2>&1 | tee -a "$LOG_FILE"
echo "✅ sw_valuation 完成: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 2. sw_industry_daily - 申万行业日线（2016-2026，约120个月）
echo "==========================================" | tee -a "$LOG_FILE"
echo "[2/6] sw_industry_daily - 申万行业日线" | tee -a "$LOG_FILE"
echo "开始: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
python -m scripts.data.tushare_downloader \
    --task sw_industry_daily \
    --start-date 20160101 \
    --end-date 20261231 \
    --sleep-seconds 40 2>&1 | tee -a "$LOG_FILE"
echo "✅ sw_industry_daily 完成: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 3. daily_kline - 补充2016年数据（12个月）
echo "==========================================" | tee -a "$LOG_FILE"
echo "[3/6] daily_kline - 补充2016年数据" | tee -a "$LOG_FILE"
echo "开始: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
python -m scripts.data.tushare_downloader \
    --task daily_kline \
    --start-date 20160101 \
    --end-date 20161231 \
    --sleep-seconds 40 2>&1 | tee -a "$LOG_FILE"
echo "✅ daily_kline 完成: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 4. northbound_hold - 补充2020-2025历史数据（72个月）
echo "==========================================" | tee -a "$LOG_FILE"
echo "[4/6] northbound_hold - 历史数据 2020-2025" | tee -a "$LOG_FILE"
echo "开始: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
python -m scripts.data.tushare_downloader \
    --task northbound_hold \
    --start-date 20200101 \
    --end-date 20251231 \
    --sleep-seconds 40 2>&1 | tee -a "$LOG_FILE"
echo "✅ northbound_hold 完成: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 5. forecast - 业绩预告（2016-2026）
echo "==========================================" | tee -a "$LOG_FILE"
echo "[5/6] forecast - 业绩预告" | tee -a "$LOG_FILE"
echo "开始: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
python -m scripts.data.tushare_downloader \
    --task forecast \
    --start-date 20160101 \
    --end-date 20261231 \
    --sleep-seconds 40 2>&1 | tee -a "$LOG_FILE"
echo "✅ forecast 完成: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 6. express - 业绩快报（2016-2026）
echo "==========================================" | tee -a "$LOG_FILE"
echo "[6/6] express - 业绩快报" | tee -a "$LOG_FILE"
echo "开始: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
python -m scripts.data.tushare_downloader \
    --task express \
    --start-date 20160101 \
    --end-date 20261231 \
    --sleep-seconds 40 2>&1 | tee -a "$LOG_FILE"
echo "✅ express 完成: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "==========================================" | tee -a "$LOG_FILE"
echo "全部下载完成: $(date)" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
