#!/bin/bash
# 监控数据下载进度
# 创建时间: 2026-02-19

echo "=========================================="
echo "数据下载进度监控"
echo "时间: $(date)"
echo "=========================================="
echo ""

# 检查 sw_valuation
echo "📊 [1] sw_valuation (2016-2019)"
if [ -f "data/states/sw_valuation.json" ]; then
    echo "   状态文件: $(cat data/states/sw_valuation.json)"
fi
if [ -f "logs/data/sw_valuation_retry.log" ]; then
    LAST_LINE=$(tail -1 logs/data/sw_valuation_retry.log)
    echo "   最新: $LAST_LINE"
fi
echo ""

# 检查缺失数据下载
echo "📊 [2] 缺失数据下载 (5个任务)"
LATEST_LOG=$(ls -t logs/data/download_missing_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "   日志: $LATEST_LOG"
    echo "   最新进度:"
    tail -5 "$LATEST_LOG" | sed 's/^/   /'
fi
echo ""

# 检查北向持股历史数据
echo "📊 [3] northbound_hold 历史数据 (2020-2025)"
LATEST_LOG=$(ls -t logs/data/northbound_history_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "   日志: $LATEST_LOG"
    echo "   最新进度:"
    tail -5 "$LATEST_LOG" | sed 's/^/   /'
fi
echo ""

# 统计已完成的数据文件
echo "=========================================="
echo "数据文件统计"
echo "=========================================="
echo "核心数据文件:"
echo "  - daily_kline: $([ -f data/tushare/daily.parquet ] && echo '✅' || echo '❌')"
echo "  - daily_basic: $([ -f data/tushare/daily_basic.parquet ] && echo '✅' || echo '❌')"
echo "  - sw_industry_daily: $([ -f data/tushare/sw_industry/sw_industry_daily.parquet ] && echo '✅' || echo '❌')"
echo "  - sw_valuation: $([ -f data/tushare/macro/sw_valuation.parquet ] && echo '✅' || echo '❌')"
echo "  - forecast: $([ -f data/tushare/fundamental/forecast.parquet ] && echo '✅' || echo '❌')"
echo "  - express: $([ -f data/tushare/fundamental/express.parquet ] && echo '✅' || echo '❌')"
echo "  - margin_detail: $([ -f data/tushare/margin_detail.parquet ] && echo '✅' || echo '❌')"
echo "  - northbound_hold: $([ -f data/tushare/northbound/northbound_hold.parquet ] && echo '✅' || echo '❌')"
echo ""

echo "=========================================="
echo "监控完成: $(date)"
echo "=========================================="
