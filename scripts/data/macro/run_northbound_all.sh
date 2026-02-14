#!/bin/bash
# 依次获取北向资金数据（避免IP限制）

echo "========================================="
echo "北向资金数据获取（依次运行）"
echo "========================================="

# 统一入口顺序执行（避免并发触发IP限制）
echo ""
echo "[1/3] 获取北向资金流向数据..."
python3 scripts/data/tushare_downloader.py --task northbound_flow --start-date 20200101 --end-date $(date +%Y%m%d) --resume
echo "✓ 资金流向数据获取完成"
sleep 60

echo ""
echo "[2/3] 获取北向资金持仓数据..."
python3 scripts/data/tushare_downloader.py --task northbound_hold --start-date 20200101 --end-date $(date +%Y%m%d) --resume
echo "✓ 持仓数据获取完成"
sleep 60

echo ""
echo "[3/3] 获取北向资金持仓TOP10数据..."
python3 scripts/data/tushare_downloader.py --task northbound_top10 --start-date 20200101 --end-date $(date +%Y%m%d) --resume
echo "✓ TOP10数据获取完成"

echo ""
echo "========================================="
echo "所有北向资金数据获取完成！"
echo "========================================="
