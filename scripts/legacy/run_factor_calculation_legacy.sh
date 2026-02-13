#!/bin/bash
# 后台运行因子计算任务

echo "开始后台计算股票因子..."
echo "日志文件: data/tushare/factors/calculate_factors.log"
echo ""

# 创建日志目录
mkdir -p data/tushare/factors

# 后台运行并保存日志
nohup venv/bin/python scripts/models/calculate_stock_factors.py \
    > data/tushare/factors/calculate_factors.log 2>&1 &

echo "任务已在后台运行，进程ID: $!"
echo "查看日志: tail -f data/tushare/factors/calculate_factors.log"
echo ""
echo "检查进程状态: ps aux | grep calculate_stock_factors"
echo ""
echo "停止任务: kill $!"
