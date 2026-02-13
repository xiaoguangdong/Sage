#!/bin/bash
# 后台运行因子计算任务
# 使用方法: ./scripts/run_factor_calculation.sh

echo "开始后台计算股票因子..."
echo "日志文件: data/processed/factors/calculate_factors.log"
echo ""

# 创建日志目录
mkdir -p data/processed/factors

# 后台运行并保存日志
nohup venv/bin/python scripts/models/calculate_stock_factors.py \
    > data/processed/factors/calculate_factors.log 2>&1 &

echo "任务已在后台运行，进程ID: $!"
echo "查看日志: tail -f data/processed/factors/calculate_factors.log"
echo ""
echo "检查进程状态: ps aux | grep calculate_stock_factors"
echo ""
echo "停止任务: kill $!"
