#!/bin/bash
# 检查宏观数据更新情况
# 使用方法: ./scripts/data/macro/check_macro_data.sh

echo "=== 宏观数据更新检查 ==="
echo "检查时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 数据目录
DATA_ROOT="${SAGE_DATA_ROOT_PRIMARY:-data}"
DATA_DIR="${DATA_ROOT}/raw/tushare/macro"
LOG_DIR="logs/macro"

# 检查最新数据文件
echo "=== 最新数据文件 ==="
echo ""
echo "NBS数据:"
ls -lt $DATA_DIR/nbs_*.csv 2>/dev/null | head -3 | awk '{print $9, $6, $7, $8}'
echo ""
echo "Tushare数据:"
ls -lt $DATA_DIR/tushare_*.parquet 2>/dev/null | head -5 | awk '{print $9, $6, $7, $8}'
ls -lt $DATA_DIR/yield_*.parquet 2>/dev/null | head -2 | awk '{print $9, $6, $7, $8}'
echo ""

# 检查数据完整性
echo "=== 数据完整性检查 ==="
echo ""

# 检查本月数据
CURRENT_MONTH=$(date +%Y%m)
echo "检查 ${CURRENT_MONTH} 月份数据:"
echo ""

# NBS数据
NBS_FILES=$(ls $DATA_DIR/nbs_*_${CURRENT_MONTH}*.csv 2>/dev/null | wc -l)
echo "  NBS数据文件: ${NBS_FILES}个"
if [ $NBS_FILES -eq 3 ]; then
    echo "  ✓ NBS数据完整"
else
    echo "  ✗ NBS数据不完整（应为3个）"
fi

# Tushare数据
TUSHARE_FILES=$(ls $DATA_DIR/tushare_*.parquet 2>/dev/null | wc -l)
YIELD_FILES=$(ls $DATA_DIR/yield_*.parquet 2>/dev/null | wc -l)
echo "  Tushare数据文件: ${TUSHARE_FILES}个"
echo "  国债收益率文件: ${YIELD_FILES}个"
if [ $TUSHARE_FILES -ge 3 ] && [ $YIELD_FILES -ge 2 ]; then
    echo "  ✓ Tushare数据完整"
else
    echo "  ✗ Tushare数据不完整（应至少3个+2个收益率）"
fi

echo ""
echo "=== 检查完成 ==="
