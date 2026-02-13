#!/bin/bash
# 获取所有宏观数据（NBS + Tushare）
# 使用方法: ./scripts/macro/fetch_all_macro_data.sh

echo "=== 开始获取宏观数据 ==="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 创建数据目录
mkdir -p data/tushare/macro

# 获取NBS数据（上个月）
echo "步骤1: 获取国家统计局数据..."
python3 scripts/macro/fetch_nbs_data.py

# 获取Tushare数据（最近3个月）
echo ""
echo "步骤2: 获取Tushare宏观数据..."
python3 scripts/macro/fetch_tushare_macro.py

# 获取北上资金数据
echo ""
echo "步骤3: 获取北上资金数据..."
python3 scripts/macro/fetch_northbound.py

# 映射NBS数据到申万行业（一级+二级）
echo ""
echo "步骤4: 映射NBS数据到申万行业（一级+二级）..."
python3 scripts/macro/map_nbs_to_sw_all.py

echo ""
echo "=== 数据获取完成 ==="
echo "数据目录: data/tushare/macro"
echo ""
echo "数据文件:"
ls -lh data/tushare/macro/*.csv data/tushare/macro/*.parquet 2>/dev/null | awk '{print $9, "("$5") bytes'}