#!/bin/bash
# 概念成分股数据更新脚本
# 使用方法：
#   ./update_concept_data.sh init    # 初始化（获取基准数据）
#   ./update_concept_data.sh update  # 周度更新
#   ./update_concept_data.sh calc    # 只计算表现

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# 激活虚拟环境
source venv/bin/activate

# 执行脚本
case "$1" in
    init)
        echo "=== 初始化：获取基准概念成分股数据 ==="
        python scripts/data/tushare_suite.py --action concept_update_tushare --mode init
        ;;
    update)
        echo "=== 周度更新：获取最新数据并计算 ==="
        python scripts/data/tushare_suite.py --action concept_update_tushare --mode update \
            --start-date 20240924 \
            --end-date 20241231 \
            --min-stock-count 10
        ;;
    calc)
        echo "=== 计算概念表现 ==="
        python scripts/data/tushare_suite.py --action concept_update_tushare --mode calculate \
            --start-date 20240924 \
            --end-date 20241231 \
            --min-stock-count 10
        ;;
    *)
        echo "使用方法："
        echo "  $0 init    # 初始化（获取基准数据）"
        echo "  $0 update  # 周度更新"
        echo "  $0 calc    # 只计算表现"
        exit 1
        ;;
esac
