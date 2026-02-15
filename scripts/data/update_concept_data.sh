#!/bin/bash
# 概念板块数据更新脚本
# 使用方法：
#   ./update_concept_data.sh init    # 初始化（获取同花顺概念列表与成分）
#   ./update_concept_data.sh update  # 周度更新（获取同花顺概念列表与成分）
#   ./update_concept_data.sh calc    # 仅刷新同花顺概念列表与成分（预留评分计算）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# 激活虚拟环境
source venv/bin/activate

# 执行脚本
case "$1" in
    init)
        echo "=== 初始化：获取同花顺概念列表与成分 ==="
        python scripts/data/tushare_downloader.py --task ths_index
        python scripts/data/tushare_downloader.py --task ths_member
        ;;
    update)
        echo "=== 周度更新：获取同花顺概念列表与成分 ==="
        python scripts/data/tushare_downloader.py --task ths_index
        python scripts/data/tushare_downloader.py --task ths_member
        ;;
    calc)
        echo "=== 计算概念表现（当前未实现，先更新同花顺概念成分） ==="
        python scripts/data/tushare_downloader.py --task ths_index
        python scripts/data/tushare_downloader.py --task ths_member
        ;;
    *)
        echo "使用方法："
        echo "  $0 init    # 初始化（获取基准数据）"
        echo "  $0 update  # 周度更新"
        echo "  $0 calc    # 只计算表现"
        exit 1
        ;;
esac
