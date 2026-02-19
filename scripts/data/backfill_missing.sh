#!/bin/bash
# 精确补数据脚本：每个任务指定缺失的时间段
# 基于 2026-02-19 data_integrity_checker 检查结果

set -o pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="logs/data"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SLEEP=40

run_task() {
    local task="$1"
    local desc="$2"
    shift 2
    local extra_args="$@"

    echo ""
    echo "=========================================="
    echo "[${task}] ${desc}"
    echo "参数: ${extra_args}"
    echo "开始: $(date)"
    echo "=========================================="

    # shellcheck disable=SC2086
    python -u -m scripts.data.tushare_downloader \
        --task "$task" \
        --resume \
        --sleep-seconds "$SLEEP" \
        $extra_args \
        2>&1 | tee "$LOG_DIR/${TIMESTAMP}_${task}.log" || {
        echo "⚠️ ${task} 失败"
        return 1
    }
    echo "✅ ${task} 完成: $(date)"
}

echo "开始精确补数据: $(date)"
echo "共 9 个任务"

# 1. daily_kline: 有2016(12月)和2020-2026，缺2017-2019
run_task daily_kline "日线K线 补2017-2019" --start-date 20170101 --end-date 20191231

# 2. daily_basic: _all文件有2020-2026，缺2016-2019
run_task daily_basic "日线基本面 补2016-2019" --start-date 20160101 --end-date 20191231

# 3. ths_daily: 只有910K小文件(2016-2019)，缺2020-2026
run_task ths_daily "同花顺概念日线 补2020-2026" --start-date 20200101 --end-date 20261231

# 4. sw_valuation: 有2020-2026，缺2016-2019
run_task sw_valuation "申万估值 补2016-2019" --start-date 20160101 --end-date 20191231

# 5. dividend: 只有61条，几乎全缺
run_task dividend "分红送股 全量补" --start-date 20160101 --end-date 20261231

# 6. margin: 有2016-2019，缺2020-2026
run_task margin "融资融券 补2020-2026" --start-date 20200101 --end-date 20261231

# 7. northbound_flow: 有2016-2019，缺2020-2026
run_task northbound_flow "北向资金流向 补2020-2026" --start-date 20200101 --end-date 20261231

# 8. northbound_hold: 有2016和2026，中间2020-2025空洞
run_task northbound_hold "北向持股 补2017-2025" --start-date 20170101 --end-date 20251231

# 9. northbound_top10: 有2016-2019，缺2020-2026
run_task northbound_top10 "北向Top10 补2020-2026" --start-date 20200101 --end-date 20261231

echo ""
echo "=========================================="
echo "全部补数据完成: $(date)"
echo "=========================================="
