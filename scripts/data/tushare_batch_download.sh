#!/bin/bash
# 串行下载 Tushare 数据任务
# 用法:
#   bash download_missing_data.sh                          # 下载全部任务
#   bash download_missing_data.sh ths_daily ths_member     # 只下载指定任务
#   bash download_missing_data.sh --start 20200101 --end 20241231 daily_kline daily_basic
#   bash download_missing_data.sh --list                   # 列出所有可用任务

set -o pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="logs/data"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SLEEP=40
DATE_START="20160101"
DATE_END="20261231"

# 所有任务定义：name|描述|额外参数（$DS/$DE 会被替换为实际日期）
# 顺序考虑依赖关系
ALL_TASKS=(
    # --- 基础元数据（single模式，无日期参数）---
    "sw_industry_classify|申万行业分类|"
    "ths_index|同花顺概念指数列表|"
    # --- 依赖元数据的 list 任务 ---
    "sw_index_member|申万行业成分股|"
    "ths_member|同花顺概念成分股|"
    "sw_industry_daily|申万行业日线|--start-date \$DS --end-date \$DE"
    # --- 日线行情 ---
    "daily_kline|日线K线|--start-date \$DS --end-date \$DE"
    "daily_basic|日线基本面|--start-date \$DS --end-date \$DE"
    "index_ohlc|指数OHLC|--start-date \$DS --end-date \$DE"
    # --- 财务数据（year_quarters模式）---
    "fina_indicator_vip|财务指标VIP|"
    "fina_mainbz_vip|主营业务VIP|"
    "income|利润表|"
    "balancesheet|资产负债表|"
    "cashflow|现金流量表|"
    "dividend|分红送股|--start-date \$DS --end-date \$DE"
    # --- 资金流向 ---
    "hs300_constituents|沪深300成分股|--start-date \$DS --end-date \$DE"
    "hs300_moneyflow_bydate|沪深300资金流向|--start-date \$DS --end-date 20200101"
    "margin|融资融券|--start-date \$DS --end-date \$DE"
    # --- 北向资金 ---
    "northbound_flow|北向资金流向|--start-date \$DS --end-date \$DE"
    "northbound_hold|北向持股|--start-date \$DS --end-date \$DE"
    "northbound_top10|北向Top10|--start-date \$DS --end-date \$DE"
    # --- 同花顺概念日线 ---
    "ths_daily|同花顺概念日线|--start-date \$DS --end-date \$DE"
    # --- 宏观数据 ---
    "yield_10y|10年国债收益率|"
    "yield_2y|2年国债收益率|"
    "cn_cpi|CPI|--start-date \$DS --end-date \$DE"
    "cn_ppi|PPI|--start-date \$DS --end-date \$DE"
    "cn_pmi|PMI|--start-date \$DS --end-date \$DE"
    "sw_valuation|申万估值|--start-date \$DS --end-date \$DE"
    # --- 公告/研报 ---
    "tushare_anns|公告|--start-date \$DS --end-date \$DE"
    "tushare_reports|研报|--start-date \$DS --end-date \$DE"
    # --- 期权 ---
    "opt_daily|期权日线|--start-date \$DS --end-date \$DE"
    # --- 其他 ---
    "dc_index|东财指数|--start-date \$DS --end-date \$DE"
)

# --- 解析参数 ---
SELECTED_TASKS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start)
            DATE_START="$2"; shift 2 ;;
        --end)
            DATE_END="$2"; shift 2 ;;
        --sleep)
            SLEEP="$2"; shift 2 ;;
        --list)
            echo "可用任务列表："
            for entry in "${ALL_TASKS[@]}"; do
                IFS='|' read -r name desc _ <<< "$entry"
                printf "  %-30s %s\n" "$name" "$desc"
            done
            exit 0 ;;
        --help|-h)
            echo "用法:"
            echo "  bash $0                                    # 下载全部任务"
            echo "  bash $0 task1 task2 ...                    # 只下载指定任务"
            echo "  bash $0 --start 20200101 --end 20241231    # 指定日期范围"
            echo "  bash $0 --list                             # 列出所有可用任务"
            echo "  bash $0 --sleep 60                         # 设置请求间隔(秒)"
            exit 0 ;;
        -*)
            echo "未知参数: $1"; exit 1 ;;
        *)
            SELECTED_TASKS+=("$1"); shift ;;
    esac
done

# --- 构建运行任务列表 ---
RUN_TASKS=()
if [[ ${#SELECTED_TASKS[@]} -eq 0 ]]; then
    # 没有指定任务，运行全部
    RUN_TASKS=("${ALL_TASKS[@]}")
else
    # 按指定顺序筛选任务
    for sel in "${SELECTED_TASKS[@]}"; do
        FOUND=0
        for entry in "${ALL_TASKS[@]}"; do
            IFS='|' read -r name _ _ <<< "$entry"
            if [[ "$name" == "$sel" ]]; then
                RUN_TASKS+=("$entry")
                FOUND=1
                break
            fi
        done
        if [[ $FOUND -eq 0 ]]; then
            echo "⚠️ 未知任务: $sel（用 --list 查看可用任务）"
        fi
    done
fi

TOTAL=${#RUN_TASKS[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "没有可运行的任务"
    exit 1
fi

echo "=========================================="
echo "开始串行下载 Tushare 数据（共 ${TOTAL} 个任务）"
echo "日期范围: $DATE_START ~ $DATE_END"
echo "请求间隔: ${SLEEP}s"
echo "时间: $(date)"
echo "=========================================="

PASS=0
FAIL=0
for i in "${!RUN_TASKS[@]}"; do
    IFS='|' read -r TASK_NAME TASK_DESC TASK_ARGS_TPL <<< "${RUN_TASKS[$i]}"
    NUM=$((i + 1))

    # 替换日期占位符
    DS="$DATE_START" DE="$DATE_END"
    TASK_ARGS=$(eval echo "$TASK_ARGS_TPL")

    echo ""
    echo "[Task ${NUM}/${TOTAL}] ${TASK_NAME} (${TASK_DESC})"
    echo "开始时间: $(date)"

    # shellcheck disable=SC2086
    python -u -m scripts.data.tushare_downloader \
        --task "$TASK_NAME" \
        --resume \
        --sleep-seconds "$SLEEP" \
        $TASK_ARGS \
        2>&1 | tee "$LOG_DIR/${TIMESTAMP}_${TASK_NAME}.log" || {
        echo "⚠️ [Task ${NUM}/${TOTAL}] ${TASK_NAME} 失败，继续下一个任务"
        FAIL=$((FAIL + 1))
        continue
    }

    echo "✅ [Task ${NUM}/${TOTAL}] ${TASK_NAME} 完成: $(date)"
    PASS=$((PASS + 1))
done

echo ""
echo "=========================================="
echo "全部完成: $(date)"
echo "成功: ${PASS}  失败: ${FAIL}  总计: ${TOTAL}"
echo "=========================================="
