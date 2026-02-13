#!/bin/bash
# 依次获取北向资金数据（避免IP限制）

echo "========================================="
echo "北向资金数据获取（依次运行）"
echo "========================================="

# 1. 获取资金流向
echo ""
echo "[1/3] 获取北向资金流向数据..."
nohup python3 scripts/macro/fetch_northbound_history.py flow > logs/fetch_northbound_flow.log 2>&1 &
FLOW_PID=$!
echo "  PID: $FLOW_PID"
echo "  日志: logs/fetch_northbound_flow.log"
tail -f logs/fetch_northbound_flow.log &
TAIL_PID=$!

# 等待资金流向完成
wait $FLOW_PID
kill $TAIL_PID 2>/dev/null

echo ""
echo "✓ 资金流向数据获取完成"
sleep 60  # 等待60秒，避免IP限制

# 2. 获取持仓数据
echo ""
echo "[2/3] 获取北向资金持仓数据..."
nohup python3 scripts/macro/fetch_northbound_history.py hold > logs/fetch_northbound_hold.log 2>&1 &
HOLD_PID=$!
echo "  PID: $HOLD_PID"
echo "  日志: logs/fetch_northbound_hold.log"
tail -f logs/fetch_northbound_hold.log &
TAIL_PID=$!

wait $HOLD_PID
kill $TAIL_PID 2>/dev/null

echo ""
echo "✓ 持仓数据获取完成"
sleep 60

# 3. 获取TOP10数据
echo ""
echo "[3/3] 获取北向资金持仓TOP10数据..."
nohup python3 scripts/macro/fetch_northbound_history.py top10 > logs/fetch_northbound_top10.log 2>&1 &
TOP10_PID=$!
echo "  PID: $TOP10_PID"
echo "  日志: logs/fetch_northbound_top10.log"
tail -f logs/fetch_northbound_top10.log &
TAIL_PID=$!

wait $TOP10_PID
kill $TAIL_PID 2>/dev/null

echo ""
echo "✓ TOP10数据获取完成"
sleep 60

echo ""
echo "========================================="
echo "所有北向资金数据获取完成！"
echo "========================================="