#!/bin/bash
# 启动TensorBoard
# 使用方法: ./scripts/start_tensorboard.sh

echo "启动TensorBoard..."
echo "日志目录: logs/tensorboard"
echo "访问地址: http://localhost:6006"
echo ""

# 检查是否安装了tensorboard
if ! command -v tensorboard &> /dev/null; then
    echo "错误: tensorboard未安装"
    echo "请运行: pip install tensorboard"
    exit 1
fi

# 启动TensorBoard
tensorboard --logdir=logs/tensorboard --port=6006