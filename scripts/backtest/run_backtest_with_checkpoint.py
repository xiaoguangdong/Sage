#!/usr/bin/env python3
"""
直接使用checkpoint中的模型进行回测
"""

import os

import pandas as pd
import torch
import yaml
from utils.logger import get_logger

logger = get_logger(__name__)


def load_config():
    with open("./config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_model_and_config():
    """直接从checkpoint加载模型"""
    logger.info("加载best_model checkpoint...")

    checkpoint = torch.load("./checkpoints/best_model.pth", map_location="cpu")

    # 从checkpoint推断模型参数
    state_dict = checkpoint["model_state_dict"]

    # 推断hidden_dim (双向LSTM，hidden_dim = lstm.weight_ih_l0.shape[0] / 4)
    lstm_weight_ih = state_dict["lstm.weight_ih_l0"]
    hidden_dim = lstm_weight_ih.shape[0] // 8  # 双向，每个方向4个门
    input_dim = lstm_weight_ih.shape[1]

    logger.info("✓ 加载checkpoint成功")
    logger.info(f"  - 输入维度: {input_dim}")
    logger.info(f"  - 隐藏层维度: {hidden_dim}")
    logger.info(f"  - 验证准确率: {checkpoint['val_acc']:.4f}")
    logger.info(f"  - 训练轮数: {checkpoint['epoch']}")

    return checkpoint, input_dim, hidden_dim


def prepare_test_data():
    """准备测试数据"""
    logger.info("加载测试数据...")

    data_path = "./data/processed/A_share_day_final_2year.parquet"
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        return None

    data = pd.read_parquet(data_path)

    # 使用最后15%的数据作为测试集
    test_stocks = {}
    for stock_code, df in data.groupby("code"):
        n_samples = len(df)
        test_start = int(n_samples * 0.85)
        test_df = df.iloc[test_start:].copy()
        if len(test_df) > 20:
            test_stocks[stock_code] = test_df

    logger.info(f"✓ 加载了 {len(test_stocks)} 只股票的测试数据")
    return test_stocks


def extract_features_simple(stock_data, input_dim=30):
    """简化特征提取，使用原始数据的基本指标"""
    stock_features = {}

    for stock_code, df in stock_data.items():
        # 计算基本特征
        df["return_1d"] = df["close"].pct_change(1)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_10d"] = df["close"].pct_change(10)

        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        df["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]
        df["close_open_ratio"] = (df["close"] - df["open"]) / df["open"]

        # MA
        df["MA5"] = df["close"].rolling(5).mean()
        df["MA10"] = df["close"].rolling(10).mean()
        df["MA20"] = df["close"].rolling(20).mean()
        df["MA_ratio_5"] = df["close"] / df["MA5"]
        df["MA_ratio_10"] = df["close"] / df["MA10"]
        df["MA_ratio_20"] = df["close"] / df["MA20"]

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # 波动率
        df["volatility_5d"] = df["return_1d"].rolling(5).std()
        df["volatility_10d"] = df["return_1d"].rolling(10).std()

        # 添加label
        df["future_return"] = df["close"].pct_change(5).shift(-5)
        df["label"] = 1  # 默认震荡
        df.loc[df["future_return"] >= 0.03, "label"] = 2  # 涨
        df.loc[df["future_return"] <= -0.03, "label"] = 0  # 跌

        # 选择前30个特征列
        feature_cols = [
            col
            for col in df.columns
            if col not in ["date", "code", "open", "high", "low", "close", "volume", "amount", "future_return"]
        ][:input_dim]

        df_features = df[["date", "code"] + feature_cols].copy()
        df_features = df_features.dropna()

        if len(df_features) > 0:
            stock_features[stock_code] = df_features

    return stock_features


def run_backtest(checkpoint, stock_features, input_dim, hidden_dim):
    """运行回测"""
    logger.info("=" * 70)
    logger.info("开始回测")
    logger.info("=" * 70)

    seq_length = 20

    for stock_code, df in stock_features.items():
        if len(df) < seq_length:
            continue

        # 准备数据
        feature_cols = [col for col in df.columns if col not in ["date", "code", "label"]][:input_dim]

        X = df[feature_cols].values
        df["label"].values

        # 滑动窗口预测
        for i in range(seq_length, len(X)):
            X_seq = X[i - seq_length : i].reshape(1, seq_length, -1)
            torch.FloatTensor(X_seq)

            with torch.no_grad():
                checkpoint["model_state_dict"]
                # 这里简化处理，实际需要构建完整模型
                # 暂时跳过，直接统计准确率
                pass

    # 简化版本：只做数据统计
    total_samples = sum(len(df) for df in stock_features.values())
    logger.info(f"测试样本总数: {total_samples}")
    logger.info(f"测试股票数: {len(stock_features)}")

    # 实际完整回测需要构建模型，这里先跳过
    logger.info("\n⚠️  完整回测需要构建模型架构")
    logger.info("建议：重新训练模型或使用main.py的完整流程")

    return 0.0


def main():
    print("=" * 70)
    print("Best Model Checkpoint 分析")
    print("=" * 70)

    # 1. 加载checkpoint
    checkpoint, input_dim, hidden_dim = load_model_and_config()

    print("\n模型配置:")
    print(f"  - 输入维度: {input_dim}")
    print(f"  - 隐藏层维度: {hidden_dim}")

    # 2. 准备测试数据
    test_data = prepare_test_data()
    if test_data is None:
        print("无法加载测试数据")
        return

    # 3. 提取特征
    print(f"\n提取{input_dim}个特征...")
    stock_features = extract_features_simple(test_data, input_dim)
    print(f"✓ 特征提取完成，共 {len(stock_features)} 只股票")

    print("\n建议:")
    print("  1. 使用 main.py 重新训练（推荐）")
    print("  2. 修改代码以匹配checkpoint的完整模型架构")


if __name__ == "__main__":
    main()
