#!/usr/bin/env python3
"""
使用best_model进行回测
"""

import torch
import pandas as pd
import numpy as np
from models.lstm_attention import BidirectionalLSTMAttention
from features.technical_indicators import TechnicalIndicators
from utils.logger import get_logger
import yaml
import os

logger = get_logger(__name__)

def load_config():
    with open('./config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_best_model():
    """加载best_model (30个特征)"""
    config = load_config()

    # 创建模型配置
    model_config = {
        'input_dim': 30,  # best_model使用30个特征
        'hidden_dim': 256,
        'num_layers': 2,
        'num_classes': 3,
        'dropout': 0.2
    }

    model = BidirectionalLSTMAttention(model_config)

    # 加载checkpoint
    checkpoint = torch.load('./checkpoints/best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"✓ 加载best_model成功")
    logger.info(f"  - 验证准确率: {checkpoint['val_acc']:.4f}")
    logger.info(f"  - 训练轮数: {checkpoint['epoch']}")

    return model, checkpoint['val_acc']

def prepare_test_data():
    """准备测试数据"""
    logger.info("加载测试数据...")

    # 加载数据
    data_path = './data/processed/A_share_day_final_2year.parquet'
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        return None

    data = pd.read_parquet(data_path)

    # 使用最后15%的数据作为测试集
    test_stocks = {}
    for stock_code, df in data.groupby('code'):
        n_samples = len(df)
        test_start = int(n_samples * 0.85)
        test_df = df.iloc[test_start:].copy()
        if len(test_df) > 20:  # 至少20天数据
            test_stocks[stock_code] = test_df

    logger.info(f"✓ 加载了 {len(test_stocks)} 只股票的测试数据")
    return test_stocks

def extract_features(stock_data, top_k=30):
    """提取Top 30特征"""
    feature_extractor = TechnicalIndicators(load_config()['features'])

    stock_features = {}
    for stock_code, df in stock_data.items():
        # 提取特征
        df_features = feature_extractor.calculate_all(df)

        # 选择Top 30特征（基于之前训练的特征重要性）
        # 这里简化处理，选择常见的30个技术指标
        feature_cols = [col for col in df_features.columns
                       if col not in ['date', 'code', 'open', 'high', 'low', 'close',
                                     'volume', 'amount', 'label', 'future_return']]

        # 取前30个特征
        selected_features = feature_cols[:30]
        df_features = df_features[['date', 'code'] + selected_features + ['label']].copy()

        # 删除包含NaN的行
        df_features = df_features.dropna()

        if len(df_features) > 0:
            stock_features[stock_code] = df_features

    return stock_features

def run_backtest(model, stock_features):
    """运行回测"""
    logger.info("="*70)
    logger.info("开始回测")
    logger.info("="*70)

    correct = 0
    total = 0
    predictions = []

    seq_length = 20

    for stock_code, df in stock_features.items():
        if len(df) < seq_length:
            continue

        # 准备数据
        feature_cols = [col for col in df.columns
                       if col not in ['date', 'code', 'label']][:30]

        X = df[feature_cols].values
        y = df['label'].values

        # 滑动窗口预测
        for i in range(seq_length, len(X)):
            X_seq = X[i-seq_length:i].reshape(1, seq_length, -1)
            X_tensor = torch.FloatTensor(X_seq)

            with torch.no_grad():
                output = model(X_tensor)
                pred = output.argmax(dim=1).item()

            correct += (pred == y[i])
            total += 1

            predictions.append({
                'stock': stock_code,
                'date': df.iloc[i]['date'],
                'prediction': pred,
                'actual': y[i]
            })

    accuracy = correct / total if total > 0 else 0

    logger.info("="*70)
    logger.info("回测完成")
    logger.info("="*70)
    logger.info(f"测试样本数: {total}")
    logger.info(f"预测正确数: {correct}")
    logger.info(f"测试准确率: {accuracy:.4f}")

    # 详细统计
    pred_df = pd.DataFrame(predictions)
    if len(pred_df) > 0:
        logger.info(f"\n各类别预测统计:")
        logger.info(pred_df['prediction'].value_counts().to_string())

        logger.info(f"\n混淆矩阵:")
        cm = pd.crosstab(pred_df['actual'], pred_df['prediction'],
                        rownames=['Actual'], colnames=['Predicted'])
        logger.info(cm.to_string())

    return accuracy

def main():
    print("="*70)
    print("Best Model 回测测试")
    print("="*70)

    # 1. 加载模型
    model, val_acc = load_best_model()
    print(f"\n模型验证准确率: {val_acc:.4f}\n")

    # 2. 准备测试数据
    test_data = prepare_test_data()
    if test_data is None:
        print("无法加载测试数据")
        return

    # 3. 提取特征
    print("提取特征...")
    stock_features = extract_features(test_data, top_k=30)
    print(f"✓ 特征提取完成，共 {len(stock_features)} 只股票\n")

    # 4. 运行回测
    test_acc = run_backtest(model, stock_features)

    print(f"\n对比:")
    print(f"  验证集准确率: {val_acc:.4f}")
    print(f"  测试集准确率: {test_acc:.4f}")

if __name__ == '__main__':
    main()