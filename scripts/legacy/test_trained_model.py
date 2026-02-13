#!/usr/bin/env python3
"""
测试已训练的模型
"""

import torch
import numpy as np
import pandas as pd
from models.lstm_attention import LSTMAttention
import yaml
from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_config():
    with open('./config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path, input_dim, config):
    model = LSTMAttention(
        input_dim=input_dim,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def main():
    config = load_config()
    
    print("="*70)
    print("测试已训练模型")
    print("="*70)
    
    # 检查可用的checkpoint
    import os
    checkpoints = {
        'simple_model.pth': './checkpoints/simple_model.pth',
        'best_model.pth': './checkpoints/best_model.pth'
    }
    
    available = {k: v for k, v in checkpoints.items() if os.path.exists(v)}
    
    if not available:
        print("没有找到可用的模型文件")
        return
    
    print(f"\n找到 {len(available)} 个模型文件:")
    for name, path in available.items():
        print(f"  - {name}: {path}")
    
    # 使用best_model
    checkpoint_path = available.get('best_model.pth', list(available.values())[0])
    print(f"\n使用模型: {checkpoint_path}")
    
    # 加载模型
    input_dim = 35  # Top 35特征
    model = load_model(checkpoint_path, input_dim, config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'val_acc' in checkpoint:
        print(f"验证准确率: {checkpoint['val_acc']:.4f}")
    if 'epoch' in checkpoint:
        print(f"训练轮数: {checkpoint['epoch']}")
    
    print("\n模型加载成功!")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试模型推理
    print("\n测试模型推理...")
    batch_size = 4
    seq_len = 20
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出概率:\n{output}")
    
    print("\n✓ 模型可以正常推理，可以继续后续流程")

if __name__ == '__main__':
    main()