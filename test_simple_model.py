#!/usr/bin/env python3
"""
测试simple_model
"""

import torch
import torch.nn as nn
import yaml

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

def main():
    print("="*70)
    print("测试simple_model")
    print("="*70)

    # 加载checkpoint
    checkpoint = torch.load('./checkpoints/simple_model.pth', map_location='cpu')

    # 从参数推断维度
    lstm_weight_ih = checkpoint['lstm.weight_ih_l0']
    hidden_dim = lstm_weight_ih.shape[0] // 4  # LSTM有4个门
    input_dim = lstm_weight_ih.shape[1]
    output_dim = checkpoint['fc.weight'].shape[0]

    print(f"\n推断的模型参数:")
    print(f"  - 输入维度: {input_dim}")
    print(f"  - 隐藏层维度: {hidden_dim}")
    print(f"  - 输出维度: {output_dim}")

    # 创建模型
    model = SimpleLSTM(input_dim, hidden_dim, output_dim)
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试推理
    print("\n测试模型推理...")
    batch_size = 4
    seq_len = 20
    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出概率:\n{output}")
    print(f"预测类别: {output.argmax(dim=1)}")

    print("\n✓ 模型可以正常推理！")

if __name__ == '__main__':
    main()