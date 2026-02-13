"""
测试买卖点模型（简化版，不需要lightgbm）
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 导入被测试的模块
sys.path.append(str(Path(__file__).resolve().parents[2]))

from sage_core.models.entry_model import EntryModelLR


class TestEntryModel(unittest.TestCase):
    """测试买卖点模型"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
        
        # 模拟股票数据
        close = 10 + np.cumsum(np.random.randn(len(dates)) * 0.1)
        turnover = np.random.uniform(0.01, 0.1, len(dates))
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'code': 'sh.600000',
                'close': close[i],
                'turnover': turnover[i]
            })
        
        self.df = pd.DataFrame(data)
        self.config = {
            'rules': {
                'prob_threshold': 0.6,
                'ma_period': 20,
                'vol_threshold': 0.05
            },
            'label': {
                'horizon': 10,
                'return_threshold': 0.05,
                'max_drawdown': -0.03
            }
        }
    
    def test_create_entry_model(self):
        """测试创建买卖点模型"""
        model = EntryModelLR(self.config)
        self.assertIsNotNone(model)
    
    def test_train_and_predict(self):
        """测试训练和预测"""
        model = EntryModelLR(self.config)
        
        # 创建标签
        labels = model.create_entry_label(self.df)
        df_clean = self.df.dropna()
        labels = labels.loc[df_clean.index]
        
        # 训练模型
        model.train(df_clean, labels)
        self.assertTrue(model.is_trained)
        
        # 预测
        df_predict = model.predict(df_clean.tail(30))
        self.assertIn('entry_prob', df_predict.columns)
        self.assertIn('entry_signal', df_predict.columns)
        self.assertIn('above_ma', df_predict.columns)
        self.assertIn('low_vol', df_predict.columns)
    
    def test_prepare_features(self):
        """测试特征准备"""
        model = EntryModelLR(self.config)
        df_features = model.prepare_features(self.df)
        
        # 检查关键特征是否存在
        self.assertIn('price_ma20_ratio', df_features.columns)
        self.assertIn('volume_ma20_ratio', df_features.columns)
        self.assertIn('macd', df_features.columns)
        self.assertIn('volatility_5d', df_features.columns)
    
    def test_create_entry_label(self):
        """测试标签创建"""
        model = EntryModelLR(self.config)
        labels = model.create_entry_label(self.df)
        
        # 检查标签值是否为0或1
        self.assertTrue(all(labels.isin([0, 1])))


if __name__ == "__main__":
    unittest.main()
