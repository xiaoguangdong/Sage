"""
测试数据加载器（简化版，使用unittest）
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# 导入被测试的模块
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import DataLoader
from data.universe import Universe


class TestDataLoader(unittest.TestCase):
    """测试数据加载器"""
    
    def test_load_baostock_data_file_not_exists(self):
        """测试加载不存在的文件"""
        loader = DataLoader('data')
        result = loader.load_baostock_data('sh.999999')
        self.assertIsNone(result)
    
    def test_check_data_quality(self):
        """测试数据质量检查"""
        # 创建测试数据
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', '2020-01-10'),
            'stock': ['sh.999999'] * 10,
            'close': np.random.uniform(10, 20, 10),
            'volume': np.random.randint(1000000, 10000000, 10),
            'amount': np.random.uniform(1e8, 2e8, 10)
        })
        
        loader = DataLoader('data')
        report = loader.check_data_quality(df)
        
        self.assertIsNotNone(report)
        self.assertIn('total_rows', report)
        self.assertEqual(report['total_rows'], 10)
        self.assertIn('date_range', report)


class TestUniverse(unittest.TestCase):
    """测试股票池筛选"""
    
    def test_filter_stocks_no_filter(self):
        """测试不进行任何筛选"""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', '2020-01-10'),
            'stock': ['sh.999999'] * 10,
            'close': np.random.uniform(10, 20, 10),
            'is_st': [False] * 10,
            'is_suspended': [False] * 10,
            'turnover': np.random.uniform(2, 5, 10),
            'market_cap': np.random.uniform(200, 500, 10) * 1e8
        })
        
        universe = Universe()
        result = universe.filter_stocks(df, exclude_st=False, exclude_suspended=False)
        
        self.assertEqual(len(result), 10)
    
    def test_filter_stocks_with_filters(self):
        """测试进行筛选"""
        np.random.seed(42)
        
        # 创建一些会被筛选的数据
        data = []
        for i in range(10):
            data.append({
                'date': pd.date_range('2020-01-01', '2020-01-10')[i],
                'stock': f'sh.99999{i}',
                'close': np.random.uniform(10, 20),
                'is_st': True if i < 2 else False,
                'is_suspended': True if 2 <= i < 4 else False,
                'turnover': 0.5 if i < 6 else 2.0,
                'market_cap': 50e8 if i < 8 else 200e8
            })
        
        df = pd.DataFrame(data)
        
        universe = Universe()
        result = universe.filter_stocks(
            df,
            exclude_st=True,
            exclude_suspended=True,
            min_turnover=1.0,
            min_market_cap=100e8
        )
        
        # 应该筛选掉一些股票
        self.assertLessEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()