#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
行业场景分类器

功能：
1. 识别四种场景：复苏(RECOVERY)、大涨(BOOM)、衰退(RECESSION)、震荡(NEUTRAL)
2. 支持批量分类
3. 支持自定义阈值
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class IndustryScenarioClassifier:
    """
    行业场景分类器
    
    识别场景：
    - RECOVERY: 行业复苏（最困难时期已过，供需改善）
    - RECOVERY (STRONG): 强复苏（北向资金确认）
    - BOOM / BUBBLE: 行业大涨（泡沫化）
    - INDUSTRY RECESSION: 行业衰退
    - NEUTRAL / MIXED: 震荡/观察
    """
    
    def __init__(self, custom_thresholds: Dict = None):
        """
        初始化分类器
        
        Args:
            custom_thresholds: 自定义阈值，如不指定则使用默认值
        """
        self.thresholds = custom_thresholds or self._get_default_thresholds()
    
    def _get_default_thresholds(self) -> Dict:
        """
        获取默认阈值
        
        Returns:
            Dict: 默认阈值
        """
        return {
            'systemic_recession': {
                'credit_growth': 9.5,
                'pmi': 48.5
            },
            'boom': {
                'pb_percentile': 80,
                'turnover_rate': 0.08,
                'rps_120': 90
            },
            'recovery': {
                'ppi_yoy': -2,
                'pb_percentile': 60
            },
            'recession': {
                'ppi_yoy': -5,
                'fai_yoy': 0,
                'rev_yoy': 0
            }
        }
    
    def classify(self, data: Dict) -> str:
        """
        分类场景
        
        Args:
            data: 包含所有指标的字典
                - ppi_yoy: PPI同比增速
                - fai_yoy: 固定资产投资同比
                - inv_yoy: 存货同比增速
                - rev_yoy: 营收同比增速
                - pb_percentile: PB历史分位数
                - turnover_rate: 换手率
                - rps_120: 120日相对强度
                - credit_growth: 社融增速
                - pmi_value: PMI数值
                - northbound_signal: 北向资金信号
                - industry_ratio: 行业持仓占比
        
        Returns:
            str: 场景标签
        """
        # 1. 优先判断全局衰退
        if self._check_systemic_recession(data):
            return "SYSTEMIC RECESSION"
        
        # 2. 判断行业大涨
        if self._check_boom(data):
            return "BOOM / BUBBLE"
        
        # 3. 判断行业复苏
        recovery_type = self._check_recovery(data)
        if recovery_type:
            return recovery_type
        
        # 4. 判断行业衰退
        if self._check_recession(data):
            return "INDUSTRY RECESSION"
        
        # 5. 默认状态
        return "NEUTRAL / MIXED"
    
    def _check_systemic_recession(self, data: Dict) -> bool:
        """
        检查是否系统衰退
        
        Args:
            data: 指标数据
        
        Returns:
            bool: 是否系统衰退
        """
        credit_growth = data.get('credit_growth', None)
        pmi_value = data.get('pmi', None)
        
        if credit_growth is not None and credit_growth < self.thresholds['systemic_recession']['credit_growth']:
            return True
        
        if pmi_value is not None and pmi_value < self.thresholds['systemic_recession']['pmi']:
            return True
        
        return False
    
    def _check_boom(self, data: Dict) -> bool:
        """
        检查是否大涨
        
        Args:
            data: 指标数据
        
        Returns:
            bool: 是否大涨
        """
        pb_percentile = data.get('pb_percentile', 0)
        turnover_rate = data.get('turnover_rate', 0)
        rps_120 = data.get('rps_120', 0)
        
        return (
            pb_percentile > self.thresholds['boom']['pb_percentile'] and
            turnover_rate > self.thresholds['boom']['turnover_rate'] and
            rps_120 > self.thresholds['boom']['rps_120']
        )
    
    def _check_recovery(self, data: Dict) -> str:
        """
        检查是否复苏
        
        Args:
            data: 指标数据
        
        Returns:
            str: 场景标签（如果不是复苏则返回None）
        """
        ppi_yoy = data.get('ppi_yoy', -999)
        inv_yoy = data.get('inv_yoy', 0)
        rev_yoy = data.get('rev_yoy', 0)
        pb_percentile = data.get('pb_percentile', 100)
        northbound_signal = data.get('northbound_signal', 0)
        
        # 判断条件
        ppi_improving = ppi_yoy > self.thresholds['recovery']['ppi_yoy']
        inventory_cleared = inv_yoy < rev_yoy and inv_yoy < 10
        reasonable_val = pb_percentile < self.thresholds['recovery']['pb_percentile']
        
        # 北向资金信号
        northbound_increasing = northbound_signal > 0
        
        if ppi_improving and inventory_cleared and reasonable_val and northbound_increasing:
            return "RECOVERY (STRONG)"
        elif ppi_improving and inventory_cleared and reasonable_val:
            return "RECOVERY"
        
        return None
    
    def _check_recession(self, data: Dict) -> bool:
        """
        检查是否衰退
        
        Args:
            data: 指标数据
        
        Returns:
            bool: 是否衰退
        """
        ppi_yoy = data.get('ppi_yoy', 999)
        fai_yoy = data.get('fai_yoy', None)
        rev_yoy = data.get('rev_yoy', None)
        
        conditions = [
            ppi_yoy < self.thresholds['recession']['ppi_yoy']
        ]
        
        if fai_yoy is not None:
            conditions.append(fai_yoy < self.thresholds['recession']['fai_yoy'])
        
        if rev_yoy is not None:
            conditions.append(rev_yoy < self.thresholds['recession']['rev_yoy'])
        
        return all(conditions)
    
    def batch_classify(self, df: pd.DataFrame) -> pd.Series:
        """
        批量分类
        
        Args:
            df: 包含所有指标的数据框
        
        Returns:
            pd.Series: 场景标签序列
        """
        results = []
        for idx, row in df.iterrows():
            scenario = self.classify(row.to_dict())
            results.append(scenario)
        return pd.Series(results, index=df.index)
    
    def get_scenario_distribution(self, scenarios: List[str]) -> Dict[str, int]:
        """
        获取场景分布
        
        Args:
            scenarios: 场景标签列表
        
        Returns:
            Dict: 场景分布统计
        """
        distribution = {}
        for scenario in scenarios:
            distribution[scenario] = distribution.get(scenario, 0) + 1
        return distribution


def main():
    """测试分类器"""
    classifier = IndustryScenarioClassifier()
    
    # 测试各种场景
    test_cases = [
        {
            'name': '系统衰退',
            'data': {
                'credit_growth': 9.0,
                'pmi': 48.0,
                'ppi_yoy': -3,
                'pb_percentile': 30,
                'turnover_rate': 0.02,
                'rps_120': 40
            }
        },
        {
            'name': '行业大涨',
            'data': {
                'credit_growth': 12,
                'pmi': 51,
                'ppi_yoy': 10,
                'pb_percentile': 90,
                'turnover_rate': 0.12,
                'rps_120': 95
            }
        },
        {
            'name': '行业复苏',
            'data': {
                'credit_growth': 11,
                'pmi': 50,
                'ppi_yoy': -1,
                'inv_yoy': 2,
                'rev_yoy': 8,
                'pb_percentile': 40,
                'turnover_rate': 0.04,
                'rps_120': 70,
                'northbound_signal': 0
            }
        },
        {
            'name': '强复苏',
            'data': {
                'credit_growth': 11,
                'pmi': 50,
                'ppi_yoy': -1,
                'inv_yoy': 2,
                'rev_yoy': 8,
                'pb_percentile': 40,
                'turnover_rate': 0.04,
                'rps_120': 70,
                'northbound_signal': 1
            }
        },
        {
            'name': '行业衰退',
            'data': {
                'credit_growth': 11,
                'pmi': 50,
                'ppi_yoy': -6,
                'fai_yoy': -2,
                'rev_yoy': -3,
                'pb_percentile': 50,
                'turnover_rate': 0.03,
                'rps_120': 45
            }
        },
        {
            'name': '震荡',
            'data': {
                'credit_growth': 11,
                'pmi': 50,
                'ppi_yoy': 1,
                'pb_percentile': 70,
                'turnover_rate': 0.05,
                'rps_120': 65
            }
        }
    ]
    
    print("=" * 80)
    print("行业场景分类器测试")
    print("=" * 80)
    print()
    
    for test in test_cases:
        scenario = classifier.classify(test['data'])
        print(f"测试: {test['name']}")
        print(f"结果: {scenario}")
        print(f"指标: {test['data']}")
        print()


if __name__ == '__main__':
    main()