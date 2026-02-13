#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
周期拐点检测器

功能：
1. 检测复苏拐点
2. 检测衰退拐点
3. 检测周期阶段
4. 支持多周期分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum


class CyclePhase(Enum):
    """周期阶段"""
    RECOVERY = "复苏期"
    EXPANSION = "扩张期"
    PEAK = "顶峰期"
    RECESSION = "衰退期"
    BOTTOM = "底部期"


class CycleDetector:
    """
    周期拐点检测器
    
    检测类型：
    1. 复苏拐点：PPI连续回升 + CAPEX连续增长
    2. 衰退拐点：PPI连续下跌 + CAPEX连续下滑
    3. 周期阶段：根据指标组合判断当前处于哪个阶段
    """
    
    def __init__(self, window: int = 3):
        """
        初始化检测器
        
        Args:
            window: 检测窗口大小（月数）
        """
        self.window = window
    
    def detect_recovery(
        self,
        ppi_series: pd.Series,
        fai_series: pd.Series,
        inventory_series: Optional[pd.Series] = None
    ) -> Dict:
        """
        检测复苏拐点
        
        条件：
        1. PPI连续2个月回升
        2. FAI连续3个月增长
        3. （可选）存货增速下降（被动去库）
        
        Args:
            ppi_series: PPI序列
            fai_series: FAI序列
            inventory_series: 存货序列（可选）
        
        Returns:
            Dict: 检测结果
        """
        result = {
            'is_recovery': False,
            'confidence': 0,
            'signals': {},
            'phase': None
        }
        
        # 1. 检测PPI回升
        ppi_improving = self._detect_continuous_increase(ppi_series, periods=2)
        result['signals']['ppi_improving'] = ppi_improving
        
        # 2. 检测FAI增长
        fai_expanding = self._detect_continuous_increase(fai_series, periods=3)
        result['signals']['fai_expanding'] = fai_expanding
        
        # 3. 检测库存去化（可选）
        if inventory_series is not None:
            inventory_clearing = self._detect_decreasing_trend(inventory_series)
            result['signals']['inventory_clearing'] = inventory_clearing
        
        # 4. 综合判断
        if ppi_improving and fai_expanding:
            result['is_recovery'] = True
            
            # 计算置信度
            confidence = 0.6  # 基础置信度
            if inventory_series is not None and result['signals']['inventory_clearing']:
                confidence += 0.2  # 库存去化增加置信度
            
            # PPI回升幅度
            ppi_change = ppi_series.iloc[-1] - ppi_series.iloc[-self.window]
            if ppi_change > 2:
                confidence += 0.2  # PPI大幅回升增加置信度
            
            result['confidence'] = min(confidence, 1.0)
            result['phase'] = CyclePhase.RECOVERY
        
        return result
    
    def detect_recession(
        self,
        ppi_series: pd.Series,
        fai_series: pd.Series,
        roe_series: Optional[pd.Series] = None
    ) -> Dict:
        """
        检测衰退拐点
        
        条件：
        1. PPI连续6个月下跌
        2. FAI连续3季度下滑
        3. （可选）ROE连续下降
        
        Args:
            ppi_series: PPI序列
            fai_series: FAI序列
            roe_series: ROE序列（可选）
        
        Returns:
            Dict: 检测结果
        """
        result = {
            'is_recession': False,
            'confidence': 0,
            'signals': {},
            'phase': None
        }
        
        # 1. 检测PPI下跌
        ppi_declining = self._detect_continuous_decrease(ppi_series, periods=6)
        result['signals']['ppi_declining'] = ppi_declining
        
        # 2. 检测FAI下滑
        fai_shrinking = self._detect_continuous_decrease(fai_series, periods=3)
        result['signals']['fai_shrinking'] = fai_shrinking
        
        # 3. 检测ROE下降（可选）
        if roe_series is not None:
            roe_declining = self._detect_decreasing_trend(roe_series)
            result['signals']['roe_declining'] = roe_declining
        
        # 4. 综合判断
        if ppi_declining and fai_shrinking:
            result['is_recession'] = True
            
            # 计算置信度
            confidence = 0.6  # 基础置信度
            if roe_series is not None and result['signals']['roe_declining']:
                confidence += 0.2  # ROE下降增加置信度
            
            # PPI下跌幅度
            ppi_change = ppi_series.iloc[-1] - ppi_series.iloc[-self.window]
            if ppi_change < -3:
                confidence += 0.2  # PPI大幅下跌增加置信度
            
            result['confidence'] = min(confidence, 1.0)
            result['phase'] = CyclePhase.RECESSION
        
        return result
    
    def detect_cycle_phase(
        self,
        ppi_series: pd.Series,
        fai_series: pd.Series,
        inventory_series: Optional[pd.Series] = None
    ) -> Dict:
        """
        检测周期阶段
        
        根据指标组合判断当前处于哪个阶段：
        - 复苏期：PPI回升 + FAI增长
        - 扩张期：PPI高位 + FAI高增长
        - 顶峰期：PPI高位 + FAI下滑
        - 衰退期：PPI下跌 + FAI下滑
        - 底部期：PPI低位 + FAI低位
        
        Args:
            ppi_series: PPI序列
            fai_series: FAI序列
            inventory_series: 存货序列（可选）
        
        Returns:
            Dict: 周期阶段信息
        """
        result = {
            'phase': None,
            'confidence': 0,
            'characteristics': {}
        }
        
        # 计算趋势
        ppi_trend = self._calculate_trend(ppi_series)
        fai_trend = self._calculate_trend(fai_series)
        
        ppi_level = self._get_level(ppi_series)
        fai_level = self._get_level(fai_series)
        
        # 判断周期阶段
        if ppi_trend > 0.5 and fai_trend > 0.3:
            if ppi_level > 0.7 and fai_level > 0.7:
                result['phase'] = CyclePhase.EXPANSION
                result['confidence'] = 0.8
            else:
                result['phase'] = CyclePhase.RECOVERY
                result['confidence'] = 0.7
        elif ppi_trend < -0.5 and fai_trend < -0.3:
            result['phase'] = CyclePhase.RECESSION
            result['confidence'] = 0.8
        elif ppi_level > 0.7 and fai_trend < -0.3:
            result['phase'] = CyclePhase.PEAK
            result['confidence'] = 0.6
        elif ppi_level < 0.3 and fai_level < 0.3:
            result['phase'] = CyclePhase.BOTTOM
            result['confidence'] = 0.7
        else:
            result['phase'] = CyclePhase.RECOVERY  # 默认为复苏
            result['confidence'] = 0.5
        
        # 记录特征
        result['characteristics'] = {
            'ppi_trend': ppi_trend,
            'fai_trend': fai_trend,
            'ppi_level': ppi_level,
            'fai_level': fai_level
        }
        
        # 库存状态（可选）
        if inventory_series is not None:
            inv_trend = self._calculate_trend(inventory_series)
            result['characteristics']['inventory_trend'] = inv_trend
            
            # 库存去化信号
            if inv_trend < 0:
                result['characteristics']['inventory_status'] = 'clearing'
            elif inv_trend > 0.3:
                result['characteristics']['inventory_status'] = 'accumulating'
            else:
                result['characteristics']['inventory_status'] = 'stable'
        
        return result
    
    def detect_inflection_points(
        self,
        series: pd.Series,
        method: str = 'slope_change'
    ) -> List[Tuple[int, str]]:
        """
        检测拐点
        
        Args:
            series: 时间序列
            method: 检测方法 ('slope_change', 'extremum')
        
        Returns:
            List[Tuple[int, str]]: 拐点列表 (索引, 类型)
        """
        inflection_points = []
        
        if method == 'slope_change':
            # 斜率变化法
            for i in range(2, len(series) - 2):
                # 前一段趋势
                prev_slope = series.iloc[i] - series.iloc[i-2]
                # 后一段趋势
                next_slope = series.iloc[i+2] - series.iloc[i]
                
                # 拐点判断
                if prev_slope * next_slope < 0:  # 斜率符号改变
                    if prev_slope > 0:
                        point_type = 'peak'  # 峰值
                    else:
                        point_type = 'bottom'  # 底部
                    inflection_points.append((i, point_type))
        
        elif method == 'extremum':
            # 极值点法
            for i in range(1, len(series) - 1):
                if series.iloc[i] > series.iloc[i-1] and series.iloc[i] > series.iloc[i+1]:
                    inflection_points.append((i, 'peak'))
                elif series.iloc[i] < series.iloc[i-1] and series.iloc[i] < series.iloc[i+1]:
                    inflection_points.append((i, 'bottom'))
        
        return inflection_points
    
    def _detect_continuous_increase(
        self,
        series: pd.Series,
        periods: int
    ) -> bool:
        """
        检测是否连续增加
        
        Args:
            series: 序列
            periods: 周期数
        
        Returns:
            bool: 是否连续增加
        """
        if len(series) < periods + 1:
            return False
        
        for i in range(len(series) - periods, len(series)):
            if pd.isna(series.iloc[i]) or pd.isna(series.iloc[i-1]):
                return False
            if series.iloc[i] <= series.iloc[i-1]:
                return False
        
        return True
    
    def _detect_continuous_decrease(
        self,
        series: pd.Series,
        periods: int
    ) -> bool:
        """
        检测是否连续减少
        
        Args:
            series: 序列
            periods: 周期数
        
        Returns:
            bool: 是否连续减少
        """
        if len(series) < periods + 1:
            return False
        
        for i in range(len(series) - periods, len(series)):
            if pd.isna(series.iloc[i]) or pd.isna(series.iloc[i-1]):
                return False
            if series.iloc[i] >= series.iloc[i-1]:
                return False
        
        return True
    
    def _detect_decreasing_trend(self, series: pd.Series) -> bool:
        """
        检测下降趋势
        
        Args:
            series: 序列
        
        Returns:
            bool: 是否下降趋势
        """
        if len(series) < 3:
            return False
        
        # 计算斜率
        x = np.arange(len(series))
        y = series.values
        
        # 去除NaN
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 3:
            return False
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        # 线性回归
        slope = np.polyfit(x_valid, y_valid, 1)[0]
        
        return slope < 0
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """
        计算趋势（-1到1）
        
        Args:
            series: 序列
        
        Returns:
            float: 趋势值
        """
        if len(series) < 3:
            return 0
        
        # 计算线性回归斜率
        x = np.arange(len(series))
        y = series.values
        
        # 去除NaN
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 3:
            return 0
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        # 归一化
        y_normalized = (y_valid - np.mean(y_valid)) / (np.std(y_valid) + 1e-8)
        
        # 线性回归
        slope = np.polyfit(x_valid, y_normalized, 1)[0]
        
        # 归一化到[-1, 1]
        return np.clip(slope * 10, -1, 1)
    
    def _get_level(self, series: pd.Series) -> float:
        """
        获取水平（0到1）
        
        Args:
            series: 序列
        
        Returns:
            float: 水平值
        """
        if len(series) == 0:
            return 0.5
        
        current = series.iloc[-1]
        min_val = series.min()
        max_val = series.max()
        
        if max_val == min_val:
            return 0.5
        
        return (current - min_val) / (max_val - min_val)


def main():
    """测试检测器"""
    detector = CycleDetector(window=3)
    
    # 创建模拟数据
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='M')
    
    # 模拟复苏期数据
    ppi_recovery = pd.Series([-5, -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    fai_recovery = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    inventory_recovery = pd.Series([15, 14, 12, 10, 8, 6, 5, 4, 3, 2, 1, 0])
    
    # 模拟衰退期数据
    ppi_recession = pd.Series([7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4])
    fai_recession = pd.Series([13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2])
    
    print("=" * 80)
    print("周期拐点检测器测试")
    print("=" * 80)
    print()
    
    # 测试复苏检测
    print("1. 复苏拐点检测")
    print("-" * 80)
    recovery_result = detector.detect_recovery(ppi_recovery, fai_recovery, inventory_recovery)
    print(f"是否复苏: {recovery_result['is_recovery']}")
    print(f"置信度: {recovery_result['confidence']:.2f}")
    print(f"周期阶段: {recovery_result['phase']}")
    print(f"信号: {recovery_result['signals']}")
    print()
    
    # 测试衰退检测
    print("2. 衰退拐点检测")
    print("-" * 80)
    recession_result = detector.detect_recession(ppi_recession, fai_recession)
    print(f"是否衰退: {recession_result['is_recession']}")
    print(f"置信度: {recession_result['confidence']:.2f}")
    print(f"周期阶段: {recession_result['phase']}")
    print(f"信号: {recession_result['signals']}")
    print()
    
    # 测试周期阶段检测
    print("3. 周期阶段检测")
    print("-" * 80)
    phase_result = detector.detect_cycle_phase(ppi_recovery, fai_recovery, inventory_recovery)
    print(f"当前阶段: {phase_result['phase']}")
    print(f"置信度: {phase_result['confidence']:.2f}")
    print(f"特征: {phase_result['characteristics']}")
    print()
    
    # 测试拐点检测
    print("4. 拐点检测")
    print("-" * 80)
    inflection_points = detector.detect_inflection_points(ppi_recovery, method='extremum')
    print(f"发现{len(inflection_points)}个拐点:")
    for idx, point_type in inflection_points:
        print(f"  位置{idx}: {point_type}")


if __name__ == '__main__':
    main()