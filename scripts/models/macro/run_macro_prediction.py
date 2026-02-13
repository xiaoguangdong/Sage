#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
宏观经济预测主运行脚本

功能：
1. 从2020年开始每日预测
2. 支持指定日期范围预测
3. 输出预测结果到控制台
4. 支持回测模式
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from sage_core.models.macro_predictor import MacroPredictor
from scripts.data.macro.clean_macro_data import MacroDataProcessor


class MacroPredictionRunner:
    """
    宏观预测运行器
    
    功能：
    1. 加载数据
    2. 初始化预测模型
    3. 执行预测
    4. 输出结果
    """
    
    def __init__(
        self,
        data_dir: str = 'data/tushare/macro',
        processed_dir: str = 'data/processed',
        config_path: str = 'config/sw_nbs_mapping.yaml'
    ):
        """
        初始化运行器
        
        Args:
            data_dir: 原始数据目录
            processed_dir: 处理后数据目录
            config_path: 配置文件路径
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.config_path = config_path
        self.predictor = None
        self.macro_data = None
        self.industry_data = None
        self.northbound_data = None
    
    def load_data(self):
        """加载数据"""
        print("=" * 80)
        print("加载数据")
        print("=" * 80)
        
        # 检查是否有处理后的数据
        macro_path = os.path.join(self.processed_dir, 'macro_features.parquet')
        industry_path = os.path.join(self.processed_dir, 'industry_features.parquet')
        northbound_path = os.path.join(self.processed_dir, 'northbound_features.parquet')
        
        if os.path.exists(macro_path) and os.path.exists(industry_path):
            print("加载处理后的数据...")
            self.macro_data = pd.read_parquet(macro_path)
            self.industry_data = pd.read_parquet(industry_path)
            
            if os.path.exists(northbound_path):
                self.northbound_data = pd.read_parquet(northbound_path)
            else:
                self.northbound_data = None
                print("北向资金数据不存在，将不使用")
            
            print(f"宏观数据: {len(self.macro_data)}条")
            print(f"行业数据: {len(self.industry_data)}条")
            if self.northbound_data is not None:
                print(f"北向资金: {len(self.northbound_data)}条")
        else:
            print("处理后数据不存在，开始处理原始数据...")
            processor = MacroDataProcessor(self.data_dir)
            data = processor.process_all()
            
            self.macro_data = data['macro']
            self.industry_data = data['industry']
            self.northbound_data = data['northbound']
            
            # 保存处理后的数据
            os.makedirs(self.processed_dir, exist_ok=True)
            self.macro_data.to_parquet(macro_path, index=False)
            self.industry_data.to_parquet(industry_path, index=False)
            
            if self.northbound_data is not None and len(self.northbound_data) > 0:
                self.northbound_data.to_parquet(northbound_path, index=False)
    
    def init_predictor(self):
        """初始化预测模型"""
        print("\n" + "=" * 80)
        print("初始化预测模型")
        print("=" * 80)
        
        self.predictor = MacroPredictor(self.config_path)
        print("预测模型初始化完成")
    
    def predict_daily(
        self,
        start_date: str,
        end_date: str,
        output_file: Optional[str] = None
    ) -> List[Dict]:
        """
        每日预测
        
        Args:
            start_date: 开始日期（格式：YYYY-MM-DD）
            end_date: 结束日期（格式：YYYY-MM-DD）
            output_file: 输出文件路径（可选）
        
        Returns:
            List[Dict]: 预测结果列表
        """
        print("\n" + "=" * 80)
        print(f"每日预测: {start_date} ~ {end_date}")
        print("=" * 80)
        
        # 生成日期序列
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        dates = pd.date_range(start_dt, end_dt, freq='D')
        
        results = []
        
        for i, date in enumerate(dates, 1):
            date_str = date.strftime('%Y-%m-%d')
            
            print(f"\n[{i}/{len(dates)}] 预测日期: {date_str}")
            print("-" * 80)
            
            # 执行预测
            result = self.predictor.predict(
                date=date_str,
                macro_data=self.macro_data,
                industry_data=self.industry_data,
                northbound_data=self.northbound_data
            )
            
            # 输出结果
            self._print_prediction_result(result)
            
            results.append(result)
        
        # 保存结果
        if output_file:
            self._save_results(results, output_file)
            print(f"\n预测结果已保存到: {output_file}")
        
        return results
    
    def predict_backtest(
        self,
        backtest_periods: List[Dict[str, str]]
    ) -> Dict[str, List[Dict]]:
        """
        回测预测
        
        Args:
            backtest_periods: 回测周期列表
                [{'start': '2024-09-01', 'end': '2025-01-01', 'name': '牛市1'}, ...]
        
        Returns:
            Dict: 各周期的预测结果
        """
        print("\n" + "=" * 80)
        print("回测预测模式")
        print("=" * 80)
        
        all_results = {}
        
        for period in backtest_periods:
            print(f"\n回测周期: {period['name']} ({period['start']} ~ {period['end']})")
            print("=" * 80)
            
            results = self.predict_daily(
                start_date=period['start'],
                end_date=period['end']
            )
            
            all_results[period['name']] = results
        
        # 输出回测汇总
        self._print_backtest_summary(all_results)
        
        return all_results
    
    def _print_prediction_result(self, result: Dict):
        """
        打印预测结果
        
        Args:
            result: 预测结果
        """
        print(f"系统场景: {result['systemic_scenario']}")
        print(f"风险等级: {result['risk_level']}")
        print(f"摘要: {result['summary']}")
        
        if result['opportunity_industries']:
            print(f"\n机会行业 ({len(result['opportunity_industries'])}个):")
            for i, ind in enumerate(result['opportunity_industries'][:10], 1):  # 最多显示10个
                print(f"  {i}. {ind['industry']} ({ind['scenario']}) - 景气度: {ind['boom_score']:.1f}")
        else:
            print("\n暂无机会行业")
    
    def _print_backtest_summary(self, all_results: Dict[str, List[Dict]]):
        """
        打印回测汇总
        
        Args:
            all_results: 所有回测结果
        """
        print("\n" + "=" * 80)
        print("回测汇总")
        print("=" * 80)
        
        for period_name, results in all_results.items():
            print(f"\n周期: {period_name}")
            print(f"预测天数: {len(results)}")
            
            # 统计系统衰退天数
            recession_days = sum(1 for r in results if r['systemic_scenario'] == 'SYSTEMIC RECESSION')
            print(f"系统衰退天数: {recession_days} ({recession_days/len(results)*100:.1f}%)")
            
            # 统计机会行业数量
            opportunity_counts = [len(r['opportunity_industries']) for r in results]
            avg_opportunities = np.mean(opportunity_counts) if opportunity_counts else 0
            max_opportunities = max(opportunity_counts) if opportunity_counts else 0
            print(f"平均机会行业数: {avg_opportunities:.1f}")
            print(f"最大机会行业数: {max_opportunities}")
            
            # 统计场景分布
            scenario_counts = {}
            for r in results:
                for ind in r['opportunity_industries']:
                    scenario = ind['scenario']
                    scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            
            if scenario_counts:
                print(f"场景分布:")
                for scenario, count in scenario_counts.items():
                    print(f"  - {scenario}: {count}次")
    
    def _save_results(self, results: List[Dict], output_file: str):
        """
        保存预测结果
        
        Args:
            results: 预测结果列表
            output_file: 输出文件路径
        """
        # 转换为DataFrame
        records = []
        for result in results:
            record = {
                'date': result['date'],
                'systemic_scenario': result['systemic_scenario'],
                'risk_level': result['risk_level'],
                'summary': result['summary'],
                'opportunity_count': len(result['opportunity_industries'])
            }
            
            # 添加TOP 5机会行业
            for i in range(5):
                if i < len(result['opportunity_industries']):
                    ind = result['opportunity_industries'][i]
                    record[f'top{i+1}_industry'] = ind['industry']
                    record[f'top{i+1}_scenario'] = ind['scenario']
                    record[f'top{i+1}_score'] = ind['boom_score']
                else:
                    record[f'top{i+1}_industry'] = ''
                    record[f'top{i+1}_scenario'] = ''
                    record[f'top{i+1}_score'] = 0
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # 保存
        if output_file.endswith('.csv'):
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
        else:
            df.to_parquet(output_file, index=False)
    
    def run(
        self,
        mode: str = 'daily',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_file: Optional[str] = None
    ):
        """
        运行预测
        
        Args:
            mode: 运行模式 ('daily', 'backtest')
            start_date: 开始日期
            end_date: 结束日期
            output_file: 输出文件
        """
        # 加载数据
        self.load_data()
        
        # 初始化模型
        self.init_predictor()
        
        # 执行预测
        if mode == 'daily':
            if start_date is None:
                start_date = '2020-01-01'
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            results = self.predict_daily(start_date, end_date, output_file)
        
        elif mode == 'backtest':
            # 定义回测周期（3个牛市阶段）
            backtest_periods = [
                {
                    'start': '2024-09-01',
                    'end': '2025-01-01',
                    'name': '牛市阶段1'
                },
                {
                    'start': '2024-02-01',
                    'end': '2024-04-01',
                    'name': '牛市阶段2'
                },
                {
                    'start': '2020-07-01',
                    'end': '2020-09-01',
                    'name': '牛市阶段3'
                }
            ]
            
            results = self.predict_backtest(backtest_periods)
            
            # 保存结果
            if output_file:
                for period_name, period_results in results.items():
                    period_output = output_file.replace('.csv', f'_{period_name}.csv')
                    self._save_results(period_results, period_output)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='宏观经济预测')
    parser.add_argument('--mode', type=str, default='backtest', 
                       choices=['daily', 'backtest'],
                       help='运行模式: daily(每日预测) 或 backtest(回测)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径')
    parser.add_argument('--data-dir', type=str, default='data/tushare/macro',
                       help='数据目录')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                       help='处理后数据目录')
    parser.add_argument('--config', type=str, default='config/sw_nbs_mapping.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建运行器
    runner = MacroPredictionRunner(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        config_path=args.config
    )
    
    # 运行预测
    runner.run(
        mode=args.mode,
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
