#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBS工业品数据对齐脚本

功能：
1. 解析NBS原始JSON数据（工业品产量、固定资产投资、价格指数）
2. 提取关键工业品指标
3. 对齐到申万行业分类
4. 输出可用于预测的特征数据
"""

import pandas as pd
import json
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.data.macro.paths import MACRO_DIR


class NBSIndustrialDataAligner:
    """NBS工业品数据对齐器"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化对齐器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir or str(MACRO_DIR)
        
        # 工业品到申万行业的映射（简化版，需要完善）
        self.product_to_sw_industry = {
            # 钢铁相关
            'A02090101': '钢铁',  # 铁矿石原矿产量
            'A02090105': '钢铁',  # 生铁产量
            'A02090109': '钢铁',  # 粗钢产量
            'A02090113': '钢铁',  # 钢材产量
            
            # 有色金属相关
            'A02090117': '有色金属',  # 十种有色金属产量
            'A02090121': '有色金属',  # 精炼铜产量
            'A02090125': '有色金属',  # 电解铝产量
            'A02090129': '有色金属',  # 铅产量
            'A02090133': '有色金属',  # 锌产量
            
            # 化工相关
            'A02090137': '基础化工',  # 硫酸产量
            'A02090141': '基础化工',  # 烧碱产量
            'A02090145': '基础化工',  # 纯碱产量
            'A02090149': '基础化工',  # 乙烯产量
            'A02090153': '基础化工',  # 化肥产量
            'A02090157': '基础化工',  # 化学农药原药产量
            
            # 建材相关
            'A02090161': '建筑材料',  # 水泥产量
            'A02090165': '建筑材料',  # 平板玻璃产量
            
            # 能源相关
            'A02090169': '煤炭',  # 原煤产量
            'A02090173': '石油石化',  # 原油产量
            'A02090177': '石油石化',  # 天然气产量
            'A02090181': '电力设备',  # 发电量
            
            # 汽车相关
            'A02090185': '汽车',  # 汽车产量
            'A02090189': '汽车',  # 轿车产量
            
            # 电子相关
            'A02090193': '电子',  # 移动通信手持机产量
            'A02090197': '电子',  # 微型计算机设备产量
            'A02090201': '电子',  # 集成电路产量
            
            # 家电相关
            'A02090205': '家用电器',  # 家用电冰箱产量
            'A02090209': '家用电器',  # 房间空气调节器产量
            'A02090213': '家用电器',  # 家用洗衣机产量
        }

        # 基于产品名称的行业关键词映射（用于 nbs_output_2020.csv 等）
        self.product_keywords = [
            (['钢', '铁矿', '生铁', '粗钢', '钢材', '钢筋', '线材', '钢带', '冷轧'], '钢铁'),
            (['玻璃'], '建筑材料'),
            (['汽车', '轿车', 'SUV', '载货汽车'], '汽车'),
            (['动车组', '铁路机车'], '机械设备'),
            (['船舶'], '国防军工'),
            (['工业机器人', '拖拉机'], '机械设备'),
            (['大气污染防治设备'], '环保'),
            (['原盐', '磷矿石'], '基础化工'),
            (['乳制品', '成品糖', '白酒', '植物油', '饲料', '鲜、冷藏肉'], '食品饮料'),
        ]
    
    def parse_nbs_json(self, json_file: str) -> pd.DataFrame:
        """
        解析NBS JSON数据
        
        Args:
            json_file: JSON文件路径
        
        Returns:
            DataFrame: 解析后的数据
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'returndata' not in data:
            return pd.DataFrame()
        
        returndata = data['returndata']
        
        # 获取维度节点
        zb_nodes = []  # 指标节点
        sj_nodes = []  # 时间节点
        
        for node in returndata.get('wdnodes', []):
            if node['wdcode'] == 'zb':
                zb_nodes = node['nodes']
            elif node['wdcode'] == 'sj':
                sj_nodes = node['nodes']
        
        # 创建指标代码到名称的映射
        zb_code_to_name = {node['code']: node['name'] for node in zb_nodes}
        zb_code_to_unit = {node['code']: node.get('unit', '') for node in zb_nodes}
        
        # 创建时间代码到名称的映射
        sj_code_to_name = {node['code']: node['name'] for node in sj_nodes}
        
        # 解析数据节点
        records = []
        for datanode in returndata.get('datanodes', []):
            if not datanode['data']['hasdata']:
                continue
            
            # 获取指标代码和时间代码
            wds = datanode['wds']
            zb_code = None
            sj_code = None
            
            for wd in wds:
                if wd['wdcode'] == 'zb':
                    zb_code = wd['valuecode']
                elif wd['wdcode'] == 'sj':
                    sj_code = wd['valuecode']
            
            if zb_code and sj_code:
                value = datanode['data']['data']
                records.append({
                    'product_code': zb_code,
                    'product_name': zb_code_to_name.get(zb_code, ''),
                    'time_code': sj_code,
                    'time_name': sj_code_to_name.get(sj_code, ''),
                    'value': value
                })
        
        df = pd.DataFrame(records)
        
        # 转换时间代码为日期
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['time_code'].astype(str), format='%Y%m')
        
        return df
    
    def align_to_sw_industry(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对齐到申万行业
        
        Args:
            df: 原始数据
        
        Returns:
            DataFrame: 对齐后的数据
        """
        if len(df) == 0:
            return pd.DataFrame()
        
        # 添加申万行业映射
        df['sw_industry'] = df['product_code'].map(self.product_to_sw_industry)

        # 基于名称的回退映射
        if 'product_name' in df.columns:
            missing_mask = df['sw_industry'].isna()
            if missing_mask.any():
                df.loc[missing_mask, 'sw_industry'] = df.loc[missing_mask, 'product_name'].apply(
                    self._infer_industry_by_name
                )
        
        # 移除无法映射的数据
        df = df[df['sw_industry'].notna()]
        
        return df

    def _infer_industry_by_name(self, name: str) -> Optional[str]:
        if not isinstance(name, str) or not name:
            return None
        for keywords, industry in self.product_keywords:
            if any(k in name for k in keywords):
                return industry
        return None

    def load_output_csv(self) -> Optional[pd.DataFrame]:
        """
        加载工业品产量CSV（优先nbs_output_2020.csv，其次nbs_output_202512.csv）
        """
        candidates = [
            os.path.join(self.data_dir, 'nbs_output_2020.csv'),
            os.path.join(self.data_dir, 'nbs_output_202512.csv'),
        ]
        for path in candidates:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if 'product' in df.columns:
                    df = df.rename(columns={'product': 'product_name'})
                return df
        return None
    
    def calculate_growth_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算增长率
        
        Args:
            df: 包含原始值的数据
        
        Returns:
            DataFrame: 包含增长率的数据
        """
        if len(df) == 0:
            return df
        
        # 按产品和时间排序
        df = df.sort_values(['product_code', 'date']).reset_index(drop=True)
        
        # 计算同比增长率
        df['yoy'] = df.groupby('product_code')['value'].pct_change(periods=12) * 100
        
        # 计算环比增长率
        df['mom'] = df.groupby('product_code')['value'].pct_change() * 100
        
        return df
    
    def aggregate_to_industry_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        聚合到行业级别
        
        Args:
            df: 对齐后的数据
        
        Returns:
            DataFrame: 行业级别数据
        """
        if len(df) == 0:
            return pd.DataFrame()
        
        # 按行业和日期分组，计算平均值
        industry_df = df.groupby(['sw_industry', 'date']).agg({
            'yoy': 'mean',
            'mom': 'mean',
            'value': 'sum'
        }).reset_index()
        
        # 重命名列
        industry_df = industry_df.rename(columns={
            'yoy': 'output_yoy',
            'mom': 'output_mom'
        })
        
        return industry_df
    
    def process_all(self) -> Dict[str, pd.DataFrame]:
        """
        处理所有NBS数据
        
        Returns:
            Dict: 包含所有处理后的数据
        """
        print("=" * 80)
        print("NBS工业品数据对齐")
        print("=" * 80)
        
        results = {}
        
        # 1. 处理工业品产量数据
        print("\n1. 处理工业品产量数据...")
        output_df = self.load_output_csv()
        if output_df is None:
            output_file = os.path.join(self.data_dir, 'A020901_工业品产量.json')
            if os.path.exists(output_file):
                output_df = self.parse_nbs_json(output_file)
        if output_df is not None and len(output_df) > 0:
            print(f"  原始数据: {len(output_df)}条记录")

            if 'date' in output_df.columns:
                output_df['date'] = pd.to_datetime(output_df['date'])
            else:
                output_df['date'] = pd.to_datetime(output_df['time_code'].astype(str), format='%Y%m')

            # 计算增长率
            if 'output_value' in output_df.columns:
                output_df = output_df.rename(columns={'output_value': 'value'})
            output_df = self.calculate_growth_rate(output_df)

            # 对齐到申万行业
            output_df = self.align_to_sw_industry(output_df)
            print(f"  对齐后: {len(output_df)}条记录")

            # 聚合到行业级别
            industry_output_df = self.aggregate_to_industry_level(output_df)
            print(f"  行业级别: {len(industry_output_df)}条记录")

            results['output'] = {
                'raw': output_df,
                'industry': industry_output_df
            }
        
        # 2. 处理固定资产投资数据
        print("\n2. 处理固定资产投资数据...")
        fai_file = os.path.join(self.data_dir, 'A0403_固定资产投资.json')
        if os.path.exists(fai_file):
            fai_df = self.parse_nbs_json(fai_file)
            print(f"  原始数据: {len(fai_df)}条记录")
            
            # 计算增长率
            fai_df = self.calculate_growth_rate(fai_df)
            
            # 对齐到申万行业
            fai_df = self.align_to_sw_industry(fai_df)
            print(f"  对齐后: {len(fai_df)}条记录")
            
            # 聚合到行业级别
            industry_fai_df = self.aggregate_to_industry_level(fai_df)
            print(f"  行业级别: {len(industry_fai_df)}条记录")
            
            results['fai'] = {
                'raw': fai_df,
                'industry': industry_fai_df
            }
        
        # 3. 处理价格指数数据
        print("\n3. 处理价格指数数据...")
        price_file = os.path.join(self.data_dir, 'A010D02_价格指数.json')
        if os.path.exists(price_file):
            price_df = self.parse_nbs_json(price_file)
            print(f"  原始数据: {len(price_df)}条记录")
            
            # 价格指数本身就是增长率形式，不需要额外计算
            price_df['yoy'] = price_df['value'] - 100  # 转换为百分比形式
            
            # 对齐到申万行业
            price_df = self.align_to_sw_industry(price_df)
            print(f"  对齐后: {len(price_df)}条记录")
            
            # 聚合到行业级别
            industry_price_df = self.aggregate_to_industry_level(price_df)
            print(f"  行业级别: {len(industry_price_df)}条记录")
            
            results['price'] = {
                'raw': price_df,
                'industry': industry_price_df
            }
        
        # 4. 合并所有数据
        print("\n4. 合并所有数据...")
        all_industry_data = []
        
        for data_type, data_dict in results.items():
            if 'industry' in data_dict and len(data_dict['industry']) > 0:
                df = data_dict['industry'].copy()
                df['data_type'] = data_type
                
                # 统一列名
                if 'output_yoy' in df.columns:
                    df['yoy'] = df['output_yoy']
                if 'fai_yoy' in df.columns:
                    df['yoy'] = df['fai_yoy']
                if 'yoy' in df.columns and 'output_yoy' not in df.columns and data_type == 'output':
                    df['output_yoy'] = df['yoy']

                if 'output_mom' in df.columns:
                    df['mom'] = df['output_mom']
                if 'fai_mom' in df.columns:
                    df['mom'] = df['fai_mom']
                if 'mom' in df.columns and 'output_mom' not in df.columns and data_type == 'output':
                    df['output_mom'] = df['mom']
                
                all_industry_data.append(df)
        
        if all_industry_data:
            combined_df = pd.concat(all_industry_data, ignore_index=True)
            print(f"  合并后: {len(combined_df)}条记录")
            print(f"  可用列: {combined_df.columns.tolist()}")
            
            # 检查可用的数值列
            value_cols = [col for col in combined_df.columns if col in ['yoy', 'mom', 'value']]
            print(f"  数值列: {value_cols}")
            
            if value_cols:
                # 透视表格式
                pivot_df = combined_df.pivot_table(
                    index=['sw_industry', 'date'],
                    columns='data_type',
                    values=value_cols,
                    aggfunc='mean'
                )
                
                # 展平列名
                pivot_df.columns = [f'{data_type}_{metric}' for metric, data_type in pivot_df.columns]
                pivot_df = pivot_df.reset_index()
                
                print(f"  透视表: {len(pivot_df)}条记录")
                print(f"  列名: {pivot_df.columns.tolist()}")
                
                results['combined'] = pivot_df
            else:
                print("  警告: 没有可用的数值列")
                results['combined'] = pd.DataFrame()
        
        print("\n" + "=" * 80)
        print("数据对齐完成")
        print("=" * 80)
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="宏观数据目录")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    aligner = NBSIndustrialDataAligner(data_dir=args.data_dir)
    results = aligner.process_all()

    # 保存结果
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if 'combined' in results:
        output_file = os.path.join(output_dir, 'nbs_industrial_aligned.parquet')
        results['combined'].to_parquet(output_file, index=False)
        print(f"\n对齐后的数据已保存到: {output_file}")

        # 显示样例数据
        print("\n样例数据:")
        print(results['combined'].head(10))


if __name__ == '__main__':
    main()
