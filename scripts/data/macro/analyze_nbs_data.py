#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBS数据分析脚本

深入分析国家统计局原始数据，发现工业品产量、投资、价格的变化规律
"""

import pandas as pd
import json
import os
from datetime import datetime
import numpy as np

from scripts.data.macro.paths import MACRO_DIR

class NBSDataAnalyzer:
    """NBS数据分析器"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化分析器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir or str(MACRO_DIR)
    
    def parse_nbs_json(self, json_file: str) -> pd.DataFrame:
        """解析NBS JSON数据"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'returndata' not in data:
            return pd.DataFrame()
        
        returndata = data['returndata']
        
        # 获取维度节点
        zb_nodes = []
        sj_nodes = []
        
        for node in returndata.get('wdnodes', []):
            if node['wdcode'] == 'zb':
                zb_nodes = node['nodes']
            elif node['wdcode'] == 'sj':
                sj_nodes = node['nodes']
        
        # 创建映射
        zb_code_to_name = {node['code']: node['name'] for node in zb_nodes}
        zb_code_to_unit = {node['code']: node.get('unit', '') for node in zb_nodes}
        sj_code_to_name = {node['code']: node['name'] for node in sj_nodes}
        
        # 解析数据节点
        records = []
        for datanode in returndata.get('datanodes', []):
            if not datanode['data']['hasdata']:
                continue
            
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
                    'product_unit': zb_code_to_unit.get(zb_code, ''),
                    'time_code': sj_code,
                    'time_name': sj_code_to_name.get(sj_code, ''),
                    'value': value
                })
        
        df = pd.DataFrame(records)
        
        # 转换时间
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['time_code'].astype(str), format='%Y%m')
        
        return df
    
    def analyze_output_trends(self, df: pd.DataFrame) -> dict:
        """
        分析工业品产量趋势
        
        Args:
            df: 工业品产量数据
        
        Returns:
            dict: 分析结果
        """
        if len(df) == 0:
            return {}
        
        print("\n" + "=" * 80)
        print("工业品产量趋势分析")
        print("=" * 80)
        
        results = {}
        
        # 1. 数据覆盖情况
        print(f"\n1. 数据覆盖情况")
        print(f"   总记录数: {len(df)}")
        print(f"   产品种类: {df['product_code'].nunique()}")
        print(f"   时间范围: {df['date'].min()} ~ {df['date'].max()}")
        print(f"   时间跨度: {(df['date'].max() - df['date'].min()).days / 365.25:.1f}年")
        
        # 2. 按产品分析
        print(f"\n2. 各产品数据覆盖")
        product_coverage = df.groupby('product_code').agg({
            'product_name': 'first',
            'product_unit': 'first',
            'value': 'count',
            'date': ['min', 'max']
        }).reset_index()
        product_coverage.columns = ['product_code', 'product_name', 'unit', 'count', 'start_date', 'end_date']
        product_coverage = product_coverage.sort_values('count', ascending=False)
        
        print(f"   TOP 10 产品:")
        for i, row in product_coverage.head(10).iterrows():
            print(f"     {i+1}. {row['product_name']} ({row['product_code']})")
            print(f"        数据点: {row['count']}个, 单位: {row['unit']}")
            print(f"        时间: {row['start_date'].strftime('%Y-%m')} ~ {row['end_date'].strftime('%Y-%m')}")
        
        # 3. 计算增长率
        df_sorted = df.sort_values(['product_code', 'date']).reset_index(drop=True)
        df_sorted['yoy'] = df_sorted.groupby('product_code')['value'].pct_change(periods=12) * 100
        df_sorted['mom'] = df_sorted.groupby('product_code')['value'].pct_change() * 100
        
        # 4. 最近3个月增长率排名
        print(f"\n3. 最近3个月环比增长率排名")
        recent_date = df_sorted['date'].max()
        recent_3m = df_sorted[df_sorted['date'] >= recent_date - pd.Timedelta(days=90)]
        
        recent_growth = recent_3m.groupby('product_code').agg({
            'product_name': 'first',
            'mom': ['mean', 'last']
        }).reset_index()
        recent_growth.columns = ['product_code', 'product_name', 'mom_avg_3m', 'mom_last']
        recent_growth = recent_growth.dropna(subset=['mom_avg_3m'])
        recent_growth = recent_growth.sort_values('mom_avg_3m', ascending=False)
        
        print(f"   环比增长最快TOP 10:")
        for i, row in recent_growth.head(10).iterrows():
            print(f"     {i+1}. {row['product_name']}: +{row['mom_avg_3m']:.2f}% (最新: {row['mom_last']:.2f}%)")
        
        print(f"\n   环比下降最快TOP 10:")
        for i, row in recent_growth.tail(10).iterrows():
            print(f"     {i+1}. {row['product_name']}: {row['mom_avg_3m']:.2f}% (最新: {row['mom_last']:.2f}%)")
        
        # 5. 同比增长率排名
        print(f"\n4. 最近12个月同比增长率排名")
        recent_12m = df_sorted[df_sorted['date'] >= recent_date - pd.Timedelta(days=365)]
        
        recent_yoy = recent_12m.groupby('product_code').agg({
            'product_name': 'first',
            'yoy': ['mean', 'last']
        }).reset_index()
        recent_yoy.columns = ['product_code', 'product_name', 'yoy_avg_12m', 'yoy_last']
        recent_yoy = recent_yoy.dropna(subset=['yoy_avg_12m'])
        recent_yoy = recent_yoy.sort_values('yoy_avg_12m', ascending=False)
        
        print(f"   同比增长最快TOP 10:")
        for i, row in recent_yoy.head(10).iterrows():
            print(f"     {i+1}. {row['product_name']}: +{row['yoy_avg_12m']:.2f}% (最新: {row['yoy_last']:.2f}%)")
        
        # 6. 拐点检测（连续2个月由负转正或由正转负）
        print(f"\n5. 增长率拐点检测")
        for product_code in df_sorted['product_code'].unique():
            product_data = df_sorted[df_sorted['product_code'] == product_code].tail(6)  # 最近6个月
            
            if len(product_data) < 4:
                continue
            
            # 检查环比拐点
            mom_changes = product_data['mom'].diff()
            
            # 检测由负转正
            if (mom_changes.iloc[-2] > 0 and 
                product_data['mom'].iloc[-3] < 0 and 
                product_data['mom'].iloc[-1] > 0):
                print(f"   ⚠️  {product_data['product_name'].iloc[-1]}: 环比由负转正！")
                print(f"      近3个月环比: {product_data['mom'].iloc[-3]:.2f}% → {product_data['mom'].iloc[-2]:.2f}% → {product_data['mom'].iloc[-1]:.2f}%")
            
            # 检测由正转负
            if (mom_changes.iloc[-2] < 0 and 
                product_data['mom'].iloc[-3] > 0 and 
                product_data['mom'].iloc[-1] < 0):
                print(f"   ⚠️  {product_data['product_name'].iloc[-1]}: 环比由正转负！")
                print(f"      近3个月环比: {product_data['mom'].iloc[-3]:.2f}% → {product_data['mom'].iloc[-2]:.2f}% → {product_data['mom'].iloc[-1]:.2f}%")
        
        results = {
            'product_coverage': product_coverage,
            'recent_growth': recent_growth,
            'recent_yoy': recent_yoy,
            'df_with_metrics': df_sorted
        }
        
        return results
    
    def analyze_fai_trends(self, df: pd.DataFrame) -> dict:
        """
        分析固定资产投资趋势
        
        Args:
            df: 固定资产投资数据
        
        Returns:
            dict: 分析结果
        """
        if len(df) == 0:
            return {}
        
        print("\n" + "=" * 80)
        print("固定资产投资趋势分析")
        print("=" * 80)
        
        # 1. 数据覆盖情况
        print(f"\n1. 数据覆盖情况")
        print(f"   总记录数: {len(df)}")
        print(f"   投资领域: {df['product_code'].nunique()}")
        print(f"   时间范围: {df['date'].min()} ~ {df['date'].max()}")
        
        # 2. 按投资领域分析
        print(f"\n2. 各投资领域数据覆盖")
        fai_coverage = df.groupby('product_code').agg({
            'product_name': 'first',
            'product_unit': 'first',
            'value': 'count',
            'date': ['min', 'max']
        }).reset_index()
        fai_coverage.columns = ['product_code', 'product_name', 'unit', 'count', 'start_date', 'end_date']
        fai_coverage = fai_coverage.sort_values('count', ascending=False)
        
        print(f"   TOP 15 投资领域:")
        for i, row in fai_coverage.head(15).iterrows():
            print(f"     {i+1}. {row['product_name']} ({row['product_code']})")
            print(f"        数据点: {row['count']}个, 单位: {row['unit']}")
        
        # 3. 计算增长率
        df_sorted = df.sort_values(['product_code', 'date']).reset_index(drop=True)
        df_sorted['yoy'] = df_sorted.groupby('product_code')['value'].pct_change(periods=12) * 100
        df_sorted['mom'] = df_sorted.groupby('product_code')['value'].pct_change() * 100
        
        # 4. 最近投资增速
        print(f"\n3. 最近投资增速分析")
        recent_date = df_sorted['date'].max()
        recent_6m = df_sorted[df_sorted['date'] >= recent_date - pd.Timedelta(days=180)]
        
        recent_fai = recent_6m.groupby('product_code').agg({
            'product_name': 'first',
            'yoy': 'last',
            'mom': 'last'
        }).reset_index()
        recent_fai = recent_fai.dropna(subset=['yoy', 'mom'])
        recent_fai = recent_fai.sort_values('yoy', ascending=False)
        
        print(f"   同比增速最快TOP 10:")
        for i, row in recent_fai.head(10).iterrows():
            print(f"     {i+1}. {row['product_name']}: 同比+{row['yoy']:.2f}%, 环比+{row['mom']:.2f}%")
        
        # 5. 投资扩张信号（连续3个月正增长）
        print(f"\n4. 投资扩张信号检测")
        for product_code in df_sorted['product_code'].unique():
            product_data = df_sorted[df_sorted['product_code'] == product_code].tail(6)
            
            if len(product_data) < 3:
                continue
            
            # 检查是否连续3个月正增长
            recent_mom = product_data['mom'].tail(3)
            if (recent_mom > 0).all():
                print(f"   🚀 {product_data['product_name'].iloc[-1]}: 连续3个月投资扩张！")
                print(f"      近3个月环比: {recent_mom.iloc[0]:.2f}% → {recent_mom.iloc[1]:.2f}% → {recent_mom.iloc[2]:.2f}%")
        
        return {'df_with_metrics': df_sorted, 'recent_fai': recent_fai}
    
    def analyze_price_trends(self, df: pd.DataFrame) -> dict:
        """
        分析价格指数趋势
        
        Args:
            df: 价格指数数据
        
        Returns:
            dict: 分析结果
        """
        if len(df) == 0:
            return {}
        
        print("\n" + "=" * 80)
        print("价格指数趋势分析")
        print("=" * 80)
        
        # 1. 数据覆盖情况
        print(f"\n1. 数据覆盖情况")
        print(f"   总记录数: {len(df)}")
        print(f"   价格种类: {df['product_code'].nunique()}")
        print(f"   时间范围: {df['date'].min()} ~ {df['date'].max()}")
        
        # 2. 按价格种类分析
        print(f"\n2. 各价格种类数据覆盖")
        price_coverage = df.groupby('product_code').agg({
            'product_name': 'first',
            'value': 'count',
            'date': ['min', 'max']
        }).reset_index()
        price_coverage.columns = ['product_code', 'product_name', 'count', 'start_date', 'end_date']
        price_coverage = price_coverage.sort_values('count', ascending=False)
        
        print(f"   TOP 15 价格种类:")
        for i, row in price_coverage.head(15).iterrows():
            print(f"     {i+1}. {row['product_name']}")
            print(f"        数据点: {row['count']}个")
        
        # 3. 转换为增长率（价格指数基数是100）
        df_sorted = df.sort_values(['product_code', 'date']).reset_index(drop=True)
        df_sorted['yoy'] = df_sorted.groupby('product_code')['value'].pct_change(periods=12) * 100
        df_sorted['mom'] = df_sorted.groupby('product_code')['value'].pct_change() * 100
        
        # 4. 通胀/通缩分析
        print(f"\n3. 通胀/通缩分析")
        recent_date = df_sorted['date'].max()
        recent_6m = df_sorted[df_sorted['date'] >= recent_date - pd.Timedelta(days=180)]
        
        recent_price = recent_6m.groupby('product_code').agg({
            'product_name': 'first',
            'value': 'last',
            'yoy': 'last',
            'mom': 'last'
        }).reset_index()
        recent_price = recent_price.dropna(subset=['yoy'])
        
        # 通缩（同比<0）
        deflation = recent_price[recent_price['yoy'] < 0].sort_values('yoy')
        print(f"   通缩（同比<0）领域（按严重程度排序）:")
        for i, row in deflation.head(10).iterrows():
            print(f"     {i+1}. {row['product_name']}: 同比{row['yoy']:.2f}%, 指数{row['value']:.1f}")
        
        # 通胀（同比>0）
        inflation = recent_price[recent_price['yoy'] > 0].sort_values('yoy', ascending=False)
        print(f"\n   通胀（同比>0）领域（按程度排序）:")
        for i, row in inflation.head(10).iterrows():
            print(f"     {i+1}. {row['product_name']}: 同比+{row['yoy']:.2f}%, 指数{row['value']:.1f}")
        
        # 5. 价格拐点检测
        print(f"\n4. 价格拐点检测")
        for product_code in df_sorted['product_code'].unique():
            product_data = df_sorted[df_sorted['product_code'] == product_code].tail(6)
            
            if len(product_data) < 3:
                continue
            
            # 检查由通缩转通胀
            if (product_data['yoy'].iloc[-3] < 0 and 
                product_data['yoy'].iloc[-2] > 0 and 
                product_data['yoy'].iloc[-1] > 0):
                print(f"   📈 {product_data['product_name'].iloc[-1]}: 由通缩转通胀！")
                print(f"      近3个月同比: {product_data['yoy'].iloc[-3]:.2f}% → {product_data['yoy'].iloc[-2]:.2f}% → {product_data['yoy'].iloc[-1]:.2f}%")
            
            # 检查由通胀转通缩
            if (product_data['yoy'].iloc[-3] > 0 and 
                product_data['yoy'].iloc[-2] < 0 and 
                product_data['yoy'].iloc[-1] < 0):
                print(f"   📉 {product_data['product_name'].iloc[-1]}: 由通胀转通缩！")
                print(f"      近3个月同比: {product_data['yoy'].iloc[-3]:.2f}% → {product_data['yoy'].iloc[-2]:.2f}% → {product_data['yoy'].iloc[-1]:.2f}%")
        
        return {'df_with_metrics': df_sorted, 'recent_price': recent_price}
    
    def comprehensive_analysis(self):
        """综合分析所有NBS数据"""
        print("=" * 80)
        print("NBS数据综合分析")
        print("=" * 80)
        
        results = {}
        
        # 1. 分析工业品产量
        output_file = os.path.join(self.data_dir, 'A020901_工业品产量.json')
        if os.path.exists(output_file):
            output_df = self.parse_nbs_json(output_file)
            if len(output_df) > 0:
                results['output'] = self.analyze_output_trends(output_df)
        
        # 2. 分析固定资产投资
        fai_file = os.path.join(self.data_dir, 'A0403_固定资产投资.json')
        if os.path.exists(fai_file):
            fai_df = self.parse_nbs_json(fai_file)
            if len(fai_df) > 0:
                results['fai'] = self.analyze_fai_trends(fai_df)
        
        # 3. 分析价格指数
        price_file = os.path.join(self.data_dir, 'A010D02_价格指数.json')
        if os.path.exists(price_file):
            price_df = self.parse_nbs_json(price_file)
            if len(price_df) > 0:
                results['price'] = self.analyze_price_trends(price_df)
        
        # 4. 综合发现
        print("\n" + "=" * 80)
        print("综合发现总结")
        print("=" * 80)
        
        print("\n📊 NBS数据分析结论:")
        print("1. 工业品产量数据：反映实体经济活跃度")
        print("2. 固定资产投资数据：反映资本开支和产能扩张意愿") 
        print("3. 价格指数数据：反映通胀/通缩压力")
        print("\n💡 投资启示:")
        print("- 产量+投资双增长：行业景气度上行")
        print("- 产量增长+价格上升：量价齐升，最佳投资机会")
        print("- 价格由负转正：通缩缓解，关注拐点机会")
        print("- 投资连续扩张：产能释放，关注供需变化")
        
        return results


def main():
    """主函数"""
    analyzer = NBSDataAnalyzer()
    results = analyzer.comprehensive_analysis()
    
    # 保存详细分析结果
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    for data_type, data_dict in results.items():
        if 'df_with_metrics' in data_dict:
            output_file = os.path.join(output_dir, f'nbs_{data_type}_analysis.parquet')
            data_dict['df_with_metrics'].to_parquet(output_file, index=False)
            print(f"\n{data_type}分析结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
