#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
固定资产投资数据对齐到申万行业

基于NBS数据分析结果，将74个投资领域对齐到申万31个一级行业
"""

import pandas as pd
import json
import yaml
import os
from typing import Dict, List

from scripts.data.macro.paths import MACRO_DIR

class FAIToSWAligner:
    """固定资产投资数据对齐器"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化对齐器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir or str(MACRO_DIR)
        
        # 加载申万行业映射配置
        mapping_file = 'config/sw_nbs_mapping.yaml'
        with open(mapping_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.sw_to_nbs = config['sw_to_nbs']
        
        # 创建NBS行业名称到申万行业的反向映射
        self.nbs_to_sw = {}
        for sw_industry, nbs_list in self.sw_to_nbs.items():
            for nbs_item in nbs_list:
                nbs_name = nbs_item['nbs_industry']
                if nbs_name not in self.nbs_to_sw:
                    self.nbs_to_sw[nbs_name] = []
                self.nbs_to_sw[nbs_name].append({
                    'sw_industry': sw_industry,
                    'weight': nbs_item['weight']
                })
    
    def parse_fai_json(self, json_file: str) -> pd.DataFrame:
        """解析固定资产投资JSON数据"""
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
                    'fai_code': zb_code,
                    'fai_name': zb_code_to_name.get(zb_code, ''),
                    'time_code': sj_code,
                    'time_name': sj_code_to_name.get(sj_code, ''),
                    'fai_yoy': value
                })
        
        df = pd.DataFrame(records)
        
        # 转换时间
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['time_code'].astype(str), format='%Y%m')
        
        return df
    
    def match_fai_to_nbs(self, fai_name: str) -> List[Dict]:
        """
        将投资领域名称匹配到NBS行业
        
        Args:
            fai_name: 投资领域名称
        
        Returns:
            List[Dict]: 匹配的NBS行业列表
        """
        matches = []
        
        # 直接匹配
        if fai_name in self.nbs_to_sw:
            return self.nbs_to_sw[fai_name]
        
        # 模糊匹配：包含关键词
        fai_keywords = {
            '医药': '医药制造业',
            '化学': '化学原料和化学制品制造业',
            '化工': '化学原料和化学制品制造业',
            '钢铁': '黑色金属冶炼和压延加工业',
            '有色金属': '有色金属冶炼和压延加工业',
            '汽车': '汽车制造业',
            '计算机': '计算机、通信和其他电子设备制造业',
            '通信': '计算机、通信和其他电子设备制造业',
            '电子': '计算机、通信和其他电子设备制造业',
            '电气': '电气机械和器材制造业',
            '机械': '通用设备制造业',
            '专用设备': '专用设备制造业',
            '金属': '金属制品业',
            '石油': '石油、煤炭及其他燃料加工业',
            '煤炭': '石油开采业',
            '电力': '电力、热力生产和供应业',
            '建筑': '建筑业',
            '房地产': '房地产业',
            '食品': '食品制造业',
            '纺织': '纺织业',
            '造纸': '造纸和纸制品业',
            '医药': '医药制造业',
            '农副': '农副食品加工业',
            '饮料': '酒、饮料和精制茶制造业',
            '家具': '家具制造业',
            '印刷': '印刷和记录媒介复制业',
            '橡胶': '橡胶和塑料制品业',
            '塑料': '橡胶和塑料制品业',
            '非金属': '非金属矿物制品业',
            '废弃': '废弃资源综合利用业',
            '运输': '交通运输、仓储和邮政业',
            '铁路': '铁路运输业',
            '道路': '道路运输业',
            '航空': '航空运输业',
            '水务': '水的生产和供应业',
            '燃气': '燃气生产和供应业',
            '环保': '生态保护和环境治理业',
        }
        
        for keyword, nbs_name in fai_keywords.items():
            if keyword in fai_name and nbs_name in self.nbs_to_sw:
                matches.extend(self.nbs_to_sw[nbs_name])
                break
        
        return matches
    
    def align_fai_to_sw(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对齐固定资产投资数据到申万行业
        
        Args:
            df: 原始FAI数据
        
        Returns:
            DataFrame: 对齐后的数据
        """
        if len(df) == 0:
            return pd.DataFrame()
        
        print("\n" + "=" * 80)
        print("将固定资产投资数据对齐到申万行业")
        print("=" * 80)
        
        aligned_records = []
        unmatched_count = 0
        
        for idx, row in df.iterrows():
            fai_name = row['fai_name']
            fai_code = row['fai_code']
            
            # 匹配到申万行业
            matches = self.match_fai_to_nbs(fai_name)
            
            if not matches:
                unmatched_count += 1
                print(f"  ⚠️  未匹配: {fai_name}")
                continue
            
            # 创建匹配记录
            for match in matches:
                sw_industry = match['sw_industry']
                weight = match['weight']
                
                aligned_records.append({
                    'date': row['date'],
                    'fai_code': fai_code,
                    'fai_name': fai_name,
                    'sw_industry': sw_industry,
                    'weight': weight,
                    'fai_yoy': row['fai_yoy'],
                    'fai_yoy_weighted': row['fai_yoy'] * weight
                })
        
        aligned_df = pd.DataFrame(aligned_records)
        
        if len(aligned_df) > 0:
            print(f"\n  对齐结果:")
            print(f"    总记录数: {len(df)}")
            print(f"    对齐记录数: {len(aligned_df)}")
            print(f"    未匹配数: {unmatched_count}")
            print(f"    对齐率: {(len(df) - unmatched_count) / len(df) * 100:.1f}%")
            
            # 统计每个申万行业的数据点
            sw_coverage = aligned_df.groupby('sw_industry').agg({
                'fai_name': 'nunique',
                'date': 'nunique'
            }).reset_index()
            sw_coverage.columns = ['sw_industry', 'fai_sources', 'data_points']
            sw_coverage = sw_coverage.sort_values('data_points', ascending=False)
            
            print(f"\n  申万行业覆盖情况 (TOP 15):")
            for i, row in sw_coverage.head(15).iterrows():
                print(f"    {i+1}. {row['sw_industry']}: {row['fai_sources']}个FAI源, {row['data_points']}个数据点")
        
        return aligned_df
    
    def aggregate_to_sw_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        聚合到申万行业级别
        
        Args:
            df: 对齐后的数据
        
        Returns:
            DataFrame: 申万行业级别的FAI数据
        """
        if len(df) == 0:
            return pd.DataFrame()
        
        # 按申万行业和日期分组，计算加权平均
        sw_fai_df = df.groupby(['sw_industry', 'date']).agg({
            'fai_yoy_weighted': 'sum',
            'weight': 'sum'
        }).reset_index()
        
        # 计算实际增长率（加权平均）
        sw_fai_df['fai_yoy'] = sw_fai_df['fai_yoy_weighted'] / sw_fai_df['weight']
        
        # 计算环比增长率
        sw_fai_df = sw_fai_df.sort_values(['sw_industry', 'date']).reset_index(drop=True)
        sw_fai_df['fai_mom'] = sw_fai_df.groupby('sw_industry')['fai_yoy'].pct_change() * 100
        
        # 选择最终列
        result_df = sw_fai_df[['sw_industry', 'date', 'fai_yoy', 'fai_mom']]
        
        print(f"\n  聚合结果:")
        print(f"    申万行业数: {result_df['sw_industry'].nunique()}")
        print(f"    总记录数: {len(result_df)}")
        print(f"    时间范围: {result_df['date'].min()} ~ {result_df['date'].max()}")
        
        return result_df
    
    def detect_expansion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检测投资扩张信号
        
        Args:
            df: 申万行业FAI数据
        
        Returns:
            DataFrame: 包含扩张信号的数据
        """
        if len(df) == 0:
            return pd.DataFrame()
        
        print("\n" + "=" * 80)
        print("检测投资扩张信号")
        print("=" * 80)
        
        results = []
        
        for sw_industry in df['sw_industry'].unique():
            industry_data = df[df['sw_industry'] == sw_industry].sort_values('date').reset_index(drop=True)
            
            if len(industry_data) < 3:
                continue
            
            # 检查最近3个月
            recent_3m = industry_data.tail(3)
            
            # 检查是否连续3个月正增长
            if (recent_3m['fai_yoy'] > 0).all():
                signal = {
                    'sw_industry': sw_industry,
                    'latest_date': recent_3m['date'].iloc[-1],
                    'fai_yoy_trend': 'EXPANSION',
                    'fai_yoy_3m_avg': recent_3m['fai_yoy'].mean(),
                    'fai_yoy_latest': recent_3m['fai_yoy'].iloc[-1],
                    'fai_mom_trend': recent_3m['fai_mom'].iloc[-1]
                }
                results.append(signal)
                
                print(f"  🚀 {sw_industry}: 连续3个月投资扩张")
                print(f"     平均增速: {signal['fai_yoy_3m_avg']:.2f}%, 最新: {signal['fai_yoy_latest']:.2f}%")
        
        return pd.DataFrame(results)
    
    def process_all(self):
        """处理所有数据"""
        print("=" * 80)
        print("固定资产投资数据对齐处理")
        print("=" * 80)
        
        # 1. 解析FAI数据
        print("\n1. 解析固定资产投资数据...")
        fai_file = os.path.join(self.data_dir, 'A0403_固定资产投资.json')
        if not os.path.exists(fai_file):
            print(f"  错误: 文件不存在 {fai_file}")
            return None
        
        fai_df = self.parse_fai_json(fai_file)
        print(f"  原始数据: {len(fai_df)}条记录")
        print(f"  投资领域: {fai_df['fai_code'].nunique()}个")
        
        # 2. 对齐到申万行业
        print("\n2. 对齐到申万行业...")
        aligned_df = self.align_fai_to_sw(fai_df)
        
        if len(aligned_df) == 0:
            print("  错误: 对齐失败")
            return None
        
        # 3. 聚合到申万行业级别
        print("\n3. 聚合到申万行业级别...")
        sw_fai_df = self.aggregate_to_sw_level(aligned_df)
        
        # 4. 检测扩张信号
        print("\n4. 检测投资扩张信号...")
        expansion_signals = self.detect_expansion_signals(sw_fai_df)
        
        # 5. 保存结果
        print("\n5. 保存结果...")
        output_dir = 'data/processed'
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存对齐后的数据
        aligned_file = os.path.join(output_dir, 'fai_aligned_to_sw.parquet')
        aligned_df.to_parquet(aligned_file, index=False)
        print(f"  对齐数据已保存: {aligned_file}")
        
        # 保存聚合后的数据
        aggregated_file = os.path.join(output_dir, 'fai_sw_industry.parquet')
        sw_fai_df.to_parquet(aggregated_file, index=False)
        print(f"  聚合数据已保存: {aggregated_file}")
        
        # 保存扩张信号
        if len(expansion_signals) > 0:
            signals_file = os.path.join(output_dir, 'fai_expansion_signals.parquet')
            expansion_signals.to_parquet(signals_file, index=False)
            print(f"  扩张信号已保存: {signals_file}")
        
        print("\n" + "=" * 80)
        print("处理完成")
        print("=" * 80)
        
        return {
            'aligned': aligned_df,
            'aggregated': sw_fai_df,
            'signals': expansion_signals
        }


def main():
    """主函数"""
    aligner = FAIToSWAligner()
    results = aligner.process_all()
    
    if results:
        print("\n处理结果:")
        print(f"  对齐记录: {len(results['aligned'])}条")
        print(f"  聚合记录: {len(results['aggregated'])}条")
        print(f"  扩张信号: {len(results['signals'])}个")


if __name__ == '__main__':
    main()
