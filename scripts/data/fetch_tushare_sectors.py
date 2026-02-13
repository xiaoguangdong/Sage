#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从Tushare获取板块数据
包括行业板块和概念板块
"""

import tushare as ts
import pandas as pd
import os
from datetime import datetime
import time

class TushareSectorFetcher:
    def __init__(self, token):
        """初始化Tushare连接"""
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.data_dir = 'data/tushare/sectors'
        os.makedirs(self.data_dir, exist_ok=True)
        
    def fetch_index_classify(self, level='L1', source='SW'):
        """获取行业分类（申万/中信等）"""
        print(f"正在获取{source}{level}行业分类...")
        
        try:
            df = self.pro.index_classify(level=level, source=source)
            print(f"  ✓ 获取到 {len(df)} 个{source}{level}行业")
            
            # 保存
            filename = f"{source}_{level}_classify.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"  ✓ 已保存到 {filepath}")
            
            return df
        except Exception as e:
            print(f"  ✗ 获取失败: {e}")
            return None
    
    def fetch_index_members(self, index_code):
        """获取指数成分股"""
        print(f"正在获取指数 {index_code} 的成分股...")
        
        try:
            df = self.pro.index_member(index_code=index_code)
            print(f"  ✓ 获取到 {len(df)} 只成分股")
            return df
        except Exception as e:
            print(f"  ✗ 获取失败: {e}")
            return None
    
    def fetch_all_index_members(self, classify_df):
        """获取所有指数的成分股"""
        all_members = []
        
        for idx, row in classify_df.iterrows():
            index_code = row['index_code']
            index_name = row['industry_name']
            
            members = self.fetch_index_members(index_code)
            if members is not None:
                members['industry_name'] = index_name
                all_members.append(members)
            
            # 避免请求过快
            time.sleep(0.1)
        
        if all_members:
            result = pd.concat(all_members, ignore_index=True)
            filepath = os.path.join(self.data_dir, 'all_index_members.csv')
            result.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"  ✓ 所有成分股已保存到 {filepath}")
            return result
        else:
            return None
    
    def fetch_concepts(self):
        """获取概念列表"""
        print("正在获取概念列表...")
        
        try:
            df = self.pro.concept()
            print(f"  ✓ 获取到 {len(df)} 个概念")
            
            # 保存
            filepath = os.path.join(self.data_dir, 'concepts.csv')
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"  ✓ 已保存到 {filepath}")
            
            return df
        except Exception as e:
            print(f"  ✗ 获取失败: {e}")
            return None
    
    def fetch_concept_details(self, concept_id):
        """获取概念成分股"""
        print(f"正在获取概念 {concept_id} 的成分股...")
        
        try:
            df = self.pro.concept_detail(id=concept_id)
            print(f"  ✓ 获取到 {len(df)} 只成分股")
            return df
        except Exception as e:
            print(f"  ✗ 获取失败: {e}")
            return None
    
    def fetch_all_concept_details(self, concepts_df, max_concepts=100):
        """获取所有概念的成分股（限制数量避免请求过多）"""
        all_details = []
        
        # 只获取前max_concepts个概念
        concepts_subset = concepts_df.head(max_concepts)
        
        for idx, row in concepts_subset.iterrows():
            concept_id = row['id']
            concept_name = row['concept_name']
            
            print(f"  [{idx+1}/{len(concepts_subset)}] 获取 {concept_name}...")
            
            details = self.fetch_concept_details(concept_id)
            if details is not None:
                details['concept_name'] = concept_name
                all_details.append(details)
            
            # 避免请求过快
            time.sleep(0.1)
        
        if all_details:
            result = pd.concat(all_details, ignore_index=True)
            filepath = os.path.join(self.data_dir, 'all_concept_details.csv')
            result.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"\n  ✓ 所有概念成分股已保存到 {filepath}")
            return result
        else:
            return None
    
    def fetch_all(self):
        """获取所有板块数据"""
        print("=" * 70)
        print("开始获取Tushare板块数据")
        print("=" * 70)
        
        # 1. 获取申万一级行业
        print("\n步骤 1/4: 获取申万一级行业")
        sw_l1 = self.fetch_index_classify(level='L1', source='SW')
        if sw_l1 is not None:
            sw_l1_members = self.fetch_all_index_members(sw_l1)
        
        # 2. 获取申万二级行业
        print("\n步骤 2/4: 获取申万二级行业")
        sw_l2 = self.fetch_index_classify(level='L2', source='SW')
        if sw_l2 is not None:
            sw_l2_members = self.fetch_all_index_members(sw_l2)
        
        # 3. 获取概念列表
        print("\n步骤 3/4: 获取概念列表")
        concepts = self.fetch_concepts()
        
        # 4. 获取概念成分股（限制前100个）
        if concepts is not None:
            print("\n步骤 4/4: 获取概念成分股（前100个）")
            concept_details = self.fetch_all_concept_details(concepts, max_concepts=100)
        
        print("\n" + "=" * 70)
        print("✓ 板块数据获取完成！")
        print("=" * 70)

if __name__ == '__main__':
    # 你需要替换成自己的Tushare token
    TOKEN = 'YOUR_TUSHARE_TOKEN_HERE'
    
    fetcher = TushareSectorFetcher(TOKEN)
    fetcher.fetch_all()