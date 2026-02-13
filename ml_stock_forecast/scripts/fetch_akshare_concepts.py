#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 akshare 获取概念板块数据（强制 IPv4）
解决 IPv6 连接不稳定的问题
"""

import socket
import pandas as pd
import requests
import time
from typing import Optional

# 强制使用 IPv4 的解决方案
old_getaddrinfo = socket.getaddrinfo

def force_ipv4_getaddrinfo(*args, **kwargs):
    """强制使用 IPv4 的 getaddrinfo"""
    res = old_getaddrinfo(*args, **kwargs)
    return [r for r in res if r[0] == socket.AF_INET]

def enable_ipv4_only():
    """启用 IPv4 模式"""
    socket.getaddrinfo = force_ipv4_getaddrinfo

def disable_ipv4_only():
    """禁用 IPv4 模式"""
    socket.getaddrinfo = old_getaddrinfo

def fetch_concept_list_direct() -> Optional[pd.DataFrame]:
    """直接从东方财富 API 获取概念板块列表"""
    base_url = 'https://79.push2.eastmoney.com/api/qt/clist/get'
    params = {
        'pn': 1,
        'pz': 500,  # 每页500条
        'po': 1,
        'np': 1,
        'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
        'fltt': 2,
        'invt': 2,
        'fid': 'f3',
        'fs': 'm:90+t:3+f:!2150',  # 概念板块
        'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152'
    }
    
    enable_ipv4_only()
    try:
        for attempt in range(3):
            try:
                response = requests.get(base_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and 'diff' in data['data']:
                        items = data['data']['diff']
                        # 转换为 DataFrame
                        df = pd.DataFrame(items)
                        # 重命名列
                        df = df.rename(columns={
                            'f12': '板块代码',
                            'f14': '板块名称',
                            'f3': '涨跌幅',
                            'f2': '最新价',
                            'f4': '涨跌额'
                        })
                        return df[['板块代码', '板块名称', '最新价', '涨跌幅', '涨跌额']]
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise
    finally:
        disable_ipv4_only()
    
    return None

def fetch_concept_history_direct(concept_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """直接从东方财富 API 获取概念板块历史K线数据"""
    # 东方财富历史数据接口
    # 格式: https://push2his.eastmoney.com/api/qt/stock/kline/get?...
    base_url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    params = {
        'secid': f'90.{concept_code}',  # 90表示概念板块
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101',  # 日线
        'fqt': '0',    # 不复权
        'beg': start_date.replace('-', ''),
        'end': end_date.replace('-', ''),
        'end': '20500101'
    }
    
    enable_ipv4_only()
    try:
        response = requests.get(base_url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'klines' in data['data']:
                klines = data['data']['klines']
                # 解析K线数据
                df = pd.DataFrame([line.split(',') for line in klines])
                df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                df['板块代码'] = concept_code
                return df
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    finally:
        disable_ipv4_only()
    
    return None

if __name__ == '__main__':
    # 测试获取概念列表
    print("测试获取概念板块列表...")
    concept_list = fetch_concept_list_direct()
    if concept_list is not None:
        print(f"✓ 成功获取 {len(concept_list)} 个概念板块")
        print("\n前10个概念:")
        print(concept_list.head(10))
        
        # 测试获取历史数据
        print("\n测试获取锂电池概念历史数据...")
        lithium = concept_list[concept_list['板块名称'].str.contains('锂')].iloc[0]
        print(f"找到: {lithium['板块名称']} ({lithium['板块代码']})")
        
        history = fetch_concept_history_direct(
            lithium['板块代码'],
            '2024-09-01',
            '2024-10-10'
        )
        if history is not None:
            print(f"✓ 成功获取 {len(history)} 条历史数据")
            print(history.head())
    else:
        print("✗ 获取失败")