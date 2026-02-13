#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用聚宽API获取申万行业估值数据

功能：
1. 获取申万一级行业的PE、PB、换手率等估值数据
2. 计算估值分位数
3. 保存为parquet格式
"""

import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
import time

try:
    import jqdatasdk as jq
    JQDATA_AVAILABLE = True
except ImportError:
    JQDATA_AVAILABLE = False
    print("警告: jqdatasdk未安装，无法使用聚宽数据")
    print("请运行: pip install jqdatasdk")


def load_config():
    """加载配置"""
    config_path = 'config/joinquant.json'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def auth_jqdata():
    """认证聚宽数据"""
    if not JQDATA_AVAILABLE:
        raise ImportError("jqdatasdk未安装")
    
    config = load_config()
    
    if not config.get('password'):
        raise ValueError("请先在config/joinquant.json中设置密码")
    
    try:
        jq.auth(config['username'], config['password'])
        print("聚宽认证成功")
        return True
    except Exception as e:
        print(f"聚宽认证失败: {e}")
        return False


def get_sw_valuation_data(start_date, end_date):
    """
    获取申万行业估值数据
    
    Args:
        start_date: 开始日期（格式：'YYYY-MM-DD'）
        end_date: 结束日期（格式：'YYYY-MM-DD'）
    
    Returns:
        DataFrame: 申万行业估值数据
    """
    if not JQDATA_AVAILABLE:
        raise ImportError("jqdatasdk未安装")
    
    from jqdatasdk import finance
    
    print(f"获取申万行业估值数据: {start_date} ~ {end_date}")
    
    # 获取所有申万一级行业
    q = jq.query(
        finance.SW1_DAILY_VALUATION.code,
        finance.SW1_DAILY_VALUATION.name,
        finance.SW1_DAILY_VALUATION.date,
        finance.SW1_DAILY_VALUATION.turnover_ratio,
        finance.SW1_DAILY_VALUATION.pe,
        finance.SW1_DAILY_VALUATION.pb,
        finance.SW1_DAILY_VALUATION.average_price,
        finance.SW1_DAILY_VALUATION.money_ratio,
        finance.SW1_DAILY_VALUATION.circulating_market_cap,
        finance.SW1_DAILY_VALUATION.average_circulating_market_cap,
        finance.SW1_DAILY_VALUATION.dividend_ratio
    ).filter(
        finance.SW1_DAILY_VALUATION.date >= start_date,
        finance.SW1_DAILY_VALUATION.date < end_date
    )
    
    df = finance.run_query(q)
    print(f"  获取到 {len(df)} 条记录")
    
    return df


def calculate_percentile(df, period_days=252*3):
    """
    计算估值分位数
    
    Args:
        df: 估值数据DataFrame
        period_days: 计算分位数的历史天数（默认3年）
    
    Returns:
        DataFrame: 包含分位数的数据
    """
    print(f"计算估值分位数（{period_days//252}年历史）")
    
    # 按行业分组计算分位数
    df = df.sort_values(['code', 'date'])
    
    result_list = []
    for code, group in df.groupby('code'):
        group = group.sort_values('date')
        
        # 计算滚动分位数
        for i in range(len(group)):
            current_date = group.iloc[i]['date']
            
            # 获取历史数据
            historical_start = current_date - timedelta(days=period_days)
            historical_data = group[
                (group['date'] >= historical_start) & 
                (group['date'] < current_date)
            ]
            
            if len(historical_data) < 60:  # 至少需要60个交易日
                pb_percentile = np.nan
                pe_percentile = np.nan
            else:
                pb_percentile = (historical_data['pb'] < group.iloc[i]['pb']).sum() / len(historical_data) * 100
                pe_percentile = (historical_data['pe'] < group.iloc[i]['pe']).sum() / len(historical_data) * 100
            
            result_list.append({
                'code': code,
                'date': current_date,
                'pb_percentile': pb_percentile,
                'pe_percentile': pe_percentile
            })
    
    percentile_df = pd.DataFrame(result_list)
    print(f"  计算完成")
    
    return percentile_df


def merge_industry_name(df):
    """
    合并行业名称
    
    Args:
        df: 数据DataFrame
    
    Returns:
        DataFrame: 包含行业名称的数据
    """
    # 获取行业名称映射
    industry_names = df[['code', 'name']].drop_duplicates().set_index('code')['name'].to_dict()
    
    # 添加行业名称列
    df['industry_name'] = df['code'].map(industry_names)
    
    return df


def save_data(df, output_path):
    """
    保存数据
    
    Args:
        df: 数据DataFrame
        output_path: 输出文件路径
    """
    df.to_parquet(output_path, index=False)
    print(f"数据已保存到: {output_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("获取申万行业估值数据（聚宽）")
    print("=" * 80)
    
    # 1. 认证聚宽数据
    if not auth_jqdata():
        return
    
    # 2. 设置日期范围（默认获取最近3年数据）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=252*3)).strftime('%Y-%m-%d')
    
    print(f"\n日期范围: {start_date} ~ {end_date}")
    
    # 3. 获取估值数据
    try:
        df = get_sw_valuation_data(start_date, end_date)
        
        if df.empty:
            print("未获取到数据")
            return
        
        # 4. 计算估值分位数
        percentile_df = calculate_percentile(df, period_days=252*3)
        
        # 5. 合并数据
        df = df.merge(percentile_df, on=['code', 'date'], how='left')
        
        # 6. 添加行业名称
        df = merge_industry_name(df)
        
        # 7. 保存数据
        output_path = 'data/tushare/macro/sw_valuation.parquet'
        save_data(df, output_path)
        
        # 8. 显示统计信息
        print("\n数据统计:")
        print(f"  记录数: {len(df)}")
        print(f"  行业数: {df['code'].nunique()}")
        print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
        print()
        print(f"  PE统计: {df['pe'].describe().to_string()}")
        print()
        print(f"  PB统计: {df['pb'].describe().to_string()}")
        print()
        print(f"  PB分位数统计: {df['pb_percentile'].describe().to_string()}")
        print()
        print(f"  PE分位数统计: {df['pe_percentile'].describe().to_string()}")
        
        # 9. 显示最新数据
        print("\n最新数据（2025年12月）:")
        latest_data = df[df['date'] == df['date'].max()].sort_values('pb_percentile')
        print(latest_data[['industry_name', 'pe', 'pb', 'pb_percentile', 'pe_percentile']].head(10).to_string(index=False))
        
    except Exception as e:
        print(f"获取数据失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()