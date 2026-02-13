#!/usr/bin/env python3
"""
下载沪深300成分股的资金流向数据（moneyflow）
"""
import os
import sys
import time
import pandas as pd
import tushare as ts
from datetime import datetime

# 配置
TUSHARE_TOKEN = "2bcc0e9feb650d9862330a9743e5cc2e6469433c4d1ea0ce2d79371e"
DATA_DIR = "data/tushare"
OUTPUT_DIR = os.path.join(DATA_DIR, "moneyflow")
START_YEAR = 2020
END_YEAR = 2026
API_DELAY = 0.3  # API调用间隔（秒）

# 初始化Tushare
pro = ts.pro_api(TUSHARE_TOKEN)

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_hs300_stocks():
    """获取沪深300成分股列表"""
    print("=" * 80)
    print("获取沪深300成分股列表...")
    print("=" * 80)
    
    # 读取成分股数据
    constituents_file = os.path.join(DATA_DIR, "constituents", "hs300_constituents_all.parquet")
    df = pd.read_parquet(constituents_file)
    
    # 获取唯一股票代码
    stocks = df['con_code'].unique().tolist()
    
    print(f"✓ 获取到 {len(stocks)} 只沪深300成分股")
    print(f"  示例: {stocks[:5]}")
    
    return stocks


def download_moneyflow_for_stock(ts_code, start_date, end_date):
    """
    下载单只股票的资金流向数据
    
    Args:
        ts_code: 股票代码
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
    
    Returns:
        DataFrame or None
    """
    try:
        df = pro.moneyflow(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is not None and not df.empty:
            return df
        else:
            return None
            
    except Exception as e:
        print(f"  ✗ {ts_code} 下载失败: {e}")
        return None


def download_all_moneyflow():
    """下载所有沪深300成分股的资金流向数据"""
    print("\n" + "=" * 80)
    print("开始下载资金流向数据")
    print("=" * 80)
    
    # 获取成分股列表
    stocks = get_hs300_stocks()
    
    # 设置日期范围
    start_date = f"{START_YEAR}0101"
    end_date = f"{END_YEAR}1231"
    
    print(f"\n下载范围: {start_date} - {end_date}")
    print(f"股票数量: {len(stocks)}")
    print(f"预计时间: {len(stocks) * API_DELAY / 60:.1f} 分钟")
    
    # 下载每只股票的数据
    all_data = []
    success_count = 0
    fail_count = 0
    
    for i, stock in enumerate(stocks, 1):
        print(f"\n进度: {i}/{len(stocks)} - {stock}")
        
        df = download_moneyflow_for_stock(stock, start_date, end_date)
        
        if df is not None:
            print(f"  ✓ 下载了 {len(df)} 条记录")
            all_data.append(df)
            success_count += 1
        else:
            fail_count += 1
        
        # API调用间隔
        time.sleep(API_DELAY)
    
    # 合并所有数据
    print("\n" + "=" * 80)
    print("合并数据...")
    print("=" * 80)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"✓ 合并完成: {len(combined_df)} 条记录")
        
        # 按股票分组保存
        print("\n" + "=" * 80)
        print("按股票分组保存...")
        print("=" * 80)
        
        for stock in stocks:
            stock_data = combined_df[combined_df['ts_code'] == stock]
            
            if not stock_data.empty:
                output_path = os.path.join(OUTPUT_DIR, f"{stock}.parquet")
                stock_data.to_parquet(output_path)
                print(f"  ✓ {stock}: {len(stock_data)} 条记录")
        
        # 保存完整数据
        all_output_path = os.path.join(OUTPUT_DIR, "moneyflow_all.parquet")
        combined_df.to_parquet(all_output_path)
        print(f"\n✓ 完整数据已保存: {all_output_path}")
        
        # 统计信息
        print("\n" + "=" * 80)
        print("下载统计")
        print("=" * 80)
        print(f"成功: {success_count}/{len(stocks)}")
        print(f"失败: {fail_count}/{len(stocks)}")
        print(f"总记录数: {len(combined_df)}")
        print(f"日期范围: {combined_df['trade_date'].min()} - {combined_df['trade_date'].max()}")
        print(f"唯一股票: {combined_df['ts_code'].nunique()}")
        
        return combined_df
    else:
        print("✗ 没有成功下载任何数据")
        return None


def main():
    """主函数"""
    print("沪深300成分股资金流向数据下载")
    print("=" * 80)
    
    # 下载数据
    result_df = download_all_moneyflow()
    
    if result_df is not None:
        print("\n" + "=" * 80)
        print("下载完成！")
        print("=" * 80)
        print(f"数据保存在: {OUTPUT_DIR}")
        
        # 显示列名
        print(f"\n数据字段:")
        for col in result_df.columns:
            print(f"  - {col}")


if __name__ == "__main__":
    main()