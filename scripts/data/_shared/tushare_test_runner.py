from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd

from scripts.data._shared.tushare_helpers import get_pro


def test_dc_api():
    pro = get_pro()
    print("=" * 80)
    print("测试Tushare概念板块接口用法")
    print("=" * 80)

    print("\n=== dc_index (概念指数) ===")
    try:
        df = pro.dc_index()
        print(f"不带参数记录数: {len(df)}")
        print(f"字段: {df.columns.tolist()}")
        print(f"唯一概念数: {df['ts_code'].nunique()}")
        print(f"日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
        print(f"\n数据预览:")
        print(df.head(3).to_string(index=False))
        time.sleep(40)
    except Exception as e:
        print(f"错误: {e}")

    print("\n=== dc_member (概念成分股) ===")
    try:
        df = pro.dc_member()
        print(f"不带参数记录数: {len(df)}")
        print(f"字段: {df.columns.tolist()}")
        print(f"概念数量: {df['ts_code'].nunique()}")
        print(f"成分股数量: {df['con_code'].nunique()}")
        print(f"日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
        print(f"\n数据预览:")
        print(df.head(3).to_string(index=False))
        time.sleep(40)
    except Exception as e:
        print(f"错误: {e}")

    print("\n=== dc_daily (概念成分股日线) ===")
    try:
        df = pro.dc_daily()
        print(f"不带参数记录数: {len(df)}")
        print(f"字段: {df.columns.tolist()}")
        if len(df) > 0:
            print(f"概念数量: {df['ts_code'].nunique()}")
            print(f"成分股数量: {df['con_code'].nunique()}")
            print(f"日期范围: {df['trade_date'].min()} {df['trade_date'].max()}")
            print(f"\n数据预览:")
            print(df.head(3).to_string(index=False))
    except Exception as e:
        print(f"错误: {e}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


def test_sw_daily():
    pro = get_pro()
    print("=" * 80)
    print("测试Tushare sw_daily接口")
    print("=" * 80)

    print("\n1. 获取申万行业指数列表...")
    try:
        index_df = pro.index_classify(level="L1", src="SW2021")
        print(f"  获取到 {len(index_df)} 个申万一级行业")
        print(f"  前5个行业:")
        print(index_df.head())
    except Exception as e:
        print(f"  获取失败: {e}")
        return

    print("\n2. 测试sw_daily接口...")
    first_index = index_df.iloc[0]["index_code"]
    print(f"  测试指数: {first_index}")

    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        df = pro.sw_daily(ts_code=first_index, start_date=start_date, end_date=end_date)
        print(f"  获取到 {len(df)} 条记录")
        print(f"  字段列表: {df.columns.tolist()}")
        print(f"  前5行数据:")
        print(df.head())

        print(f"\n3. 检查是否包含估值数据...")
        print(f"  {'✅' if 'pe' in df.columns else '❌'} 包含PE数据")
        print(f"  {'✅' if 'pb' in df.columns else '❌'} 包含PB数据")
        print(f"  {'✅' if 'turnover_rate' in df.columns else '❌'} 包含换手率数据")
    except Exception as e:
        print(f"  获取失败: {e}")
        import traceback
        traceback.print_exc()


def test_one_day_one_industry():
    pro = get_pro()
    print("=" * 80)
    print("测试Tushare sw_daily接口")
    print("=" * 80)

    print("\n测试1个行业1个交易日:")
    ts_code = "801010.SI"
    date = "20260210"

    try:
        df = pro.sw_daily(ts_code=ts_code, start_date=date, end_date=date)
        print(f"  指数: {ts_code}")
        print(f"  日期: {date}")
        print(f"  获取到 {len(df)} 条记录")
        print(f"  字段: {df.columns.tolist()}")
        print()
        print(f"  数据:")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"  获取失败: {e}")


def test_one_industry_one_month():
    pro = get_pro()
    print("=" * 80)
    print("测试Tushare sw_daily接口")
    print("=" * 80)

    print("\n测试1个行业1个月的数据:")
    ts_code = "801010.SI"
    start_date = "20260101"
    end_date = "20260131"

    try:
        df = pro.sw_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        print(f"  指数: {ts_code}")
        print(f"  日期范围: {start_date} ~ {end_date}")
        print(f"  获取到 {len(df)} 条记录")
        print(f"  字段: {df.columns.tolist()}")
        print()
        print(f"  数据预览:")
        print(df.head(10).to_string(index=False))
    except Exception as e:
        print(f"  获取失败: {e}")


def test_all_industries_one_day():
    pro = get_pro()
    print("=" * 80)
    print("测试Tushare sw_daily接口")
    print("=" * 80)

    print("\n获取申万行业指数列表...")
    indices = pro.index_classify(level="L1", src="SW2021")
    print(f"  获取到 {len(indices)} 个申万一级行业")

    print(f"\n测试获取所有行业2026-02-10的数据...")
    date = "20260210"

    all_data = []
    for i, row in indices.iterrows():
        if (i + 1) % 10 == 0:
            print(f"  进度: {i + 1}/{len(indices)}")
        try:
            df = pro.sw_daily(ts_code=row["index_code"], start_date=date, end_date=date)
            if not df.empty:
                all_data.append(df)
            time.sleep(0.5)
        except Exception as e:
            print(f"  获取 {row['index_code']} 失败: {e}")

    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        print(f"\n总记录数: {len(result_df)}")
        print(f"字段: {result_df.columns.tolist()}")
        print(f"\n数据预览:")
        print(result_df.to_string(index=False))
    else:
        print("\n未获取到数据")


def test_concept_one_day():
    pro = get_pro()
    print("=" * 80)
    print("测试概念板块接口单日数据量")
    print("=" * 80)

    test_date = "20260210"
    print(f"\n测试日期: {test_date}")

    print("\n=== dc_index (概念指数) ===")
    try:
        offset = 0
        total_index = 0
        page = 1
        while True:
            df = pro.dc_index(trade_date=test_date, offset=offset)
            if df is None or df.empty:
                break
            total_index += len(df)
            print(f"  第{page}页: {len(df)} 条记录")
            offset += len(df)
            page += 1
            if len(df) < 5000:
                break
            time.sleep(40)
        print(f"  总计: {total_index} 条记录")
    except Exception as e:
        print(f"  错误: {e}")

    print("\n=== dc_member (概念成分股) ===")
    print("只获取第一页估算数据量...")
    try:
        df = pro.dc_member(trade_date=test_date, offset=0)
        print(f"  第1页: {len(df)} 条记录")
        if len(df) > 0:
            print(f"  唯一概念数: {df['ts_code'].nunique()}")
            print(f"  唯一成分股数: {df['con_code'].nunique()}")
            if len(df) >= 5000:
                estimated_pages = len(df) // 5000 + 1
                estimated_total = len(df) * estimated_pages
                print(f"  估算总记录数: {estimated_total:,} 条")
            else:
                print(f"  总记录数: {len(df):,} 条")
    except Exception as e:
        print(f"  错误: {e}")

    print("\n" + "=" * 80)
