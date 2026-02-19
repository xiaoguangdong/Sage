#!/usr/bin/env python3
"""
复核市场状态标签规则（针对沪深300指数）
分析文档中的12状态和3状态定义，并评估其可操作性
"""

import numpy as np
import pandas as pd

from scripts.data._shared.runtime import get_tushare_root

# 读取沪深300指数数据
print("=" * 80)
print("加载沪深300指数数据")
print("=" * 80)

df = pd.read_parquet(str(get_tushare_root() / "index" / "index_000300_SH_ohlc.parquet"))
print(f"数据形状: {df.shape}")
print(f"时间范围: {df['date'].min()} 至 {df['date'].max()}")
print(f"列名: {df.columns.tolist()}")

# 按日期排序
df = df.sort_values("date").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])

print("\n" + "=" * 80)
print("计算技术指标")
print("=" * 80)

# 计算移动平均
df["ma20"] = df["close"].rolling(window=20).mean()
df["ma60"] = df["close"].rolling(window=60).mean()
df["ma250"] = df["close"].rolling(window=250).mean()

# 计算MACD
df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
df["dif"] = df["ema12"] - df["ema26"]
df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
df["macd"] = 2 * (df["dif"] - df["dea"])

# 计算RSI
delta = df["close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df["rsi"] = 100 - (100 / (1 + rs))

# 计算布林带
df["bb_middle"] = df["close"].rolling(window=20).mean()
df["bb_std"] = df["close"].rolling(window=20).std()
df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]

# 计算ATR
high_low = df["high"] - df["low"]
high_close = np.abs(df["high"] - df["close"].shift())
low_close = np.abs(df["low"] - df["close"].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df["atr"] = tr.rolling(window=14).mean()

# 计算成交量MA
df["vol_ma20"] = df["vol"].rolling(window=20).mean()
df["vol_ma250"] = df["vol"].rolling(window=250).mean()

print("✓ 技术指标计算完成")

print("\n" + "=" * 80)
print("三状态分类规则复核")
print("=" * 80)

print("\n【牛市】")
print("必要条件:")
print("  1. MA20 > MA60 > MA250（多头排列）")
print("  2. 价格在MA250上方")
print("综合标准（满足任一）:")
print("  方案A: Total_Score > 0.6 且持续10个交易日")
print("  方案B: 满足以下至少3条:")
print("    a. Total_Score > 0.5")
print("    b. 上涨家数比例 > 65%  ⚠️ 不适用于指数，需要替代指标")
print("    c. 成交量突破250日均量")
print("    d. RSI在50-70区间")
print("\n问题:")
print("  - Total_Score的计算方式未明确定义")
print("  - '上涨家数比例'不适用于单一指数")
print("  - 需要确认Total_Score的构成和权重")

print("\n【熊市】")
print("必要条件:")
print("  1. MA20 < MA60 < MA250（空头排列）")
print("  2. 价格在MA250下方")
print("综合标准（满足任一）:")
print("  方案A: Total_Score < -0.6 且持续10个交易日")
print("  方案B: 满足以下至少3条:")
print("    a. Total_Score < -0.5")
print("    b. 上涨家数比例 < 35%  ⚠️ 不适用于指数")
print("    c. 成交量萎缩至均量80%以下")
print("    d. RSI在30以下")
print("\n问题:")
print("  - '上涨家数比例'不适用于单一指数")
print("  - '所有行业板块下跌'不适用于指数")

print("\n【震荡整理】")
print("必要条件:")
print("  1. MA20、MA60、MA250交织缠绕")
print("  2. 价格在MA250±10%范围内")
print("综合标准（满足任一）:")
print("  方案A: -0.3 < Total_Score < 0.3 且持续15个交易日")
print("  方案B: 满足以下特征:")
print("    a. 布林带宽度(上轨-下轨)/中轨 < 10%")
print("    b. ATR14/收盘价 < 1.5%")
print("    c. 上涨家数比例在40%-60%之间  ⚠️ 不适用于指数")
print("\n问题:")
print("  - '上涨家数比例'不适用于单一指数")
print("  - '交织缠绕'的定义不够清晰")

print("\n" + "=" * 80)
print("十二状态分类规则复核")
print("=" * 80)

print("\n【牛市阶段细分】")
print("1. 初始牛市")
print("   条件:")
print("   - MA20上穿MA60且MA60 > MA250")
print("   - Total_Score从负转正突破0.3")
print("   - 成交量放大至MA20(Vol)×1.5")
print("   - RSI从30-50区间突破55")
print("   ✓ 所有条件都适用于指数")
print("   问题: Total_Score定义不明确")

print("\n2. 确认牛市")
print("   条件:")
print("   - MA20 > MA60 > MA250")
print("   - Total_Score > 0.6 持续10天")
print("   - 上涨家数比例 > 70% 持续5天  ⚠️ 不适用")
print("   - 布林带向上开口")
print("   问题: '上涨家数比例'需要替代指标")

print("\n3. 晚期牛市/泡沫期")
print("   条件:")
print("   - MA斜率 > 45度")
print("   - RSI > 75 持续5天")
print("   - 成交量异常放大(>MA20×2)")
print("   - 散户参与度指标 > 80%  ⚠️ 不适用")
print("   - 估值分位数 > 90%  ⚠️ 需要额外数据")
print("   问题: 需要估值数据，散户参与度不适用")

print("\n4. 牛市调整")
print("   条件:")
print("   - 牛市中出现Total_Score回落至0.2-0.4")
print("   - 回调幅度8-15%")
print("   - MA20仍 > MA60")
print("   - 成交量萎缩至MA20×0.8")
print("   ✓ 大部分条件适用")

print("\n【熊市阶段细分】")
print("1. 初始熊市")
print("   条件:")
print("   - MA20下穿MA60且MA60 < MA250")
print("   - Total_Score从正转负跌破-0.3")
print("   - 下跌放量，上涨缩量")
print("   - RSI从50-70跌破45")
print("   ✓ 所有条件都适用于指数")

print("\n2. 主跌熊市")
print("   条件:")
print("   - MA20 < MA60 < MA250")
print("   - Total_Score < -0.6 持续10天")
print("   - 上涨家数比例 < 30%  ⚠️ 不适用")
print("   - 所有行业板块下跌  ⚠️ 不适用")
print("   - 恐慌指数VIX > 25  ⚠️ 需要VIX数据")
print("   问题: 多项条件不适用于指数")

print("\n3. 晚期熊市/寻底期")
print("   条件:")
print("   - 下跌速度放缓")
print("   - 成交量极度萎缩(<MA20×0.5)")
print("   - RSI < 25 但开始走平")
print("   - 出现底背离信号")
print("   - 估值分位数 < 10%  ⚠️ 需要估值数据")
print("   问题: 需要估值数据")

print("\n4. 熊市反弹")
print("   条件:")
print("   - 熊市中Total_Score反弹至-0.2到0之间")
print("   - 反弹幅度10-20%")
print("   - 成交量温和放大")
print("   - MA20仍 < MA60")
print("   ✓ 所有条件都适用于指数")

print("\n【震荡阶段细分】")
print("1. 上升中继整理")
print("   条件:")
print("   - MA20走平或微升，MA60仍上行")
print("   - 价格在MA20±3%范围内震荡")
print("   - 成交量逐渐萎缩")
print("   - 低点逐步抬高")
print("   - 布林带收口")
print("   ✓ 所有条件都适用于指数")

print("\n2. 下降中继整理")
print("   条件:")
print("   - MA20走平或微降，MA60仍下行")
print("   - 价格在MA20±3%范围内震荡")
print("   - 反弹无力，高点逐步降低")
print("   - 成交量萎缩")
print("   ✓ 所有条件都适用于指数")

print("\n3. 箱体震荡")
print("   条件:")
print("   - MA20、MA60、MA250几乎走平")
print("   - 价格在±8%箱体内运行")
print("   - 成交量均衡")
print("   - RSI在40-60区间")
print("   ✓ 所有条件都适用于指数")

print("\n4. 扩张三角")
print("   条件:")
print("   - 布林带宽度扩大(>15%)")
print("   - ATR/价格 > 2%")
print("   - 高低点幅度逐步扩大")
print("   - 多空分歧加剧")
print("   ✓ 所有条件都适用于指数")

print("\n" + "=" * 80)
print("关键问题汇总")
print("=" * 80)

print("\n1. 【Total_Score定义不明确】")
print("   文档中多次使用Total_Score，但未明确定义其计算方式")
print("   建议：需要补充Total_Score的计算公式和权重分配")

print("\n2. 【上涨家数比例不适用】")
print("   这是全市场指标，不适用于单一指数")
print("   建议的替代指标：")
print("   - 使用成分股的涨跌统计（需要成分股数据）")
print("   - 使用指数的涨跌幅（但会丢失广度信息）")
print("   - 使用成交额变化")
print("   - 使用布林带宽度（衡量波动）")

print("\n3. 【散户参与度不适用】")
print("   这是股票特征，不适用于指数")
print("   建议：删除或替换为其他情绪指标")

print("\n4. 【行业板块不适用】")
print("   这是全市场特征，不适用于单一指数")
print("   建议：删除或替换为其他宏观指标")

print("\n5. 【估值数据缺失】")
print("   需要PE、PB等估值数据")
print("   建议：下载tushare的指数估值数据（index_dailybasic）")

print("\n6. 【VIX数据缺失】")
print("   需要中国波指或波动率数据")
print("   建议：使用ATR或历史波动率作为替代")

print("\n" + "=" * 80)
print("推荐实施方案")
print("=" * 80)

print("\n【方案A：简化版（推荐）】")
print("适用对象：沪深300指数")
print("核心思想：去除不适用于指数的条件，简化规则")

print("\n三状态简化版：")
print("【牛市】")
print("  必要条件:")
print("    1. MA20 > MA60 > MA250")
print("    2. 价格在MA250上方")
print("  综合标准（满足任一）:")
print("    A. Total_Score > 0.6 持续10天")
print("    B. 满足以下至少2条:")
print("       - Total_Score > 0.5")
print("       - 成交量 > MA20(Vol)×1.2")
print("       - RSI在50-70区间")
print("       - MACD金叉且>0")

print("\n【熊市】")
print("  必要条件:")
print("    1. MA20 < MA60 < MA250")
print("    2. 价格在MA250下方")
print("  综合标准（满足任一）:")
print("    A. Total_Score < -0.6 持续10天")
print("    B. 满足以下至少2条:")
print("       - Total_Score < -0.5")
print("       - 成交量 < MA20(Vol)×0.8")
print("       - RSI < 30")
print("       - MACD死叉且<0")

print("\n【震荡整理】")
print("  综合标准（满足任一）:")
print("    A. -0.3 < Total_Score < 0.3 持续15天")
print("    B. 满足以下至少2条:")
print("       - 价格在MA250±10%范围内")
print("       - 布林带宽度/中轨 < 10%")
print("       - ATR14/收盘价 < 1.5%")
print("       - RSI在40-60区间")

print("\n十二状态简化版：")
print("【牛市】4个状态 - 所有条件适用")
print("【熊市】4个状态 - 去除行业和上涨家数条件")
print("【震荡】4个状态 - 所有条件适用")

print("\n" + "=" * 80)
print("下一步建议")
print("=" * 80)

print("\n1. 需要明确定义Total_Score的计算方式")
print("2. 确认是否需要下载指数估值数据")
print("3. 确认是否需要下载VIX或使用ATR替代")
print("4. 使用简化版规则实现打标脚本")
print("5. 生成图表进行人工验证")

print("\n" + "=" * 80)
print("复核完成")
print("=" * 80)
