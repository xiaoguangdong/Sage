#!/usr/bin/env python3
"""
根据规则为股票打标并生成可视化图表
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta

from scripts.data._shared.runtime import get_tushare_root

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class StockLabeler:
    """股票打标器"""
    
    def __init__(self, data_dir=None, output_dir="images/label", 
                 num_stocks=30, start_year=2020, end_year=2025):
        """
        初始化打标器
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
            num_stocks: 要标记的股票数量
            start_year: 开始年份
            end_year: 结束年份
        """
        self.data_dir = data_dir or str(get_tushare_root())
        self.output_dir = output_dir
        self.num_stocks = num_stocks
        self.start_year = start_year
        self.end_year = end_year
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载股票数据"""
        print("=" * 80)
        print("加载股票数据...")
        print("=" * 80)
        
        # 加载日线数据
        dfs = []
        for year in range(self.start_year, self.end_year + 1):
            file_path = os.path.join(self.data_dir, "daily", f"daily_{year}.parquet")
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                dfs.append(df)
                print(f"  ✓ 加载 {year} 年数据: {len(df)} 条记录")
        
        if not dfs:
            raise ValueError("没有找到任何数据文件")
        
        self.all_data = pd.concat(dfs, ignore_index=True)
        print(f"\n总数据量: {len(self.all_data)} 条记录")
        print(f"股票数量: {self.all_data['ts_code'].nunique()} 只")
        
        # 过滤数据：排除ST和中小板
        self.filter_stocks()
    
    def filter_stocks(self):
        """过滤股票：排除ST和中小板"""
        print("\n" + "=" * 80)
        print("过滤股票...")
        print("=" * 80)
        
        # 排除ST股票
        non_st = ~self.all_data['name'].str.contains('ST', na=False)
        
        # 排除中小板（002开头）
        non_sme = ~self.all_data['ts_code'].str.startswith('002')
        
        # 过滤
        self.filtered_data = self.all_data[non_st & non_sme].copy()
        
        print(f"过滤后股票数量: {self.filtered_data['ts_code'].nunique()} 只")
        print(f"过滤后数据量: {len(self.filtered_data)} 条记录")
    
    def select_random_stocks(self):
        """随机选择股票"""
        print("\n" + "=" * 80)
        print(f"随机选择 {self.num_stocks} 只股票...")
        print("=" * 80)
        
        all_stocks = self.filtered_data['ts_code'].unique()
        selected_stocks = random.sample(list(all_stocks), 
                                       min(self.num_stocks, len(all_stocks)))
        
        print(f"已选择: {len(selected_stocks)} 只股票")
        for i, stock in enumerate(selected_stocks[:10], 1):
            stock_name = self.filtered_data[self.filtered_data['ts_code'] == stock]['name'].iloc[0]
            print(f"  {i}. {stock} - {stock_name}")
        
        if len(selected_stocks) > 10:
            print(f"  ... 还有 {len(selected_stocks) - 10} 只")
        
        return selected_stocks
    
    def calculate_indicators(self, df):
        """计算技术指标"""
        # 确保按日期排序
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 移动平均
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        df['ma250'] = df['close'].rolling(window=250).mean()
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = df['ema12'] - df['ema26']
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = 2 * (df['dif'] - df['dea'])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        # 成交量MA
        df['vol_ma20'] = df['vol'].rolling(window=20).mean()
        
        return df
    
    def calculate_total_score(self, df):
        """计算综合评分"""
        # 趋势评分（40%）
        trend_score = 0
        if df['close'].iloc[-1] > df['ma20'].iloc[-1]:
            trend_score += 0.15
        if df['close'].iloc[-1] > df['ma60'].iloc[-1]:
            trend_score += 0.15
        if df['close'].iloc[-1] > df['ma250'].iloc[-1]:
            trend_score += 0.10
        
        # MACD评分（10%）
        macd_score = 0
        if df['dif'].iloc[-1] > df['dea'].iloc[-1] and df['macd'].iloc[-1] > 0:
            macd_score = 0.10
        elif df['dif'].iloc[-1] > df['dea'].iloc[-1]:
            macd_score = 0.05
        else:
            macd_score = 0
        
        # RSI评分（10%）
        rsi = df['rsi'].iloc[-1]
        if rsi > 70:
            rsi_score = -0.10
        elif rsi < 30:
            rsi_score = 0.10
        else:
            rsi_score = 0
        
        # 布林带位置（10%）
        bb_pos = (df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / \
                 (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) if \
                 (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) > 0 else 0.5
        bb_score = (bb_pos - 0.5) * 0.20
        
        # 波动率评分（10%）
        atr_ratio = df['atr'].iloc[-1] / df['ma20'].iloc[-1] if df['ma20'].iloc[-1] > 0 else 0
        vol_score = -min(0.10, atr_ratio * 10)
        
        # 综合评分
        total_score = trend_score + macd_score + rsi_score + bb_score + vol_score
        
        return total_score
    
    def label_three_state(self, df):
        """三状态标签（牛、熊、震荡）"""
        labels = []
        
        for i in range(len(df)):
            if i < 250:  # 需要足够的历史数据
                labels.append(1)  # 震荡
                continue
            
            # 获取当前数据
            current = df.iloc[:i+1]
            total_score = self.calculate_total_score(current)
            
            # 判断状态
            if total_score > 0.3:
                labels.append(2)  # 牛市
            elif total_score < -0.3:
                labels.append(0)  # 熊市
            else:
                labels.append(1)  # 震荡
        
        return labels
    
    def label_twelve_state(self, df):
        """十二状态标签（细粒度）"""
        labels = []
        
        for i in range(len(df)):
            if i < 250:
                labels.append(5)  # 默认震荡
                continue
            
            current = df.iloc[:i+1]
            total_score = self.calculate_total_score(current)
            
            # 简化的十二状态映射
            # 0-3: 熊市阶段
            # 4-7: 震荡阶段
            # 8-11: 牛市阶段
            if total_score > 0.5:
                labels.append(10)  # 确认牛市
            elif total_score > 0.3:
                labels.append(9)   # 初始牛市
            elif total_score > 0:
                labels.append(8)   # 牛市调整
            elif total_score > -0.3:
                labels.append(5)   # 箱体震荡
            elif total_score > -0.5:
                labels.append(4)   # 下降中继
            else:
                labels.append(1)   # 确认熊市
        
        return labels
    
    def resample_weekly(self, df):
        """按周重采样数据"""
        # 转换日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')
        
        # 按周重采样，取最后一行
        weekly_df = df.resample('W').last().dropna()
        
        return weekly_df.reset_index()
    
    def plot_stock(self, stock_code, stock_name, df, labels_3state, labels_12state):
        """绘制股票K线图和标签"""
        fig = plt.figure(figsize=(16, 12))
        
        # 主图：K线图
        ax1 = plt.subplot(4, 1, 1)
        
        # 绘制K线
        for i in range(len(df)):
            open_price = df['open'].iloc[i]
            close_price = df['close'].iloc[i]
            high_price = df['high'].iloc[i]
            low_price = df['low'].iloc[i]
            
            # 颜色
            color = 'red' if close_price >= open_price else 'green'
            
            # 绘制实体
            ax1.plot([i, i], [low_price, high_price], color=color, linewidth=1)
            ax1.plot([i-0.3, i+0.3], [open_price, open_price], color=color, linewidth=2)
            ax1.plot([i-0.3, i+0.3], [close_price, close_price], color=color, linewidth=2)
        
        # 绘制均线
        ax1.plot(df.index, df['ma20'], label='MA20', color='orange', linewidth=1)
        ax1.plot(df.index, df['ma60'], label='MA60', color='purple', linewidth=1)
        ax1.plot(df.index, df['ma250'], label='MA250', color='blue', linewidth=1)
        
        ax1.set_title(f'{stock_code} - {stock_name} K线图', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 子图2：三状态标签
        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(df.index, labels_3state, label='三状态标签', linewidth=2)
        
        # 添加背景色
        for i in range(len(labels_3state) - 1):
            if labels_3state[i] == 2:  # 牛市
                ax2.axvspan(i, i+1, alpha=0.3, color='red')
            elif labels_3state[i] == 0:  # 熊市
                ax2.axvspan(i, i+1, alpha=0.3, color='green')
            else:  # 震荡
                ax2.axvspan(i, i+1, alpha=0.3, color='gray')
        
        ax2.set_ylabel('状态', fontsize=12)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['熊市', '震荡', '牛市'])
        ax2.set_title('三状态标签', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 子图3：十二状态标签
        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(df.index, labels_12state, label='十二状态标签', linewidth=2, color='blue')
        ax3.set_ylabel('状态', fontsize=12)
        ax3.set_title('十二状态标签', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 子图4：综合评分
        ax4 = plt.subplot(4, 1, 4)
        scores = [self.calculate_total_score(df.iloc[:i+1]) 
                  if i >= 250 else 0 for i in range(len(df))]
        ax4.plot(df.index, scores, label='综合评分', linewidth=2, color='orange')
        ax4.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='牛市阈值')
        ax4.axhline(y=-0.3, color='g', linestyle='--', alpha=0.5, label='熊市阈值')
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        ax4.set_ylabel('评分', fontsize=12)
        ax4.set_xlabel('时间', fontsize=12)
        ax4.set_title('综合评分', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left', fontsize=10)
        
        # 调整x轴
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(0, len(df) - 1)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(self.output_dir, f"{stock_code}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def process_stocks(self):
        """处理所有选中的股票"""
        print("\n" + "=" * 80)
        print("开始处理股票...")
        print("=" * 80)
        
        # 选择股票
        selected_stocks = self.select_random_stocks()
        
        # 处理每只股票
        results = []
        for i, stock_code in enumerate(selected_stocks, 1):
            print(f"\n处理 {i}/{len(selected_stocks)}: {stock_code}")
            
            try:
                # 获取股票数据
                stock_data = self.filtered_data[
                    self.filtered_data['ts_code'] == stock_code
                ].copy()
                
                if len(stock_data) < 300:
                    print(f"  ✗ 数据不足，跳过")
                    continue
                
                stock_name = stock_data['name'].iloc[0]
                
                # 计算指标
                stock_data = self.calculate_indicators(stock_data)
                
                # 按周重采样
                weekly_data = self.resample_weekly(stock_data)
                
                # 重新计算指标（周线）
                weekly_data = self.calculate_indicators(weekly_data)
                
                # 计算标签
                labels_3state = self.label_three_state(weekly_data)
                labels_12state = self.label_twelve_state(weekly_data)
                
                # 绘图
                output_path = self.plot_stock(
                    stock_code, stock_name, weekly_data, 
                    labels_3state, labels_12state
                )
                
                print(f"  ✓ 已保存: {output_path}")
                results.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'output_path': output_path,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
                results.append({
                    'stock_code': stock_code,
                    'stock_name': stock_code,
                    'output_path': None,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # 输出结果摘要
        print("\n" + "=" * 80)
        print("处理完成！")
        print("=" * 80)
        success_count = sum(1 for r in results if r['status'] == 'success')
        print(f"成功: {success_count}/{len(results)}")
        print(f"失败: {len(results) - success_count}/{len(results)}")
        print(f"\n图表保存在: {self.output_dir}")
        
        return results


def main():
    """主函数"""
    # 设置随机种子，保证可重复
    random.seed(42)
    np.random.seed(42)
    
    # 创建打标器
    labeler = StockLabeler(
        data_dir=str(get_tushare_root()),
        output_dir="images/label",
        num_stocks=30,  # 默认30只股票
        start_year=2020,
        end_year=2025
    )
    
    # 处理股票
    results = labeler.process_stocks()
    
    # 保存结果报告
    report_df = pd.DataFrame(results)
    report_path = os.path.join("images/label", "label_report.csv")
    report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
