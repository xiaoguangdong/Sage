#!/usr/bin/env python3
"""
为沪深300指数打标（三状态和十二状态）
根据规则计算市场状态标签并生成可视化图表
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from scripts.data._shared.runtime import get_tushare_root

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HS300Labeler:
    """沪深300指数打标器"""
    
    def __init__(self, data_dir=None, output_dir="images/label"):
        """
        初始化打标器
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
        """
        self.data_dir = data_dir or str(get_tushare_root())
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载沪深300指数数据"""
        print("=" * 80)
        print("加载沪深300指数数据...")
        print("=" * 80)
        
        # 读取指数数据
        file_path = os.path.join(self.data_dir, "index", "index_000300_SH_ohlc.parquet")
        self.df = pd.read_parquet(file_path)
        
        # 按日期排序
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        print(f"✓ 数据加载完成")
        print(f"  时间范围: {self.df['date'].min().strftime('%Y-%m-%d')} 至 {self.df['date'].max().strftime('%Y-%m-%d')}")
        print(f"  数据量: {len(self.df)} 条记录")
    
    def calculate_indicators(self):
        """计算所有技术指标"""
        print("\n" + "=" * 80)
        print("计算技术指标...")
        print("=" * 80)
        
        # 1. 移动平均
        self.df['ma20'] = self.df['close'].rolling(window=20).mean()
        self.df['ma60'] = self.df['close'].rolling(window=60).mean()
        self.df['ma250'] = self.df['close'].rolling(window=250).mean()
        print("✓ 移动平均 (MA20, MA60, MA250)")
        
        # 2. MACD
        self.df['ema12'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['ema26'] = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['dif'] = self.df['ema12'] - self.df['ema26']
        self.df['dea'] = self.df['dif'].ewm(span=9, adjust=False).mean()
        self.df['macd'] = 2 * (self.df['dif'] - self.df['dea'])
        print("✓ MACD")
        
        # 3. RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        print("✓ RSI")
        
        # 4. 布林带
        self.df['bb_middle'] = self.df['close'].rolling(window=20).mean()
        self.df['bb_std'] = self.df['close'].rolling(window=20).std()
        self.df['bb_upper'] = self.df['bb_middle'] + 2 * self.df['bb_std']
        self.df['bb_lower'] = self.df['bb_middle'] - 2 * self.df['bb_std']
        print("✓ 布林带")
        
        # 5. ATR
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['atr'] = tr.rolling(window=14).mean()
        print("✓ ATR")
        
        # 6. 成交量MA
        self.df['vol_ma20'] = self.df['vol'].rolling(window=20).mean()
        print("✓ 成交量MA")
        
        # 7. 价格相对MA的位置
        self.df['price_vs_ma20'] = (self.df['close'] - self.df['ma20']) / self.df['ma20'] * 100
        self.df['price_vs_ma60'] = (self.df['close'] - self.df['ma60']) / self.df['ma60'] * 100
        self.df['price_vs_ma250'] = (self.df['close'] - self.df['ma250']) / self.df['ma250'] * 100
        
        print("\n✓ 所有技术指标计算完成")
    
    def calculate_total_score(self, i):
        """
        计算综合评分 Total_Score
        
        Args:
            i: 当前行索引
            
        Returns:
            Total_Score: 综合评分 [-1, 1]
        """
        if i < 250:
            return 0
        
        # 获取当前数据
        current = self.df.iloc[:i+1]
        close = self.df['close'].iloc[i]
        ma20 = self.df['ma20'].iloc[i]
        ma60 = self.df['ma60'].iloc[i]
        ma250 = self.df['ma250'].iloc[i]
        
        # ========== 1. 趋势指标 (权重40%) ==========
        
        # A. 移动平均系统 (40%中的15%+15%+10%)
        trend_ma_score = 0
        if close > ma20:
            trend_ma_score += 0.15  # 价格在MA20上
        if close > ma60:
            trend_ma_score += 0.15  # 价格在MA60上
        if close > ma250:
            trend_ma_score += 0.10  # 价格在MA250上
        
        # B. MACD指标 (40%中的剩余部分，这里简化为趋势的一部分)
        dif = self.df['dif'].iloc[i]
        dea = self.df['dea'].iloc[i]
        macd = self.df['macd'].iloc[i]
        
        if dif > dea and macd > 0:
            macd_score = 0.10
        elif dif > dea:
            macd_score = 0.05
        else:
            macd_score = 0
        
        trend_score = trend_ma_score + macd_score
        
        # ========== 2. 动量指标 (权重30%) ==========
        
        # A. RSI (30%中的10%)
        rsi = self.df['rsi'].iloc[i]
        if rsi > 70:
            rsi_score = -0.10
        elif rsi < 30:
            rsi_score = 0.10
        else:
            rsi_score = 0
        
        # B. 布林带位置 (30%中的10%)
        bb_upper = self.df['bb_upper'].iloc[i]
        bb_lower = self.df['bb_lower'].iloc[i]
        bb_middle = self.df['bb_middle'].iloc[i]
        
        if bb_upper - bb_lower > 0:
            bb_pos = (close - bb_lower) / (bb_upper - bb_lower)
            bb_score = (bb_pos - 0.5) * 0.20  # 归一化到[-0.1, 0.1]
        else:
            bb_score = 0
        
        # C. 价格动量 (30%中的10%，使用5日涨幅)
        if i >= 5:
            momentum = (close - self.df['close'].iloc[i-5]) / self.df['close'].iloc[i-5]
            momentum_score = np.clip(momentum * 5, -0.10, 0.10)  # 限制在[-0.1, 0.1]
        else:
            momentum_score = 0
        
        momentum_total_score = (rsi_score + bb_score + momentum_score) / 3
        
        # ========== 3. 市场广度指标 (权重20%) - 使用成交量替代 ==========
        
        # 成交量比率（偏离MA20的程度）
        vol = self.df['vol'].iloc[i]
        vol_ma20 = self.df['vol_ma20'].iloc[i]
        
        if vol_ma20 > 0:
            volume_ratio = (vol / vol_ma20 - 1) * 2  # 归一化
            breadth_score = np.clip(volume_ratio * 0.2, -0.2, 0.2)
        else:
            breadth_score = 0
        
        # ========== 4. 波动率指标 (权重10%) ==========
        
        atr = self.df['atr'].iloc[i]
        if ma20 > 0:
            atr_ratio = atr / ma20 * 10
            volatility_score = -min(0.10, atr_ratio * 0.1)
        else:
            volatility_score = 0
        
        # ========== 综合评分 ==========
        total_score = (
            trend_score * 0.4 +
            momentum_total_score * 0.3 +
            breadth_score * 0.2 +
            volatility_score * 0.1
        )
        
        return total_score
    
    def label_three_state(self):
        """
        三状态标签：牛市、熊市、震荡整理
        
        Returns:
            labels: 三状态标签数组 [0=熊, 1=震荡, 2=牛]
        """
        print("\n" + "=" * 80)
        print("计算三状态标签...")
        print("=" * 80)
        
        labels = []
        
        for i in range(len(self.df)):
            if i < 250:
                labels.append(1)  # 数据不足，默认震荡
                continue
            
            close = self.df['close'].iloc[i]
            ma20 = self.df['ma20'].iloc[i]
            ma60 = self.df['ma60'].iloc[i]
            ma250 = self.df['ma250'].iloc[i]
            total_score = self.calculate_total_score(i)
            
            # ========== 牛市判断 ==========
            bull_conditions = []
            
            # 必要条件
            if ma20 > ma60 > ma250:
                bull_conditions.append("多头排列")
            if close > ma250:
                bull_conditions.append("价格在MA250上方")
            
            # 综合标准
            if total_score > 0.6:
                bull_conditions.append("评分>0.6")
            
            # 检查其他条件
            rsi = self.df['rsi'].iloc[i]
            vol = self.df['vol'].iloc[i]
            vol_ma20 = self.df['vol_ma20'].iloc[i]
            
            if total_score > 0.5:
                bull_conditions.append("评分>0.5")
            if vol > vol_ma20 * 1.2:
                bull_conditions.append("放量")
            if 50 <= rsi <= 70:
                bull_conditions.append("RSI中性偏强")
            
            # 判断牛市（满足必要条件且至少满足2个综合标准）
            if len(bull_conditions) >= 3:
                labels.append(2)  # 牛市
                continue
            
            # ========== 熊市判断 ==========
            bear_conditions = []
            
            # 必要条件
            if ma20 < ma60 < ma250:
                bear_conditions.append("空头排列")
            if close < ma250:
                bear_conditions.append("价格在MA250下方")
            
            # 综合标准
            if total_score < -0.6:
                bear_conditions.append("评分<-0.6")
            
            if total_score < -0.5:
                bear_conditions.append("评分<-0.5")
            if vol < vol_ma20 * 0.8:
                bear_conditions.append("缩量")
            if rsi < 30:
                bear_conditions.append("RSI超卖")
            
            # 判断熊市
            if len(bear_conditions) >= 3:
                labels.append(0)  # 熊市
                continue
            
            # ========== 震荡整理 ==========
            labels.append(1)  # 震荡
        
        print(f"✓ 三状态标签计算完成")
        print(f"  牛市: {labels.count(2)} 天")
        print(f"  震荡: {labels.count(1)} 天")
        print(f"  熊市: {labels.count(0)} 天")
        
        return labels
    
    def label_twelve_state(self):
        """
        十二状态标签
        
        Returns:
            labels: 十二状态标签数组
        """
        print("\n" + "=" * 80)
        print("计算十二状态标签...")
        print("=" * 80)
        
        labels = []
        state_names = {
            0: "初始熊市",
            1: "主跌熊市",
            2: "晚期熊市",
            3: "熊市反弹",
            4: "下降中继",
            5: "箱体震荡",
            6: "上升中继",
            7: "扩张三角",
            8: "牛市调整",
            9: "初始牛市",
            10: "确认牛市",
            11: "晚期牛市"
        }
        
        for i in range(len(self.df)):
            if i < 250:
                labels.append(5)  # 默认箱体震荡
                continue
            
            close = self.df['close'].iloc[i]
            ma20 = self.df['ma20'].iloc[i]
            ma60 = self.df['ma60'].iloc[i]
            ma250 = self.df['ma250'].iloc[i]
            total_score = self.calculate_total_score(i)
            rsi = self.df['rsi'].iloc[i]
            vol = self.df['vol'].iloc[i]
            vol_ma20 = self.df['vol_ma20'].iloc[i]
            
            # ========== 牛市阶段 ==========
            if ma20 > ma60 > ma250 and close > ma250:
                # 初始牛市
                if total_score > 0.3 and total_score < 0.5:
                    labels.append(9)
                # 确认牛市
                elif total_score >= 0.6:
                    labels.append(10)
                # 晚期牛市
                elif rsi > 75 and vol > vol_ma20 * 2:
                    labels.append(11)
                # 牛市调整
                elif 0.2 <= total_score <= 0.4:
                    labels.append(8)
                else:
                    labels.append(10)
            
            # ========== 熊市阶段 ==========
            elif ma20 < ma60 < ma250 and close < ma250:
                # 初始熊市
                if total_score > -0.5 and total_score < -0.2:
                    labels.append(0)
                # 主跌熊市
                elif total_score < -0.6:
                    labels.append(1)
                # 晚期熊市
                elif rsi < 25 and vol < vol_ma20 * 0.5:
                    labels.append(2)
                # 熊市反弹
                elif -0.2 <= total_score <= 0:
                    labels.append(3)
                else:
                    labels.append(1)
            
            # ========== 震荡阶段 ==========
            else:
                # 上升中继
                if close > ma20 and ma20 > ma60 and ma20 < ma250:
                    labels.append(6)
                # 下降中继
                elif close < ma20 and ma20 < ma60 and ma20 > ma250:
                    labels.append(4)
                # 扩张三角
                elif vol > vol_ma20 * 1.5:
                    labels.append(7)
                # 箱体震荡
                else:
                    labels.append(5)
        
        # 统计各状态数量
        state_counts = {}
        for label in labels:
            state_counts[label] = state_counts.get(label, 0) + 1
        
        print(f"✓ 十二状态标签计算完成")
        for state_id in range(12):
            count = state_counts.get(state_id, 0)
            name = state_names.get(state_id, f"状态{state_id}")
            print(f"  {name}: {count} 天")
        
        return labels, state_names
    
    def resample_weekly(self):
        """按周重采样数据"""
        df_weekly = self.df.copy()
        df_weekly = df_weekly.set_index('date')
        df_weekly = df_weekly.resample('W').last().dropna()
        return df_weekly.reset_index()
    
    def plot_labels(self, df, labels_3state, labels_12state, state_names):
        """绘制图表"""
        print("\n" + "=" * 80)
        print("生成图表...")
        print("=" * 80)
        
        fig = plt.figure(figsize=(16, 14))
        
        # 准备x轴
        dates = df['date']
        x_values = np.arange(len(df))
        
        # ========== 子图1: K线图 ==========
        ax1 = plt.subplot(5, 1, 1)
        
        # 绘制K线
        for i in range(len(df)):
            open_price = df['open'].iloc[i]
            close_price = df['close'].iloc[i]
            high_price = df['high'].iloc[i]
            low_price = df['low'].iloc[i]
            
            color = 'red' if close_price >= open_price else 'green'
            
            ax1.plot([x_values[i], x_values[i]], [low_price, high_price], 
                    color=color, linewidth=0.5)
            ax1.plot([x_values[i]-0.3, x_values[i]+0.3], [open_price, open_price], 
                    color=color, linewidth=1)
            ax1.plot([x_values[i]-0.3, x_values[i]+0.3], [close_price, close_price], 
                    color=color, linewidth=1)
        
        # 绘制均线
        ax1.plot(x_values, df['ma20'], label='MA20', color='orange', linewidth=1, alpha=0.8)
        ax1.plot(x_values, df['ma60'], label='MA60', color='purple', linewidth=1, alpha=0.8)
        ax1.plot(x_values, df['ma250'], label='MA250', color='blue', linewidth=1, alpha=0.8)
        
        ax1.set_title('沪深300指数 K线图', fontsize=14, fontweight='bold')
        ax1.set_ylabel('点位', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # ========== 子图2: 三状态标签 ==========
        ax2 = plt.subplot(5, 1, 2)
        ax2.plot(x_values, labels_3state, label='三状态标签', linewidth=2)
        
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
        ax2.set_title('三状态标签（牛/熊/震荡）', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # ========== 子图3: 十二状态标签 ==========
        ax3 = plt.subplot(5, 1, 3)
        ax3.plot(x_values, labels_12state, label='十二状态标签', linewidth=2, color='blue')
        ax3.set_ylabel('状态', fontsize=12)
        ax3.set_yticks(range(12))
        ax3.set_yticklabels([state_names[i] for i in range(12)], fontsize=8)
        ax3.set_title('十二状态标签（细粒度）', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # ========== 子图4: 综合评分 ==========
        ax4 = plt.subplot(5, 1, 4)
        scores = [self.calculate_total_score(i) if i >= 250 else 0 for i in range(len(df))]
        ax4.plot(x_values, scores, label='综合评分', linewidth=2, color='orange')
        ax4.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='牛市阈值')
        ax4.axhline(y=-0.6, color='g', linestyle='--', alpha=0.5, label='熊市阈值')
        ax4.axhline(y=0.3, color='r', linestyle=':', alpha=0.3, label='牛市启动')
        ax4.axhline(y=-0.3, color='g', linestyle=':', alpha=0.3, label='熊市启动')
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        ax4.set_ylabel('评分', fontsize=12)
        ax4.set_title('综合评分 Total_Score', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left', fontsize=10)
        
        # ========== 子图5: 成交量 ==========
        ax5 = plt.subplot(5, 1, 5)
        ax5.bar(x_values, df['vol'], label='成交量', color='steelblue', alpha=0.6, width=0.8)
        ax5.plot(x_values, df['vol_ma20'], label='MA20(Vol)', color='red', linewidth=1)
        ax5.set_ylabel('成交量', fontsize=12)
        ax5.set_xlabel('日期', fontsize=12)
        ax5.set_title('成交量', fontsize=12, fontweight='bold')
        ax5.legend(loc='upper left', fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # 设置x轴日期标签
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.set_xlim(0, len(df) - 1)
            tick_indices = x_values[::max(1, len(df)//10)]
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(dates.dt.strftime('%Y-%m').iloc[::max(1, len(df)//10)], 
                            rotation=45, fontsize=8)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, "hs300_labels.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 图表已保存: {output_path}")
        return output_path
    
    def process(self):
        """处理并生成标签"""
        # 计算技术指标
        self.calculate_indicators()
        
        # 按周重采样
        df_weekly = self.resample_weekly()
        print(f"\n✓ 按周重采样: {len(df_weekly)} 周")
        
        # 临时保存周线数据用于绘图
        self.df_weekly = df_weekly
        
        # 计算标签（使用日线数据）
        labels_3state = self.label_three_state()
        labels_12state, state_names = self.label_twelve_state()
        
        # 计算Total_Score序列
        total_scores = [self.calculate_total_score(i) if i >= 250 else 0 
                       for i in range(len(self.df))]
        
        # 保存标签数据
        result_df = self.df.copy()
        result_df['label_3state'] = labels_3state
        result_df['label_12state'] = labels_12state
        result_df['total_score'] = total_scores
        
        # 保存为CSV
        output_csv = os.path.join(self.output_dir, "hs300_labels.csv")
        result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n✓ 标签数据已保存: {output_csv}")
        
        # 生成图表（使用周线数据）
        # 重新计算周线的技术指标和标签
        weekly_labeler = HS300Labeler(self.data_dir, self.output_dir)
        weekly_labeler.df = df_weekly
        weekly_labeler.calculate_indicators()
        weekly_labels_3state = weekly_labeler.label_three_state()
        weekly_labels_12state, _ = weekly_labeler.label_twelve_state()
        
        output_plot = weekly_labeler.plot_labels(
            df_weekly, weekly_labels_3state, weekly_labels_12state, state_names
        )
        
        print("\n" + "=" * 80)
        print("处理完成！")
        print("=" * 80)
        print(f"图表: {output_plot}")
        print(f"数据: {output_csv}")
        
        return result_df


def main():
    """主函数"""
    print("沪深300指数打标系统")
    print("=" * 80)
    
    # 创建打标器
    labeler = HS300Labeler(
        data_dir=str(get_tushare_root()),
        output_dir="images/label"
    )
    
    # 处理并生成标签
    result_df = labeler.process()
    
    # 显示统计信息
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)
    print(f"\n三状态分布:")
    print(f"  牛市: {result_df['label_3state'].value_counts().get(2, 0)} 天")
    print(f"  震荡: {result_df['label_3state'].value_counts().get(1, 0)} 天")
    print(f"  熊市: {result_df['label_3state'].value_counts().get(0, 0)} 天")
    
    print(f"\n综合评分统计:")
    print(f"  最大值: {result_df['total_score'].max():.3f}")
    print(f"  最小值: {result_df['total_score'].min():.3f}")
    print(f"  平均值: {result_df['total_score'].mean():.3f}")
    print(f"  标准差: {result_df['total_score'].std():.3f}")


if __name__ == "__main__":
    main()
