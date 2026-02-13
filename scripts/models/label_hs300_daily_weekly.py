"""
沪深300指数市场状态打标 - 支持日线和周线
对比多种方法：
1. 原始标签（基于综合评分）
2. HMM模型（隐马尔可夫）
3. 均线确认（MA20/MA60+价格+斜率+持续确认）
4. 多均线系统（MA20/60/120/250组合）
5. 原始+均线融合
6. HMM+均线融合
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
    """沪深300指数打标器 - 支持日线和周线"""
    
    def __init__(self, data_dir=None, output_dir="images/label", timeframe='weekly'):
        """
        初始化打标器
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
            timeframe: 时间周期 ('daily' 或 'weekly')
        """
        self.data_dir = data_dir or str(get_tushare_root())
        self.output_dir = output_dir
        self.timeframe = timeframe
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载数据
        self.load_data()

        # 平滑参数
        self.smooth_method = "hmm"  # hmm / short_segment
        self.short_segment_max = 5
        self.smooth_stay_prob = 0.90
        self.smooth_obs_acc = 0.85

        # 融合参数（原始/ HMM 标签 + 均线确认）
        self.fusion_confirm_days = 3
        # 默认主信号列
        self.main_label_col = "label_ma_confirmation"
        # 置信度参数（均线确认）
        self.conf_min_scale = 1e-6
        self.conf_slope_scale = None
        self.conf_diff_scale = None
        self.conf_price_scale = None
    
    def load_data(self):
        """加载沪深300指数和成分股数据"""
        print("=" * 80)
        print(f"加载沪深300指数和成分股数据（{self.timeframe}）...")
        print("=" * 80)
        
        # 读取指数数据
        index_file = os.path.join(self.data_dir, "index", "index_000300_SH_ohlc.parquet")
        self.df_index = pd.read_parquet(index_file)
        self.df_index = self.df_index.sort_values('date').reset_index(drop=True)
        self.df_index['date'] = pd.to_datetime(self.df_index['date'])
        
        # 根据时间周期处理
        if self.timeframe == 'daily':
            self.df = self.df_index.copy()
        elif self.timeframe == 'weekly':
            # 重采样为周线
            self.df = self.df_index.resample('W-MON', on='date').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'vol': 'sum'
            }).dropna().reset_index()
        
        print(f"✓ 指数数据加载完成")
        print(f"  时间范围: {self.df['date'].min().strftime('%Y-%m-%d')} 至 {self.df['date'].max().strftime('%Y-%m-%d')}")
        print(f"  数据量: {len(self.df)} 条记录")
        
        # 读取成分股数据（用于计算广度指标）
        constituents_file = os.path.join(self.data_dir, "constituents", "hs300_constituents_all.parquet")
        self.df_constituents = pd.read_parquet(constituents_file)
        self.stock_list = self.df_constituents['con_code'].unique()
        print(f"✓ 成分股数据加载完成")
        print(f"  成分股数量: {len(self.stock_list)} 只")
        
        # 加载成分股日线数据
        self.load_stock_data()
    
    def load_stock_data(self):
        """加载成分股日线数据"""
        print("\n加载成分股日线数据...")
        
        # 读取所有年份的日线数据
        daily_files = []
        for year in range(2020, 2027):
            daily_file = os.path.join(self.data_dir, "daily", f"daily_{year}.parquet")
            if os.path.exists(daily_file):
                daily_files.append(pd.read_parquet(daily_file))
        
        # 合并所有日线数据
        if daily_files:
            self.df_daily = pd.concat(daily_files, ignore_index=True)
            self.df_daily['trade_date'] = pd.to_datetime(self.df_daily['trade_date'])
            
            # 筛选沪深300成分股
            self.df_daily = self.df_daily[
                self.df_daily['ts_code'].isin(self.stock_list)
            ]
            
            print(f"✓ 成分股日线数据加载完成")
            print(f"  数据量: {len(self.df_daily)} 条记录")
        else:
            print("⚠ 未找到日线数据文件")
            self.df_daily = pd.DataFrame()
    
    def calculate_indicators(self):
        """计算技术指标"""
        print("\n" + "=" * 80)
        print(f"计算技术指标（{self.timeframe}）...")
        print("=" * 80)
        
        df = self.df
        
        # 设置均线周期
        if self.timeframe == 'daily':
            ma_periods = [20, 60, 120, 250]
        else:
            ma_periods = [4, 12, 24, 52]  # 周线对应
        
        # 1. 移动平均
        for period in ma_periods:
            df[f'ma{period}'] = df['close'].rolling(window=period).mean()
        print(f"✓ 移动平均: {ma_periods}")
        
        # 2. MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = df['ema12'] - df['ema26']
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = 2 * (df['dif'] - df['dea'])
        print("✓ MACD")
        
        # 3. RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        print("✓ RSI")
        
        # 4. 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        print("✓ 布林带")
        
        # 5. ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        print("✓ ATR")
        
        # 6. 成交量MA
        df['vol_ma20'] = df['vol'].rolling(window=20).mean()
        print("✓ 成交量MA")
        
        # 7. 4周/4日收益和波动
        if self.timeframe == 'daily':
            df['ret_4d'] = df['close'].pct_change(4)
            df['vol_4d'] = df['ret_4d'].rolling(window=4).std()
            df['ma_diff'] = df['ma20'] - df['ma60']
            df['ma_diff_norm'] = df['ma_diff'] / df['ma60'] * 100
            df['ma_slope'] = df['ma20'].diff()
        else:
            df['ret_4w'] = df['close'].pct_change(4)
            df['vol_4w'] = df['ret_4w'].rolling(window=4).std()
            df['ma_diff'] = df['ma12'] - df['ma24']
            df['ma_diff_norm'] = df['ma_diff'] / df['ma24'] * 100
            df['ma_slope'] = df['ma12'].diff()

        # 9. 成交量比率
        df['vol_ratio'] = df['vol'] / df['vol'].rolling(window=20).mean()
        
        print("\n✓ 所有技术指标计算完成")
    
    def calculate_breadth(self):
        """计算市场广度指标"""
        print("\n" + "=" * 80)
        print("计算市场广度指标...")
        print("=" * 80)
        
        if self.df_daily.empty:
            print("⚠ 没有日线数据，跳过市场广度指标计算")
            self.df['breadth'] = 0.5
            self.df['new_high_ratio'] = 0
            return
        
        # 按周期聚合成分股数据
        if self.timeframe == 'daily':
            # 日线：直接聚合
            freq = 'D'
        else:
            # 周线：按周聚合
            freq = 'W-MON'
        
        # 计算上涨股票占比
        weekly_rising = self.df_daily.groupby(
            pd.Grouper(key='trade_date', freq=freq)
        ).apply(
            lambda x: np.sum(x['pct_chg'] > 0) / len(x) if len(x) > 0 else 0
        ).reset_index()
        
        weekly_rising.columns = ['date', 'breadth']
        
        # 计算创新高比例（简化：涨跌幅>5%）
        weekly_new_high = self.df_daily.groupby(
            pd.Grouper(key='trade_date', freq=freq)
        ).apply(
            lambda x: np.sum(x['pct_chg'] > 5) / len(x) if len(x) > 0 else 0
        ).reset_index()
        
        weekly_new_high.columns = ['date', 'new_high_ratio']
        
        # 合并到主数据
        self.df = pd.merge(
            self.df,
            weekly_rising,
            on='date',
            how='left'
        )
        
        self.df = pd.merge(
            self.df,
            weekly_new_high,
            on='date',
            how='left'
        )
        
        # 填充缺失值
        self.df['breadth'] = self.df['breadth'].fillna(0.5)
        self.df['new_high_ratio'] = self.df['new_high_ratio'].fillna(0)
        
        print("✓ 市场广度指标计算完成")

    def smooth_labels(self, labels):
        """
        标签平滑
        - hmm: 基于HMM(Viterbi)对原始标签进行平滑
        - short_segment: 短段填补
        """
        if self.smooth_method == "hmm":
            return self._smooth_labels_hmm(labels)

        max_len = self.short_segment_max
        labels = [int(v) for v in labels]
        n = len(labels)

        segments = []
        start = 0
        for i in range(1, n):
            if labels[i] != labels[i - 1]:
                segments.append((start, i - 1, labels[i - 1]))
                start = i
        segments.append((start, n - 1, labels[-1]))

        filled = labels[:]
        for idx in range(1, len(segments) - 1):
            s, e, val = segments[idx]
            prev_val = segments[idx - 1][2]
            next_val = segments[idx + 1][2]
            if prev_val == next_val and prev_val != val:
                length = e - s + 1
                if length <= max_len:
                    for j in range(s, e + 1):
                        filled[j] = prev_val

        return filled

    def _smooth_labels_hmm(self, labels):
        labels = [int(v) for v in labels]
        n = len(labels)
        states = [0, 1, 2]
        k = len(states)

        stay = float(self.smooth_stay_prob)
        obs_acc = float(self.smooth_obs_acc)
        trans = np.full((k, k), (1 - stay) / (k - 1))
        np.fill_diagonal(trans, stay)

        emit = np.full((k, k), (1 - obs_acc) / (k - 1))
        np.fill_diagonal(emit, obs_acc)

        log_trans = np.log(trans + 1e-12)
        log_emit = np.log(emit + 1e-12)
        log_init = np.log(np.full(k, 1.0 / k))

        dp = np.zeros((n, k))
        back = np.zeros((n, k), dtype=int)

        dp[0] = log_init + log_emit[:, labels[0]]

        for t in range(1, n):
            obs = labels[t]
            for s in range(k):
                scores = dp[t - 1] + log_trans[:, s]
                back[t, s] = int(np.argmax(scores))
                dp[t, s] = scores[back[t, s]] + log_emit[s, obs]

        states_seq = [0] * n
        states_seq[-1] = int(np.argmax(dp[-1]))
        for t in range(n - 2, -1, -1):
            states_seq[t] = back[t + 1, states_seq[t + 1]]

        return states_seq
    
    def calculate_score(self, i):
        """
        计算综合评分
        
        Args:
            i: 当前行索引
            
        Returns:
            score: 综合评分 [-1, 1]
        """
        df = self.df
        
        # 数据不足
        if i < 20:
            return 0
        
        # 获取数据
        close = df['close'].iloc[i]
        ma20 = df['ma20'].iloc[i]
        ma60 = df['ma60'].iloc[i]
        ma250 = df['ma250'].iloc[i] if 'ma250' in df.columns else ma60
        
        # 检查数据有效性
        if pd.isna(ma20) or pd.isna(ma60):
            return 0
        
        # 使用MA60替代MA250
        if pd.isna(ma250):
            ma250 = ma60
        
        # 1. 趋势指标 (40%)
        trend_ma_score = 0
        if close > ma20:
            trend_ma_score += 0.15
        if close > ma60:
            trend_ma_score += 0.15
        if close > ma250:
            trend_ma_score += 0.10
        
        # MACD
        dif = df['dif'].iloc[i]
        dea = df['dea'].iloc[i]
        macd = df['macd'].iloc[i]
        
        if dif > dea and macd > 0:
            macd_score = 0.10
        elif dif > dea:
            macd_score = 0.05
        else:
            macd_score = 0
        
        trend_score = trend_ma_score + macd_score
        
        # 2. 动量指标 (30%)
        rsi = df['rsi'].iloc[i]
        if rsi > 70:
            rsi_score = -0.10
        elif rsi < 30:
            rsi_score = 0.10
        else:
            rsi_score = 0
        
        # 布林带
        bb_upper = df['bb_upper'].iloc[i]
        bb_lower = df['bb_lower'].iloc[i]
        if bb_upper - bb_lower > 0:
            bb_pos = (close - bb_lower) / (bb_upper - bb_lower)
            bb_score = (bb_pos - 0.5) * 0.20
        else:
            bb_score = 0
        
        # 价格动量
        if i >= 5:
            momentum = (close - df['close'].iloc[i-5]) / df['close'].iloc[i-5]
            momentum_score = np.clip(momentum * 5, -0.10, 0.10)
        else:
            momentum_score = 0
        
        momentum_total_score = (rsi_score + bb_score + momentum_score) / 3
        
        # 3. 成交量指标 (20%)
        vol = df['vol'].iloc[i]
        vol_ma20 = df['vol_ma20'].iloc[i]
        
        if vol_ma20 > 0:
            volume_ratio = (vol / vol_ma20 - 1) * 2
            breadth_score = np.clip(volume_ratio * 0.2, -0.2, 0.2)
        else:
            breadth_score = 0
        
        # 4. 波动率指标 (10%)
        atr = df['atr'].iloc[i]
        if ma20 > 0:
            atr_ratio = atr / ma20 * 10
            volatility_score = -min(0.10, atr_ratio * 0.1)
        else:
            volatility_score = 0
        
        # 总分
        total_score = trend_score * 0.4 + momentum_total_score * 0.3 + breadth_score * 0.2 + volatility_score * 0.1
        
        return np.clip(total_score, -1, 1)
    
    def label_original(self):
        """方法1：原始标签（基于综合评分）"""
        print("\n" + "=" * 80)
        print("方法1：原始标签（基于综合评分）...")
        print("=" * 80)
        
        df = self.df
        labels = []
        scores = []
        
        for i in range(len(df)):
            score = self.calculate_score(i)
            scores.append(score)
            
            # 数据不足
            if i < 20:
                labels.append(1)
                continue
            
            # 获取关键指标
            ret_4 = df['ret_4d'].iloc[i] if 'ret_4d' in df.columns else df['ret_4w'].iloc[i]
            ma_diff = df['ma_diff_norm'].iloc[i]
            breadth = df['breadth'].iloc[i]
            
            if pd.isna(ret_4) or pd.isna(ma_diff):
                labels.append(1)
                continue
            
            # 牛市判断
            bull_conditions = []
            if ma_diff > 0:
                bull_conditions.append("MA_diff > 0")
            if ret_4 > 0:
                bull_conditions.append("4周期正收益")
            if score > 0.6:
                bull_conditions.append("评分>0.6")
            if ret_4 > 0.05:
                bull_conditions.append("4周期涨幅>5%")
            if breadth > 0.6:
                bull_conditions.append("上涨占比>60%")
            
            if len(bull_conditions) >= 3:
                labels.append(2)
                continue
            
            # 熊市判断
            bear_conditions = []
            if ma_diff < 0:
                bear_conditions.append("MA_diff < 0")
            if ret_4 < 0:
                bear_conditions.append("4周期负收益")
            if score < -0.6:
                bear_conditions.append("评分<-0.6")
            if ret_4 < -0.05:
                bear_conditions.append("4周期跌幅>5%")
            if breadth < 0.4:
                bear_conditions.append("上涨占比<40%")
            
            if len(bear_conditions) >= 3:
                labels.append(0)
                continue
            
            # 震荡
            labels.append(1)
        
        df['label_original'] = labels
        df['score'] = scores
        
        print(f"✓ 原始标签生成完成")
        print(f"  牛市: {labels.count(2)} 天/周")
        print(f"  震荡: {labels.count(1)} 天/周")
        print(f"  熊市: {labels.count(0)} 天/周")
        
        return labels

    def label_original_smooth(self):
        """方法1b：原始标签平滑"""
        print("\n" + "=" * 80)
        print("方法1b：原始标签平滑...")
        print("=" * 80)

        if 'label_original' not in self.df.columns:
            raise ValueError("请先生成原始标签 label_original")

        labels = self.df['label_original'].tolist()
        smoothed = self.smooth_labels(labels)
        self.df['label_original_smooth'] = smoothed

        print(f"✓ 原始标签平滑完成（method={self.smooth_method}, "
              f"short_segment_max={self.short_segment_max}, "
              f"stay={self.smooth_stay_prob}, obs_acc={self.smooth_obs_acc})")
        print(f"  牛市: {smoothed.count(2)} 天/周")
        print(f"  震荡: {smoothed.count(1)} 天/周")
        print(f"  熊市: {smoothed.count(0)} 天/周")

        return smoothed

    def _fuse_with_confirmation(self, base_labels, confirm_labels, confirm_days):
        fused = []
        last_confirm = None
        confirm_streak = 0

        for i in range(len(base_labels)):
            base = int(base_labels[i])
            confirm = int(confirm_labels[i])

            if not fused:
                fused.append(base)
                last_confirm = confirm
                confirm_streak = 1
                continue

            if confirm == last_confirm:
                confirm_streak += 1
            else:
                last_confirm = confirm
                confirm_streak = 1

            if base == confirm:
                fused.append(base)
                continue

            # 不一致时：等待均线连续确认
            if confirm_streak >= confirm_days:
                fused.append(confirm)
            else:
                fused.append(fused[-1])

        return fused

    def label_fused(self):
        """方法1c：原始标签与均线确认融合"""
        print("\n" + "=" * 80)
        print("方法1c：原始标签与均线确认融合...")
        print("=" * 80)

        if 'label_original' not in self.df.columns:
            raise ValueError("请先生成原始标签 label_original")
        if 'label_ma_confirmation' not in self.df.columns:
            raise ValueError("请先生成均线确认标签 label_ma_confirmation")

        orig = self.df['label_original'].tolist()
        ma = self.df['label_ma_confirmation'].tolist()
        confirm_days = self.fusion_confirm_days
        fused = self._fuse_with_confirmation(orig, ma, confirm_days)

        self.df['label_fused'] = fused

        print(f"✓ 融合完成（confirm_days={confirm_days})")
        print(f"  牛市: {fused.count(2)} 天/周")
        print(f"  震荡: {fused.count(1)} 天/周")
        print(f"  熊市: {fused.count(0)} 天/周")

        return fused

    def label_hmm_fused(self):
        """方法2b：HMM标签与均线确认融合"""
        print("\n" + "=" * 80)
        print("方法2b：HMM标签与均线确认融合...")
        print("=" * 80)

        if 'label_hmm' not in self.df.columns:
            raise ValueError("请先生成HMM标签 label_hmm")
        if 'label_ma_confirmation' not in self.df.columns:
            raise ValueError("请先生成均线确认标签 label_ma_confirmation")

        hmm_labels = self.df['label_hmm'].tolist()
        ma = self.df['label_ma_confirmation'].tolist()
        confirm_days = self.fusion_confirm_days

        fused = self._fuse_with_confirmation(hmm_labels, ma, confirm_days)
        self.df['label_hmm_fused'] = fused

        print(f"✓ 融合完成（confirm_days={confirm_days})")
        print(f"  牛市: {fused.count(2)} 天/周")
        print(f"  震荡: {fused.count(1)} 天/周")
        print(f"  熊市: {fused.count(0)} 天/周")

        return fused

    def label_main(self):
        """主信号：默认使用均线确认"""
        print("\n" + "=" * 80)
        print("主信号生成...")
        print("=" * 80)

        main_col = self.main_label_col
        if main_col not in self.df.columns:
            raise ValueError(f"主信号列不存在: {main_col}")

        self.df['label_main'] = self.df[main_col].tolist()
        labels = self.df['label_main'].tolist()

        if main_col == "label_ma_confirmation":
            self.df['label_main_confidence'] = self._calc_ma_confidence(labels)
        else:
            self.df['label_main_confidence'] = np.nan
            print(f"⚠️ 主信号来源非均线确认，置信度暂未计算：{main_col}")

        print(f"✓ 主信号完成（source={main_col})")
        print(f"  牛市: {labels.count(2)} 天/周")
        print(f"  震荡: {labels.count(1)} 天/周")
        print(f"  熊市: {labels.count(0)} 天/周")

        return labels

    def _calc_ma_confidence(self, labels):
        """
        均线确认置信度（0~1）
        - 依据：ma_diff_norm、ma_slope、价格偏离MA20
        - 牛/熊：0.5~1.0；震荡：0.2~0.5
        """
        df = self.df

        ma20 = df['ma20']
        ma60 = df['ma60']
        ma_slope = df['ma_slope']
        close = df['close']
        ma_diff_norm = df['ma_diff_norm']

        price_dist = (close / ma20 - 1.0)

        diff_abs = ma_diff_norm.abs()
        slope_abs = ma_slope.abs()
        price_abs = price_dist.abs()

        diff_scale = self.conf_diff_scale or max(diff_abs.median(skipna=True), self.conf_min_scale)
        slope_scale = self.conf_slope_scale or max(slope_abs.median(skipna=True), self.conf_min_scale)
        price_scale = self.conf_price_scale or max(price_abs.median(skipna=True), self.conf_min_scale)

        diff_score = np.tanh(diff_abs / diff_scale)
        slope_score = np.tanh(slope_abs / slope_scale)
        price_score = np.tanh(price_abs / price_scale)

        base_score = (diff_score + slope_score + price_score) / 3.0
        conf = []

        for i, label in enumerate(labels):
            if pd.isna(ma20.iloc[i]) or pd.isna(ma60.iloc[i]) or pd.isna(ma_slope.iloc[i]):
                conf.append(0.3)
                continue

            if label == 1:
                value = 0.2 + 0.3 * base_score.iloc[i]
            else:
                value = 0.5 + 0.5 * base_score.iloc[i]

            conf.append(round(float(value), 4))

        return conf
    
    def label_hmm(self, mapping_mode: str = "future_return"):
        """方法2：HMM模型（隐马尔可夫）"""
        print("\n" + "=" * 80)
        print("方法2：HMM模型（隐马尔可夫）...")
        print("=" * 80)
        
        try:
            from hmmlearn import hmm
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("⚠️ 未安装hmmlearn库，跳过HMM方法")
            self.df['label_hmm'] = [1] * len(self.df)
            return [1] * len(self.df)
        
        df = self.df
        
        # 准备特征
        feature_cols = []
        if 'ret_4d' in df.columns:
            feature_cols = ['ret_4d', 'ma_diff_norm', 'breadth', 'vol_4d', 'ma_slope', 'vol_ratio']
        else:
            feature_cols = ['ret_4w', 'ma_diff_norm', 'breadth', 'vol_4w', 'ma_slope', 'vol_ratio']
        
        df_features = df[feature_cols].dropna()
        
        if len(df_features) < 100:
            print("⚠️ 有效数据不足，无法训练HMM模型")
            self.df['label_hmm'] = [1] * len(self.df)
            return [1] * len(self.df)
        
        features = df_features.values
        
        # 标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 训练HMM（4状态）
        n_components = 4
        print(f"训练HMM模型（{n_components}状态）...")
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        model.fit(features_scaled)
        
        # 预测状态
        states = model.predict(features_scaled)
        
        # 状态映射
        if mapping_mode == "future_return":
            horizon = 20 if self.timeframe == 'daily' else 4
            fwd_ret = df['close'].shift(-horizon) / df['close'] - 1
            fwd_ret = fwd_ret.loc[df_features.index]

            state_scores = []
            for state in range(n_components):
                mask = (states == state)
                score = fwd_ret[mask].mean() if mask.any() else 0
                state_scores.append(score)

            # 由未来收益从低到高映射：最差=熊市，最好=牛市，其余=震荡
            sorted_states = sorted(range(n_components), key=lambda x: state_scores[x])
            state_mapping = {s: 1 for s in range(n_components)}
            state_mapping[sorted_states[0]] = 0
            state_mapping[sorted_states[-1]] = 2
        else:
            # 旧规则：按特征评分排序
            state_scores = []
            for state in range(n_components):
                mask = (states == state)
                if mask.sum() > 0:
                    ret_mean = features[mask, 0].mean()
                    ma_diff_mean = features[mask, 1].mean()
                    ma_slope_mean = features[mask, 4].mean()
                    score = ma_diff_mean * 50 + ma_slope_mean * 30 + ret_mean * 20
                else:
                    score = 0
                state_scores.append(score)

            sorted_states = sorted(range(n_components), key=lambda x: state_scores[x], reverse=True)
            state_mapping = {
                sorted_states[0]: 2,  # 牛市
                sorted_states[1]: 2,  # 牛市
                sorted_states[2]: 1,  # 震荡
                sorted_states[3]: 0   # 熊市
            }
        
        labels_mapped = [state_mapping[s] for s in states]
        
        # 扩展到完整数据
        labels_full = [1] * len(df)
        valid_indices = df_features.index.tolist()
        for idx, label in zip(valid_indices, labels_mapped):
            labels_full[idx] = label
        
        df['label_hmm'] = labels_full
        
        print(f"✓ HMM标签生成完成")
        print(f"  牛市: {labels_full.count(2)} 天/周")
        print(f"  震荡: {labels_full.count(1)} 天/周")
        print(f"  熊市: {labels_full.count(0)} 天/周")
        
        return labels_full
    
    def label_ma_confirmation(self, confirmation_periods=3):
        """方法3：均线确认机制（MA20/MA60+价格+斜率+持续确认）"""
        print("\n" + "=" * 80)
        print("方法3：均线确认机制...")
        print("=" * 80)
        
        df = self.df
        labels = []
        current_state = 1
        confirmation_count = 0
        
        for i in range(len(df)):
            if i < 20:
                labels.append(1)
                continue
            
            close = df['close'].iloc[i]
            ma20 = df['ma20'].iloc[i]
            ma60 = df['ma60'].iloc[i]
            ma_slope = df['ma_slope'].iloc[i]
            
            if pd.isna(ma20) or pd.isna(ma60) or pd.isna(ma_slope):
                labels.append(1)
                continue
            
            # 牛市条件
            bull_conditions = (ma20 > ma60) and (close > ma20) and (ma_slope > 0)
            
            # 熊市条件
            bear_conditions = (ma20 < ma60) and (close < ma20) and (ma_slope < 0)
            
            if bull_conditions:
                if current_state == 2:
                    labels.append(2)
                    confirmation_count += 1
                elif current_state == 1:
                    confirmation_count += 1
                    if confirmation_count >= confirmation_periods:
                        current_state = 2
                        labels.append(2)
                    else:
                        labels.append(1)
                else:
                    confirmation_count = 1
                    labels.append(1)
                    
            elif bear_conditions:
                if current_state == 0:
                    labels.append(0)
                    confirmation_count += 1
                elif current_state == 1:
                    confirmation_count += 1
                    if confirmation_count >= confirmation_periods:
                        current_state = 0
                        labels.append(0)
                    else:
                        labels.append(1)
                else:
                    confirmation_count = 1
                    labels.append(1)
                    
            else:
                current_state = 1
                confirmation_count = 0
                labels.append(1)
        
        df['label_ma_confirmation'] = labels
        
        print(f"✓ 均线确认标签生成完成")
        print(f"  牛市: {labels.count(2)} 天/周")
        print(f"  震荡: {labels.count(1)} 天/周")
        print(f"  熊市: {labels.count(0)} 天/周")
        
        return labels
    
    def label_multi_ma(self):
        """方法4：多均线系统（MA20/60/120/250组合）"""
        print("\n" + "=" * 80)
        print("方法4：多均线系统...")
        print("=" * 80)
        
        df = self.df
        labels = []
        
        # 获取均线周期
        if 'ma120' in df.columns:
            ma_periods = [20, 60, 120, 250]
        else:
            ma_periods = [4, 12, 24, 52]
        
        for i in range(len(df)):
            if i < 250 if 'ma250' in df.columns else i < 60:
                labels.append(1)
                continue
            
            close = df['close'].iloc[i]
            ma_list = [df[f'ma{p}'].iloc[i] for p in ma_periods]
            
            # 检查数据有效性
            if any(pd.isna(ma) for ma in ma_list):
                labels.append(1)
                continue
            
            # 判断均线排列
            ma_aligned = all(ma_list[i] < ma_list[i+1] for i in range(len(ma_list)-1))
            ma_aligned_reverse = all(ma_list[i] > ma_list[i+1] for i in range(len(ma_list)-1))
            
            # 判断价格位置
            price_above_all = all(close > ma for ma in ma_list)
            price_below_all = all(close < ma for ma in ma_list)
            
            # 牛市判断：多头排列 + 价格在所有均线上方
            if ma_aligned and price_above_all:
                labels.append(2)
            # 熊市判断：空头排列 + 价格在所有均线下方
            elif ma_aligned_reverse and price_below_all:
                labels.append(0)
            # 震荡
            else:
                labels.append(1)
        
        df['label_multi_ma'] = labels
        
        print(f"✓ 多均线标签生成完成")
        print(f"  牛市: {labels.count(2)} 天/周")
        print(f"  震荡: {labels.count(1)} 天/周")
        print(f"  熊市: {labels.count(0)} 天/周")
        
        return labels
    
    def generate_statistics(self):
        """生成统计信息"""
        print("\n" + "=" * 80)
        print("多种方法统计对比")
        print("=" * 80)
        
        df = self.df
        
        methods = [
            ('原始标签', 'label_original'),
            ('原始标签(平滑)', 'label_original_smooth'),
            ('原始+均线融合', 'label_fused'),
            ('HMM模型', 'label_hmm'),
            ('HMM+均线融合', 'label_hmm_fused'),
            ('均线确认', 'label_ma_confirmation'),
            ('主信号(默认)', 'label_main'),
            ('多均线系统', 'label_multi_ma')
        ]
        
        total = len(df)
        
        for name, col in methods:
            labels = df[col].tolist()
            bull = labels.count(2)
            sideways = labels.count(1)
            bear = labels.count(0)
            
            # 计算切换次数
            transitions = 0
            for i in range(1, len(labels)):
                if labels[i] != labels[i-1]:
                    transitions += 1
            
            print(f"\n{name}:")
            print(f"  牛市: {bull} 天/周 ({bull/total*100:.1f}%)")
            print(f"  震荡: {sideways} 天/周 ({sideways/total*100:.1f}%)")
            print(f"  熊市: {bear} 天/周 ({bear/total*100:.1f}%)")
            print(f"  状态切换: {transitions} 次")
            print(f"  平均持续: {total/transitions:.1f} 天/周")
    
    def visualize(self):
        """可视化结果"""
        print("\n" + "=" * 80)
        print("生成可视化图表...")
        print("=" * 80)
        
        df = self.df
        
        methods = [
            ('原始标签', 'label_original'),
            ('原始标签(平滑)', 'label_original_smooth'),
            ('原始+均线融合', 'label_fused'),
            ('HMM模型', 'label_hmm'),
            ('HMM+均线融合', 'label_hmm_fused'),
            ('均线确认', 'label_ma_confirmation'),
            ('主信号(默认)', 'label_main'),
            ('多均线系统', 'label_multi_ma')
        ]

        methods = [(name, col) for name, col in methods if col in df.columns]

        # 创建图表（动态子图数量）
        fig, axes = plt.subplots(len(methods), 1, figsize=(16, 4 * len(methods)), sharex=True)
        if len(methods) == 1:
            axes = [axes]
        
        methods = [(name, col, axes[i]) for i, (name, col) in enumerate(methods)]
        
        for name, col, ax in methods:
            # 绘制价格
            ax.plot(df['date'], df['close'], label='价格', linewidth=1.5, color='black', alpha=0.7)
            
            # 绘制MA20和MA60
            ax.plot(df['date'], df['ma20'], label='MA20', linewidth=1, color='blue', alpha=0.5)
            ax.plot(df['date'], df['ma60'], label='MA60', linewidth=1, color='orange', alpha=0.5)
            
            # 用颜色标记市场状态
            for i in range(1, len(df)):
                if df[col].iloc[i] == 2:
                    ax.axvspan(df['date'].iloc[i-1], df['date'].iloc[i], 
                              alpha=0.2, color='green')
                elif df[col].iloc[i] == 0:
                    ax.axvspan(df['date'].iloc[i-1], df['date'].iloc[i], 
                              alpha=0.2, color='red')
            
            ax.set_ylabel('价格', fontsize=11)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 格式化x轴
        axes[-1].set_xlabel('日期', fontsize=12)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=12 if self.timeframe == 'weekly' else 6))
        plt.xticks(rotation=45)
        
        plt.suptitle(f'沪深300指数 - {self.timeframe}对比（多种方法）', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        filename = f'hs300_{self.timeframe}_comparison.png'
        output_file = os.path.join(self.output_dir, filename)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存: {output_file}")
        
        plt.close()
    
    def save_results(self):
        """保存结果数据"""
        print("\n" + "=" * 80)
        print("保存结果数据...")
        print("=" * 80)
        
        filename = f'hs300_{self.timeframe}_labels.csv'
        output_file = os.path.join(self.output_dir, filename)
        self.df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 数据已保存: {output_file}")
        print(f"  总计: {len(self.df)} 条记录")
    
    def run(self):
        """运行完整流程"""
        print("=" * 80)
        print(f"沪深300指数市场状态打标（{self.timeframe}）")
        print("对比多种方法：原始、HMM、均线确认、多均线与融合")
        print("=" * 80)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 计算指标
        self.calculate_indicators()
        
        # 3. 计算广度
        self.calculate_breadth()
        
        # 4. 生成多种标签
        self.label_original()
        self.label_original_smooth()
        self.label_hmm()
        self.label_ma_confirmation()
        self.label_fused()
        self.label_hmm_fused()
        self.label_main()
        self.label_multi_ma()
        
        # 5. 生成统计
        self.generate_statistics()
        
        # 6. 可视化
        self.visualize()
        
        # 7. 保存结果
        self.save_results()
        
        print("\n" + "=" * 80)
        print("✓ 完成！")
        print("=" * 80)


if __name__ == "__main__":
    # 测试日线和周线
    print("\n" + "="*80)
    print("测试日线模式")
    print("="*80)
    labeler_daily = HS300Labeler(timeframe='daily')
    labeler_daily.run()
    
    print("\n" + "="*80)
    print("测试周线模式")
    print("="*80)
    labeler_weekly = HS300Labeler(timeframe='weekly')
    labeler_weekly.run()
