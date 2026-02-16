"""
行业特征（行业动量、行业轮动）
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union
import logging

from .base import FeatureGenerator, FeatureSpec
from .registry import register_feature

logger = logging.getLogger(__name__)


@register_feature
class IndustryFeatures(FeatureGenerator):
    """行业特征提取器"""

    spec = FeatureSpec(
        name="industry_features",
        input_fields=("date", "stock", "ts_code"),
        description="行业特征（行业动量/行业轮动/行业资金）",
    )

    def __init__(
        self,
        momentum_windows: List[int] = None,
        industry_col: str = "industry_l1",
    ):
        """
        初始化行业特征提取器

        Args:
            momentum_windows: 动量计算窗口（默认 [4, 12, 20] 周）
            industry_col: 行业分类字段名
        """
        self.momentum_windows = momentum_windows or [4, 12, 20]
        self.industry_col = industry_col

    def calculate_industry_momentum_features(
        self,
        df: pd.DataFrame,
        industry_index: Optional[pd.DataFrame] = None,
        industry_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        计算行业动量特征

        Args:
            df: 股票数据DataFrame（需包含行业字段）
            industry_index: 行业指数数据（industry_code, trade_date, close）
            industry_col: 行业字段名

        Returns:
            包含行业动量特征的DataFrame
        """
        df = df.copy()
        industry_col = industry_col or self.industry_col

        # 初始化行业动量字段
        for window in self.momentum_windows:
            df[f'industry_ret_{window}w'] = np.nan
            df[f'industry_vol_{window}w'] = np.nan
        df['industry_momentum_score'] = np.nan
        df['industry_relative_strength'] = np.nan

        # 检查是否有行业分类
        if industry_col not in df.columns:
            logger.warning(f"缺少行业分类字段: {industry_col}，跳过行业动量计算")
            return df

        # 如果有外部行业指数数据，计算行业动量
        if industry_index is not None and not industry_index.empty:
            # 标准化日期字段
            if 'trade_date' in industry_index.columns:
                industry_index = industry_index.rename(columns={'trade_date': 'date'})

            # 标准化行业代码字段
            if 'industry_code' not in industry_index.columns:
                # 尝试其他常见字段名
                for alt_col in ['sw_code', 'industry', 'sw_l1_code']:
                    if alt_col in industry_index.columns:
                        industry_index = industry_index.rename(columns={alt_col: 'industry_code'})
                        break

            if 'industry_code' not in industry_index.columns:
                logger.warning("行业指数数据缺少 industry_code 字段，跳过行业动量计算")
                return df

            # 计算行业动量
            industry_index = industry_index.sort_values(['industry_code', 'date'])
            grp = industry_index.groupby('industry_code')

            for window in self.momentum_windows:
                # 行业收益率
                industry_index[f'industry_ret_{window}w'] = grp['close'].pct_change(window)
                # 行业波动率
                industry_index[f'industry_vol_{window}w'] = grp['close'].pct_change().rolling(window).std()

            # 行业相对强度（相对于市场平均）
            if 'industry_ret_4w' in industry_index.columns:
                market_ret = industry_index.groupby('date')['industry_ret_4w'].transform('mean')
                industry_index['industry_relative_strength'] = industry_index['industry_ret_4w'] - market_ret

            # 行业动量综合评分
            industry_index['industry_momentum_score'] = (
                industry_index.get('industry_ret_4w', 0) * 0.4 +
                industry_index.get('industry_ret_12w', 0) * 0.3 +
                industry_index.get('industry_relative_strength', 0) * 0.3
            )

            # 合并到个股数据
            merge_cols = ['date', 'industry_code', 'close'] + \
                         [f'industry_ret_{w}w' for w in self.momentum_windows] + \
                         [f'industry_vol_{w}w' for w in self.momentum_windows] + \
                         ['industry_momentum_score', 'industry_relative_strength']
            merge_cols = [c for c in merge_cols if c in industry_index.columns]

            # 建立个股行业代码到行业指数代码的映射
            # 假设 industry_col 字段存储的是行业名称或代码
            df = self._merge_industry_features(df, industry_index[merge_cols], industry_col)

        logger.info("行业动量特征计算完成")
        return df

    def _merge_industry_features(
        self,
        stock_df: pd.DataFrame,
        industry_df: pd.DataFrame,
        industry_col: str,
    ) -> pd.DataFrame:
        """将行业特征合并到个股数据"""
        # 尝试建立行业代码映射
        # 如果 industry_col 是行业名称，需要映射到 industry_code

        if 'industry_code' in stock_df.columns:
            # 个股数据已有行业代码，直接合并
            merge_df = stock_df.merge(
                industry_df,
                on=['date', 'industry_code'],
                how='left',
                suffixes=('', '_ind')
            )
        elif industry_col in stock_df.columns:
            # 尝试通过行业名称匹配
            # 假设 industry_df 中有 industry_name 字段
            if 'industry_name' in industry_df.columns:
                stock_df = stock_df.rename(columns={industry_col: 'industry_name'})
                merge_df = stock_df.merge(
                    industry_df,
                    on=['date', 'industry_name'],
                    how='left',
                    suffixes=('', '_ind')
                )
            else:
                # 按行业分组计算平均动量
                # 先计算每个日期每个行业的平均动量
                industry_features = industry_df.groupby(['date', 'industry_code']).agg({
                    f'industry_ret_{w}w': 'first' for w in self.momentum_windows
                }).reset_index()

                # 创建行业名称到代码的映射（简化处理）
                logger.warning("无法直接匹配行业代码，使用行业分组统计")
                merge_df = stock_df.copy()

                # 对于每个行业分组，计算行业内平均特征
                for window in self.momentum_windows:
                    # 计算行业内个股相对收益作为代理
                    merge_df[f'industry_ret_{window}w'] = merge_df.groupby([industry_col, 'date'])['close'].transform(
                        lambda x: x.pct_change(window).mean() if len(x) > 1 else np.nan
                    )
        else:
            merge_df = stock_df.copy()

        return merge_df

    def calculate_industry_relative_features(
        self,
        df: pd.DataFrame,
        industry_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        计算行业内相对特征（行业内排名/相对强度）

        Args:
            df: 股票数据DataFrame
            industry_col: 行业字段名

        Returns:
            包含行业内相对特征的DataFrame
        """
        df = df.copy()
        industry_col = industry_col or self.industry_col

        if industry_col not in df.columns:
            logger.warning(f"缺少行业分类字段: {industry_col}，跳过行业内相对特征计算")
            return df

        df = df.sort_values(['ts_code', 'date'])

        # 行业内收益排名
        for window in [4, 12, 20]:
            if f'ret_{window}w' in df.columns:
                df[f'industry_ret_rank_{window}w'] = df.groupby([industry_col, 'date'])[f'ret_{window}w'].transform(
                    lambda x: x.rank(pct=True)
                )

        # 行业内市值排名
        if 'total_mv' in df.columns:
            df['industry_mv_rank'] = df.groupby([industry_col, 'date'])['total_mv'].transform(
                lambda x: x.rank(pct=True)
            )

        # 行业内相对强度（相对于行业中位数）
        if 'ret_4w' in df.columns:
            df['industry_median_ret'] = df.groupby([industry_col, 'date'])['ret_4w'].transform('median')
            df['relative_to_industry'] = df['ret_4w'] - df['industry_median_ret']

        # 行业内估值相对位置
        if 'pe_ttm' in df.columns:
            df['industry_pe_rank'] = df.groupby([industry_col, 'date'])['pe_ttm'].transform(
                lambda x: x.rank(pct=True)
            )

        logger.info("行业内相对特征计算完成")
        return df

    def calculate_industry_flow_features(
        self,
        df: pd.DataFrame,
        industry_northbound: Optional[pd.DataFrame] = None,
        industry_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        计算行业资金特征

        Args:
            df: 股票数据DataFrame
            industry_northbound: 行业北向资金数据
            industry_col: 行业字段名

        Returns:
            包含行业资金特征的DataFrame
        """
        df = df.copy()
        industry_col = industry_col or self.industry_col

        # 初始化行业资金字段
        df['industry_northbound_ratio'] = np.nan
        df['industry_northbound_change'] = np.nan

        if industry_northbound is not None and not industry_northbound.empty:
            # 标准化日期字段
            if 'trade_date' in industry_northbound.columns:
                industry_northbound = industry_northbound.rename(columns={'trade_date': 'date'})

            # 标准化行业代码字段
            for alt_col in ['industry_code', 'sw_code', 'industry']:
                if alt_col in industry_northbound.columns:
                    industry_northbound = industry_northbound.rename(columns={alt_col: 'industry_code'})
                    break

            # 合并行业北向资金数据
            if 'industry_code' in industry_northbound.columns:
                merge_cols = ['date', 'industry_code', 'northbound_ratio', 'northbound_change']
                merge_cols = [c for c in merge_cols if c in industry_northbound.columns]

                if len(merge_cols) >= 3:
                    df = df.merge(
                        industry_northbound[merge_cols],
                        on=['date', 'industry_code'],
                        how='left',
                        suffixes=('', '_ind_nb')
                    )

        logger.info("行业资金特征计算完成")
        return df

    def calculate_all_features(
        self,
        df: pd.DataFrame,
        industry_index: Optional[pd.DataFrame] = None,
        industry_northbound: Optional[pd.DataFrame] = None,
        industry_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        计算所有行业特征

        Args:
            df: 股票数据DataFrame
            industry_index: 行业指数数据
            industry_northbound: 行业北向资金数据
            industry_col: 行业字段名

        Returns:
            包含所有特征的DataFrame
        """
        logger.info("开始计算行业特征...")

        # 检查必要的列
        required_cols = ['ts_code', 'date']
        for col in required_cols:
            if col not in df.columns:
                if col == 'date' and 'trade_date' in df.columns:
                    df['date'] = df['trade_date']
                else:
                    raise ValueError(f"缺少必要的列: {col}")

        # 计算各类特征
        df = self.calculate_industry_momentum_features(df, industry_index, industry_col)
        df = self.calculate_industry_relative_features(df, industry_col)
        df = self.calculate_industry_flow_features(df, industry_northbound, industry_col)

        logger.info(f"行业特征计算完成")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        return self.calculate_all_features(df)

    def get_feature_names(self) -> List[str]:
        """返回生成的特征名称列表"""
        return [
            # 行业动量特征
            'industry_ret_4w',
            'industry_ret_12w',
            'industry_ret_20w',
            'industry_vol_4w',
            'industry_vol_12w',
            'industry_momentum_score',
            'industry_relative_strength',
            # 行业内相对特征
            'industry_ret_rank_4w',
            'industry_ret_rank_12w',
            'industry_mv_rank',
            'relative_to_industry',
            'industry_pe_rank',
            # 行业资金特征
            'industry_northbound_ratio',
            'industry_northbound_change',
        ]


if __name__ == "__main__":
    # 测试行业特征提取
    logging.basicConfig(level=logging.INFO)

    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='W-FRI')  # 周频
    stocks = ['600519.SH', '000858.SZ', '000001.SZ', '600036.SH']
    industries = {'600519.SH': '食品饮料', '000858.SZ': '食品饮料',
                  '000001.SZ': '银行', '600036.SH': '银行'}

    data = []
    for stock in stocks:
        for i, date in enumerate(dates):
            data.append({
                'ts_code': stock,
                'date': date.strftime('%Y%m%d'),
                'stock': stock,
                'close': 100 + np.cumsum(np.random.randn(len(dates)))[i] * 0.05,
                'industry_l1': industries.get(stock, '其他'),
                'ret_4w': np.random.uniform(-0.1, 0.1),
                'ret_12w': np.random.uniform(-0.2, 0.2),
                'total_mv': np.random.uniform(1e9, 1e11),
            })

    df = pd.DataFrame(data)

    # 模拟行业指数数据
    industry_index = pd.DataFrame({
        'industry_code': ['食品饮料'] * len(dates) + ['银行'] * len(dates),
        'date': [d.strftime('%Y%m%d') for d in dates] * 2,
        'close': np.concatenate([
            3000 + np.cumsum(np.random.randn(len(dates)) * 10),
            2000 + np.cumsum(np.random.randn(len(dates)) * 8)
        ]),
    })

    feature_extractor = IndustryFeatures()
    df_with_features = feature_extractor.calculate_all_features(
        df,
        industry_index=industry_index,
        industry_col='industry_l1'
    )

    print(f"\n生成的特征列:")
    print([c for c in df_with_features.columns if c in feature_extractor.get_feature_names()])

    print(f"\n数据预览:")
    print(df_with_features[['ts_code', 'date', 'industry_l1', 'industry_ret_4w', 'industry_momentum_score']].tail(10))
