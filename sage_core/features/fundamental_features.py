"""
基本面特征（估值、质量）
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging

from .base import FeatureGenerator, FeatureSpec
from .registry import register_feature

logger = logging.getLogger(__name__)


@register_feature
class FundamentalFeatures(FeatureGenerator):
    """基本面特征提取器（估值/质量）"""

    spec = FeatureSpec(
        name="fundamental_features",
        input_fields=("date", "stock", "ts_code"),
        description="基本面特征（估值/质量/财务指标）",
    )

    def __init__(
        self,
        valuation_windows: List[int] = None,
        quality_windows: List[int] = None,
    ):
        """
        初始化基本面特征提取器

        Args:
            valuation_windows: 估值分位数计算窗口（默认 [252, 504] 即1年/2年）
            quality_windows: 质量指标滚动窗口（默认 [4, 12] 即4周/12周）
        """
        self.valuation_windows = valuation_windows or [252, 504]
        self.quality_windows = quality_windows or [4, 12]

    def calculate_valuation_features(
        self,
        df: pd.DataFrame,
        daily_basic: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算估值特征

        Args:
            df: 股票数据DataFrame（需包含 ts_code, date）
            daily_basic: 每日基本面数据（pe_ttm, pb 等）

        Returns:
            包含估值特征的DataFrame
        """
        df = df.copy()

        # 如果有外部 daily_basic 数据，合并进来
        if daily_basic is not None and not daily_basic.empty:
            # 确保日期格式一致
            if 'trade_date' in daily_basic.columns:
                daily_basic = daily_basic.rename(columns={'trade_date': 'date'})

            # 合并基本面数据
            merge_cols = ['ts_code', 'date']
            available_cols = [c for c in merge_cols if c in daily_basic.columns]
            if len(available_cols) >= 2:
                # 只保留需要的估值字段
                val_cols = ['ts_code', 'date', 'pe_ttm', 'pb', 'pe', 'total_mv', 'circ_mv']
                val_cols = [c for c in val_cols if c in daily_basic.columns]
                daily_basic_subset = daily_basic[val_cols].copy()
                df = df.merge(daily_basic_subset, on=['ts_code', 'date'], how='left')

        # 计算估值分位数（需要历史数据）
        df = df.sort_values(['ts_code', 'date'])

        # PE 分位数（历史百分位）
        if 'pe_ttm' in df.columns:
            for window in self.valuation_windows:
                df[f'pe_percentile_{window}d'] = df.groupby('ts_code')['pe_ttm'].transform(
                    lambda x: x.rolling(window, min_periods=int(window * 0.5)).rank(pct=True)
                )
            # 默认使用 252 日分位数
            df['pe_percentile'] = df.get('pe_percentile_252d', df.get('pe_percentile_504d', np.nan))

        # PB 分位数（历史百分位）
        if 'pb' in df.columns:
            for window in self.valuation_windows:
                df[f'pb_percentile_{window}d'] = df.groupby('ts_code')['pb'].transform(
                    lambda x: x.rolling(window, min_periods=int(window * 0.5)).rank(pct=True)
                )
            df['pb_percentile'] = df.get('pb_percentile_252d', df.get('pb_percentile_504d', np.nan))

        # PE/PB 合成评分（低估值加分）
        if 'pe_ttm' in df.columns and 'pb' in df.columns:
            # 标准化后取负（低估值 = 高评分）
            df['pe_zscore'] = df.groupby('ts_code')['pe_ttm'].transform(
                lambda x: (x - x.rolling(252, min_periods=60).mean()) / x.rolling(252, min_periods=60).std()
            )
            df['pb_zscore'] = df.groupby('ts_code')['pb'].transform(
                lambda x: (x - x.rolling(252, min_periods=60).mean()) / x.rolling(252, min_periods=60).std()
            )
            # 估值评分（越低越好）
            df['valuation_score'] = -df['pe_zscore'] * 0.5 - df['pb_zscore'] * 0.5

        logger.info("估值特征计算完成")
        return df

    def calculate_quality_features(
        self,
        df: pd.DataFrame,
        fina_indicator: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算质量特征

        Args:
            df: 股票数据DataFrame
            fina_indicator: 财务指标数据（roe, gross_margin, roic 等）

        Returns:
            包含质量特征的DataFrame
        """
        df = df.copy()

        # 如果有外部财务指标数据，合并进来
        if fina_indicator is not None and not fina_indicator.empty:
            # 确保日期格式一致
            if 'ann_date' in fina_indicator.columns:
                fina_indicator = fina_indicator.rename(columns={'ann_date': 'date'})
            if 'end_date' in fina_indicator.columns:
                fina_indicator = fina_indicator.drop(columns=['end_date'], errors='ignore')

            # 合并财务指标
            merge_cols = ['ts_code', 'date']
            available_cols = [c for c in merge_cols if c in fina_indicator.columns]
            if len(available_cols) >= 2:
                # 只保留需要的质量字段
                quality_cols = ['ts_code', 'date', 'roe', 'roe_dt', 'grossprofit_margin',
                               'netprofit_margin', 'roic', 'ocfps', 'debt_to_assets']
                quality_cols = [c for c in quality_cols if c in fina_indicator.columns]
                fina_subset = fina_indicator[quality_cols].copy()
                df = df.merge(fina_subset, on=['ts_code', 'date'], how='left')

        # ROE 处理（优先使用 roe_dt，即扣非ROE）
        if 'roe_dt' in df.columns:
            df['roe'] = df['roe'].fillna(df['roe_dt'])
        elif 'roe' not in df.columns:
            df['roe'] = np.nan

        # gross_margin 处理
        if 'grossprofit_margin' in df.columns:
            df['gross_margin'] = df['grossprofit_margin']
        elif 'gross_margin' not in df.columns:
            df['gross_margin'] = np.nan

        # ROIC 处理
        if 'roic' not in df.columns:
            df['roic'] = np.nan

        # 计算质量变化（财务指标加速度）
        df = df.sort_values(['ts_code', 'date'])
        for col in ['roe', 'gross_margin', 'roic']:
            if col in df.columns:
                df[f'{col}_change'] = df.groupby('ts_code')[col].diff()
                df[f'{col}_ma4'] = df.groupby('ts_code')[col].transform(
                    lambda x: x.rolling(4, min_periods=1).mean()
                )

        # 综合质量评分
        quality_components = []
        for col in ['roe', 'gross_margin', 'roic']:
            if col in df.columns:
                # 标准化
                df[f'{col}_zscore'] = df.groupby('ts_code')[col].transform(
                    lambda x: (x - x.rolling(252, min_periods=60).mean()) / x.rolling(252, min_periods=60).std()
                )
                quality_components.append(f'{col}_zscore')

        if quality_components:
            weights = {'roe_zscore': 0.4, 'gross_margin_zscore': 0.35, 'roic_zscore': 0.25}
            df['quality_score'] = sum(
                df.get(c, 0) * weights.get(c, 1/len(quality_components))
                for c in quality_components
            )
        else:
            df['quality_score'] = np.nan

        logger.info("质量特征计算完成")
        return df

    def calculate_all_features(
        self,
        df: pd.DataFrame,
        daily_basic: Optional[pd.DataFrame] = None,
        fina_indicator: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算所有基本面特征

        Args:
            df: 股票数据DataFrame
            daily_basic: 每日基本面数据
            fina_indicator: 财务指标数据

        Returns:
            包含所有特征的DataFrame
        """
        logger.info("开始计算基本面特征...")

        # 检查必要的列
        required_cols = ['ts_code', 'date']
        for col in required_cols:
            if col not in df.columns:
                if col == 'date' and 'trade_date' in df.columns:
                    df['date'] = df['trade_date']
                else:
                    raise ValueError(f"缺少必要的列: {col}")

        # 计算各类特征
        df = self.calculate_valuation_features(df, daily_basic)
        df = self.calculate_quality_features(df, fina_indicator)

        # 统计新增特征
        base_cols = set(df.columns)
        new_features = [c for c in df.columns if c not in base_cols or
                        c in ['pe_ttm', 'pb', 'pe_percentile', 'pb_percentile',
                              'roe', 'gross_margin', 'roic', 'quality_score', 'valuation_score']]

        logger.info(f"基本面特征计算完成，新增特征: {len(new_features)}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        return self.calculate_all_features(df)

    def get_feature_names(self) -> List[str]:
        """返回生成的特征名称列表"""
        return [
            # 估值特征
            'pe_ttm', 'pb', 'pe_percentile', 'pb_percentile',
            'pe_zscore', 'pb_zscore', 'valuation_score',
            # 质量特征
            'roe', 'gross_margin', 'roic',
            'roe_change', 'gross_margin_change', 'roic_change',
            'roe_ma4', 'gross_margin_ma4', 'roic_ma4',
            'roe_zscore', 'gross_margin_zscore', 'roic_zscore',
            'quality_score',
        ]


if __name__ == "__main__":
    # 测试基本面特征提取
    logging.basicConfig(level=logging.INFO)

    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    stocks = ['600519.SH', '000858.SZ']

    data = []
    for stock in stocks:
        for i, date in enumerate(dates):
            data.append({
                'ts_code': stock,
                'date': date.strftime('%Y%m%d'),
                'stock': stock,
                'close': 100 + np.cumsum(np.random.randn(len(dates)))[i] * 0.01,
                'pe_ttm': np.random.uniform(10, 50),
                'pb': np.random.uniform(1, 5),
                'roe': np.random.uniform(0.05, 0.25),
                'gross_margin': np.random.uniform(0.2, 0.5),
                'roic': np.random.uniform(0.05, 0.2),
            })

    df = pd.DataFrame(data)

    feature_extractor = FundamentalFeatures()
    df_with_features = feature_extractor.calculate_all_features(df)

    print(f"\n生成的特征列:")
    print([c for c in df_with_features.columns if c in feature_extractor.get_feature_names()])

    print(f"\n数据预览:")
    print(df_with_features[['ts_code', 'date', 'pe_percentile', 'pb_percentile', 'quality_score']].tail(10))
