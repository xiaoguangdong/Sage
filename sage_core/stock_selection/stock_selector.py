from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SelectionConfig:
    model_type: str = "rule"  # rule / lgbm / xgb

    # 标签配置
    label_horizons: Tuple[int, ...] = (120,)
    label_weights: Tuple[float, ...] = (1.0,)
    risk_adjusted: bool = True
    vol_window: int = 20
    industry_rank: bool = True
    industry_col: Optional[str] = "industry_l1"
    rank_mode: str = "industry"  # industry / market / auto
    
    # P0: 行业中性化标签（防过拟合核心）
    label_neutralized: bool = True  # 标签 = 个股收益 - 行业平均收益
    market_neutralized: bool = False  # 标签 = 个股收益 - 市场平均收益
    
    # P2: 因子 IC 筛选（防过拟合）
    ic_filter_enabled: bool = True   # 启用 IC 筛选
    ic_threshold: float = 0.02       # IC 阈值（|IC| > 0.02）
    ic_ir_threshold: float = 0.3     # IC_IR 阈值
    max_corr_threshold: float = 0.7  # 共线性阈值

    # 列名配置
    date_col: str = "trade_date"
    code_col: str = "ts_code"
    price_col: str = "close"

    # 特征
    feature_cols: Optional[Tuple[str, ...]] = None
    rule_weights: Optional[Dict[str, float]] = None
    min_feature_count: int = 8
    max_feature_count: int = 30

    # 模型参数
    lgbm_params: Dict[str, object] = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.02,        # ↓ 降低学习率
        "num_leaves": 15,             # ↓ 从31降到15，降低复杂度
        "max_depth": 4,               # ↓ 限制树深
        "min_data_in_leaf": 500,      # ↑ 防过拟合核心参数
        "feature_fraction": 0.6,      # ↓ 特征采样
        "bagging_fraction": 0.6,      # ↓ 数据采样
        "bagging_freq": 5,
        "lambda_l1": 1.0,             # L1 正则
        "lambda_l2": 10.0,            # L2 正则（增强）
        "min_gain_to_split": 0.01,    # 分裂最小增益
        "verbosity": -1,
    })
    xgb_params: Dict[str, object] = field(default_factory=lambda: {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 200,
    })

    def normalized_label_weights(self) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
        horizons = self.label_horizons or (20,)
        weights = self.label_weights or ()
        if len(weights) != len(horizons):
            weights = tuple([1.0 / len(horizons)] * len(horizons))
        total = float(sum(weights))
        if total <= 0:
            weights = tuple([1.0 / len(horizons)] * len(horizons))
        else:
            weights = tuple(w / total for w in weights)
        return horizons, weights


class StockSelector:
    def __init__(self, config: Optional[SelectionConfig] = None):
        self.config = config or SelectionConfig()
        self.model = None
        self.feature_cols: Optional[List[str]] = None
        self.feature_medians: Dict[str, float] = {}
        self.is_trained = False

    # -----------------------------
    # 公共入口
    # -----------------------------
    def fit(self, df: pd.DataFrame) -> "StockSelector":
        df_features = self.prepare_features(df)
        labels = self.build_labels(df_features)

        train_df = df_features.copy()
        train_df["label"] = labels
        train_df = train_df.dropna(subset=["label"])

        feature_cols = self._infer_feature_cols(train_df)
        train_df = self._coerce_numeric_features(train_df, feature_cols)
        feature_cols = self._select_usable_features(train_df, feature_cols)
        if not feature_cols:
            raise ValueError("训练特征不可用，请检查特征覆盖率")
        train_df = self._fill_missing_features(train_df, feature_cols, fit=True)

        if train_df.empty:
            raise ValueError("训练数据为空，请检查输入数据与标签窗口")

        # P2: 因子 IC 筛选（防过拟合）
        feature_cols = self._filter_features_by_ic(train_df, list(feature_cols))
        if not feature_cols:
            raise ValueError("IC筛选后无有效因子，请降低 ic_threshold")
        
        self.feature_cols = feature_cols

        if self.config.model_type == "rule":
            self._fit_rule(train_df, feature_cols)
        elif self.config.model_type == "lgbm":
            self._fit_lgbm(train_df, feature_cols)
        elif self.config.model_type == "xgb":
            X = train_df[feature_cols].values
            y = train_df["label"].values
            self._fit_xgb(X, y)
        else:
            raise ValueError(f"未知模型类型: {self.config.model_type}")

        self.is_trained = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("模型尚未训练")

        df_features = self.prepare_features(df)
        feature_cols = self.feature_cols or self._infer_feature_cols(df_features)
        missing_cols = [c for c in feature_cols if c not in df_features.columns]
        for col in missing_cols:
            df_features[col] = np.nan
        df_features = self._coerce_numeric_features(df_features, feature_cols)
        df_features = self._fill_missing_features(df_features, feature_cols, fit=False)

        if df_features.empty:
            raise ValueError("预测数据为空，请检查输入数据")

        if self.config.model_type == "rule":
            scores = self._score_rule(df_features, feature_cols)
        else:
            scores = self.model.predict(df_features[feature_cols].values)

        df_result = df_features.copy()
        df_result["score"] = scores
        df_result["rank"] = self._rank_by_date(df_result)
        df_result["confidence"] = self._confidence_by_date(df_result)
        return df_result

    def select_top(self, df: pd.DataFrame, top_n: int = 10, trade_date: Optional[str] = None) -> pd.DataFrame:
        df_pred = self.predict(df)
        date_col = self.config.date_col
        if trade_date is not None and date_col in df_pred.columns:
            df_pred = df_pred[df_pred[date_col] == trade_date]
            return df_pred.sort_values("score", ascending=False).head(top_n)

        if date_col in df_pred.columns:
            return df_pred.sort_values([date_col, "score"], ascending=[True, False]).groupby(date_col).head(top_n)

        return df_pred.sort_values("score", ascending=False).head(top_n)

    # -----------------------------
    # 特征与标签
    # -----------------------------
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self._validate_columns(df, [self.config.date_col, self.config.code_col, self.config.price_col])
        df = df.sort_values([self.config.code_col, self.config.date_col])

        if self.config.feature_cols:
            return df

        code_col = self.config.code_col
        price_col = self.config.price_col
        date_col = self.config.date_col

        group = df.groupby(code_col, group_keys=False)
        returns = group[price_col].pct_change()
        df["ret_1d"] = returns

        df["ret_5d"] = group[price_col].pct_change(5)
        df["ret_10d"] = group[price_col].pct_change(10)
        df["ret_20d"] = group[price_col].pct_change(20)
        df["ret_60d"] = group[price_col].pct_change(60)
        df["ma_10_ratio"] = df[price_col] / group[price_col].transform(lambda s: s.rolling(10).mean())
        df["ma_20_ratio"] = df[price_col] / group[price_col].transform(lambda s: s.rolling(20).mean())
        df["ma_60_ratio"] = df[price_col] / group[price_col].transform(lambda s: s.rolling(60).mean())
        df["vol_20d"] = returns.groupby(df[code_col]).rolling(20).std().reset_index(level=0, drop=True)
        df["downside_vol_20d"] = returns.where(returns < 0).groupby(df[code_col]).rolling(20).std().reset_index(level=0, drop=True)
        rolling_dd = group[price_col].transform(lambda s: s / s.rolling(60).max() - 1)
        df["max_drawdown_60d"] = rolling_dd.groupby(df[code_col]).rolling(60).min().reset_index(level=0, drop=True)

        if "turnover" in df.columns:
            df["turnover_20d_mean"] = group["turnover"].transform(lambda s: s.rolling(20).mean())
            df["turnover_20d_std"] = group["turnover"].transform(lambda s: s.rolling(20).std())
            df["turnover_ratio_20d"] = df["turnover"] / df["turnover_20d_mean"].replace(0, np.nan)
            df["liquidity_stability"] = -df["turnover_20d_std"]

        if "amount" in df.columns:
            df["amount_20d_mean"] = group["amount"].transform(lambda s: s.rolling(20).mean())
            df["amount_ratio_20d"] = df["amount"] / df["amount_20d_mean"].replace(0, np.nan)

        if self.config.industry_col and self.config.industry_col in df.columns:
            df["industry_ret_20d"] = df.groupby([date_col, self.config.industry_col])["ret_20d"].transform("mean")
            df["industry_ret_60d"] = df.groupby([date_col, self.config.industry_col])["ret_60d"].transform("mean")

        pe_col = self._first_existing_column(df, ["pe_ttm", "pe", "peTTM"])
        pb_col = self._first_existing_column(df, ["pb", "pbMRQ"])
        roe_col = self._first_existing_column(df, ["roe", "roe_dt", "roe_ttm"])
        roic_col = self._first_existing_column(df, ["roic", "roic_ttm"])
        gross_margin_col = self._first_existing_column(df, ["gross_margin", "grossprofit_margin"])
        profit_yoy_col = self._first_existing_column(df, ["netprofit_yoy", "profit_yoy", "dt_netprofit_yoy"])
        debt_assets_col = self._first_existing_column(df, ["debt_to_assets", "debt_to_asset"])
        ocf_profit_col = self._first_existing_column(df, ["ocf_to_profit", "ocfps"])
        nb_hold_col = self._first_existing_column(df, ["northbound_hold_ratio", "hold_ratio"])
        nb_flow_col = self._first_existing_column(df, ["northbound_net_flow", "northbound_net_flow_20d", "net_flow"])
        beta_col = self._first_existing_column(df, ["beta_120d", "beta"])

        if pe_col and pe_col != "pe_ttm":
            df["pe_ttm"] = pd.to_numeric(df[pe_col], errors="coerce")
        if pb_col and pb_col != "pb":
            df["pb"] = pd.to_numeric(df[pb_col], errors="coerce")
        if roe_col and roe_col != "roe":
            df["roe"] = pd.to_numeric(df[roe_col], errors="coerce")
        if roic_col and roic_col != "roic":
            df["roic"] = pd.to_numeric(df[roic_col], errors="coerce")
        if gross_margin_col and gross_margin_col != "gross_margin":
            df["gross_margin"] = pd.to_numeric(df[gross_margin_col], errors="coerce")
        if profit_yoy_col and profit_yoy_col != "netprofit_yoy":
            df["netprofit_yoy"] = pd.to_numeric(df[profit_yoy_col], errors="coerce")
        if debt_assets_col and debt_assets_col != "debt_to_assets":
            df["debt_to_assets"] = pd.to_numeric(df[debt_assets_col], errors="coerce")
        if ocf_profit_col and ocf_profit_col != "ocf_to_profit":
            df["ocf_to_profit"] = pd.to_numeric(df[ocf_profit_col], errors="coerce")
        if nb_hold_col and nb_hold_col != "northbound_hold_ratio":
            df["northbound_hold_ratio"] = pd.to_numeric(df[nb_hold_col], errors="coerce")
        if nb_flow_col and nb_flow_col != "northbound_net_flow_20d":
            flow = pd.to_numeric(df[nb_flow_col], errors="coerce")
            df["northbound_net_flow_20d"] = flow.groupby(df[code_col]).rolling(20).mean().reset_index(level=0, drop=True)
        if beta_col and beta_col != "beta_120d":
            df["beta_120d"] = pd.to_numeric(df[beta_col], errors="coerce")

        if "pe_ttm" in df.columns:
            df["pe_percentile"] = df.groupby(date_col)["pe_ttm"].rank(pct=True)
        if "pb" in df.columns:
            df["pb_percentile"] = df.groupby(date_col)["pb"].rank(pct=True)

        df["trade_date"] = df[date_col]
        return df

    def build_labels(self, df: pd.DataFrame) -> pd.Series:
        cfg = self.config
        self._validate_columns(df, [cfg.date_col, cfg.code_col, cfg.price_col])

        horizons, weights = cfg.normalized_label_weights()
        code_col = cfg.code_col
        date_col = cfg.date_col
        price_col = cfg.price_col

        group = df.groupby(code_col, group_keys=False)
        future_returns: List[Tuple[pd.Series, float]] = []

        base_returns = group[price_col].pct_change()
        vol = base_returns.groupby(df[code_col]).rolling(cfg.vol_window).std().reset_index(level=0, drop=True)
        vol = vol.replace(0, np.nan)

        for horizon, weight in zip(horizons, weights):
            future_ret = group[price_col].shift(-horizon) / df[price_col] - 1
            
            # P0: 行业中性化（防过拟合核心）
            if cfg.label_neutralized and cfg.industry_col and cfg.industry_col in df.columns:
                # 计算行业平均收益（按日期+行业分组）
                df_temp = df.copy()
                df_temp['_future_ret'] = future_ret.values if hasattr(future_ret, 'values') else future_ret
                industry_mean = df_temp.groupby([date_col, cfg.industry_col])['_future_ret'].transform('mean')
                # 行业中性化：个股收益 - 行业平均收益
                future_ret = future_ret - industry_mean
            elif cfg.market_neutralized:
                # 市场中性化
                df_temp = df.copy()
                df_temp['_future_ret'] = future_ret.values if hasattr(future_ret, 'values') else future_ret
                market_mean = df_temp.groupby(date_col)['_future_ret'].transform('mean')
                future_ret = future_ret - market_mean
            
            if cfg.risk_adjusted:
                label = future_ret / (vol + 1e-8)
            else:
                label = future_ret

            if cfg.industry_rank:
                mode = (cfg.rank_mode or "auto").lower()
                if mode == "industry":
                    if cfg.industry_col and cfg.industry_col in df.columns:
                        label = label.groupby([df[date_col], df[cfg.industry_col]]).rank(pct=True)
                    else:
                        logger.warning("industry_col 未配置或缺失，已回退到全市场排名")
                        label = label.groupby(df[date_col]).rank(pct=True)
                elif mode == "market":
                    label = label.groupby(df[date_col]).rank(pct=True)
                else:  # auto
                    if cfg.industry_col and cfg.industry_col in df.columns:
                        label = label.groupby([df[date_col], df[cfg.industry_col]]).rank(pct=True)
                    else:
                        label = label.groupby(df[date_col]).rank(pct=True)
            if label.notna().any():
                future_returns.append((label, weight))

        if not future_returns:
            return pd.Series(index=df.index, dtype="float64")

        total_weight = float(sum(weight for _, weight in future_returns)) or 1.0
        mask_any = None
        combined = None
        for label, weight in future_returns:
            scaled = (weight / total_weight) * label.fillna(0)
            if combined is None:
                combined = scaled
                mask_any = label.notna()
            else:
                combined = combined + scaled
                mask_any = mask_any | label.notna()

        combined = combined.astype("float64")
        combined[~mask_any] = np.nan
        return combined

    # -----------------------------
    # 模型训练
    # -----------------------------
    def _fit_rule(self, train_df: pd.DataFrame, feature_cols: Sequence[str]) -> None:
        if self.config.rule_weights:
            weights = self.config.rule_weights
        else:
            weights = {col: 1.0 for col in feature_cols}

        total = float(sum(weights.values())) or 1.0
        self.rule_weights = {k: v / total for k, v in weights.items() if k in feature_cols}

    def _score_rule(self, df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
        weights = getattr(self, "rule_weights", None) or {col: 1.0 for col in feature_cols}
        total = float(sum(weights.values())) or 1.0
        weights = {k: v / total for k, v in weights.items() if k in feature_cols}

        zscores = []
        for col in feature_cols:
            series = df[col]
            mean = series.mean()
            std = series.std()
            if std == 0 or np.isnan(std):
                z = pd.Series(0.0, index=series.index)
            else:
                z = (series - mean) / std
            zscores.append(z * weights.get(col, 0.0))

        score = sum(zscores)
        return score.to_numpy()

    def _fit_lgbm(self, train_df: pd.DataFrame, feature_cols: Sequence[str]) -> None:
        try:
            import lightgbm as lgb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("未安装 lightgbm，请先安装依赖") from exc

        params = dict(self.config.lgbm_params)
        date_col = self.config.date_col
        code_col = self.config.code_col
        objective = str(params.get("objective", "regression")).lower()
        rank_objectives = {"lambdarank", "rank_xendcg", "xendcg"}
        use_ranking_group = objective in rank_objectives and date_col in train_df.columns

        training_frame = train_df
        if use_ranking_group:
            sort_cols = [date_col]
            if code_col in train_df.columns:
                sort_cols.append(code_col)
            training_frame = train_df.sort_values(sort_cols).reset_index(drop=True)

        X = training_frame[list(feature_cols)].values
        y = training_frame["label"].values

        if use_ranking_group:
            y = self._to_lgbm_rank_labels(y)
            groups = (
                training_frame.groupby(date_col, sort=True)
                .size()
                .astype(int)
                .tolist()
            )
            dataset = lgb.Dataset(X, label=y, group=groups)
        else:
            dataset = lgb.Dataset(X, label=y)

        self.model = lgb.train(
            params,
            dataset,
            num_boost_round=int(params.get("num_boost_round", 500)),
            valid_sets=[dataset],
            callbacks=[
                lgb.log_evaluation(period=50),
                lgb.early_stopping(stopping_rounds=int(params.get("early_stopping_rounds", 50)), verbose=True),
            ],
        )

    def _to_lgbm_rank_labels(self, y: np.ndarray) -> np.ndarray:
        series = pd.Series(y, dtype="float64").replace([np.inf, -np.inf], np.nan)
        if series.notna().sum() == 0:
            return np.zeros(len(series), dtype=np.int32)
        ranked = series.rank(pct=True, method="first").fillna(0.0)
        labels = np.floor(ranked.to_numpy() * 30.0).astype(np.int32)
        return np.clip(labels, 0, 30).astype(np.int32)

    def _fit_xgb(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import xgboost as xgb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("未安装 xgboost，请先安装依赖") from exc

        params = dict(self.config.xgb_params)
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        self.model = model

    # -----------------------------
    # 工具方法
    # -----------------------------
    def _infer_feature_cols(self, df: pd.DataFrame) -> List[str]:
        if self.config.feature_cols:
            return list(self.config.feature_cols)

        preferred = [
            "roe",
            "roic",
            "gross_margin",
            "netprofit_yoy",
            "debt_to_assets",
            "ocf_to_profit",
            "pe_percentile",
            "pb_percentile",
            "ret_5d",
            "ret_10d",
            "ret_20d",
            "ret_60d",
            "industry_ret_20d",
            "industry_ret_60d",
            "ma_20_ratio",
            "ma_60_ratio",
            "northbound_hold_ratio",
            "northbound_net_flow_20d",
            "turnover_ratio_20d",
            "amount_ratio_20d",
            "vol_20d",
            "downside_vol_20d",
            "max_drawdown_60d",
            "beta_120d",
            "liquidity_stability",
        ]
        exclude = {
            self.config.date_col,
            self.config.code_col,
            self.config.price_col,
            "label",
        }
        if self.config.industry_col:
            exclude.add(self.config.industry_col)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        leak_patterns = ("label", "future", "target", "fwd", "next_")
        selected = [c for c in preferred if c in numeric_cols and c not in exclude]
        selected = [c for c in selected if not any(p in c.lower() for p in leak_patterns)]

        if len(selected) < max(1, int(self.config.min_feature_count)):
            fallback = [c for c in numeric_cols if c not in exclude]
            fallback = [c for c in fallback if not any(p in c.lower() for p in leak_patterns)]
            selected = fallback

        max_count = int(self.config.max_feature_count) if self.config.max_feature_count else 30
        if max_count > 0:
            selected = selected[:max_count]
        return selected

    def _validate_columns(self, df: pd.DataFrame, columns: Iterable[str]) -> None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必要字段: {missing}")

    def _rank_by_date(self, df: pd.DataFrame) -> pd.Series:
        date_col = self.config.date_col
        if date_col in df.columns:
            return df.groupby(date_col)["score"].rank(ascending=False, method="first")
        return df["score"].rank(ascending=False, method="first")

    def _confidence_by_date(self, df: pd.DataFrame) -> pd.Series:
        date_col = self.config.date_col
        if date_col in df.columns:
            return df.groupby(date_col)["score"].rank(pct=True)
        return df["score"].rank(pct=True)

    def _first_existing_column(self, df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _coerce_numeric_features(self, df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _select_usable_features(self, df: pd.DataFrame, feature_cols: Sequence[str]) -> List[str]:
        usable: List[str] = []
        for col in feature_cols:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            coverage = float(series.notna().mean()) if len(series) else 0.0
            if coverage < 0.15:
                continue
            if series.dropna().nunique() <= 1:
                continue
            usable.append(col)
        return usable

    def _filter_features_by_ic(self, df: pd.DataFrame, feature_cols: List[str], label_col: str = "label") -> List[str]:
        """
        P2: 因子 IC 筛选（防过拟合）
        
        筛选条件：
        1. |IC| > ic_threshold
        2. IC_IR > ic_ir_threshold
        3. 共线性过滤（|corr| < max_corr_threshold）
        """
        if not self.config.ic_filter_enabled:
            return feature_cols
        
        date_col = self.config.date_col
        if label_col not in df.columns:
            return feature_cols
        
        # 1. 计算 IC（按日期截面）
        ic_series = {}
        for col in feature_cols:
            if col not in df.columns:
                continue
            ic_list = []
            for date, group in df.groupby(date_col):
                if len(group) < 30:  # 样本太少跳过
                    continue
                valid = group[[col, label_col]].dropna()
                if len(valid) < 20:
                    continue
                ic = valid[col].corr(valid[label_col], method="spearman")
                if not np.isnan(ic):
                    ic_list.append(ic)
            if ic_list:
                ic_series[col] = pd.Series(ic_list)
        
        if not ic_series:
            return feature_cols
        
        # 2. IC 筛选
        selected = []
        ic_stats = {}
        for col, ics in ic_series.items():
            mean_ic = ics.mean()
            ic_ir = ics.mean() / (ics.std() + 1e-8) if ics.std() > 0 else 0
            ic_stats[col] = {"mean_ic": mean_ic, "ic_ir": ic_ir}
            
            # 筛选条件
            if abs(mean_ic) >= self.config.ic_threshold:
                if ic_ir >= self.config.ic_ir_threshold:
                    selected.append(col)
        
        logger.info(f"IC筛选: {len(feature_cols)} -> {len(selected)} 因子")
        
        # 3. 共线性过滤
        if len(selected) > 1:
            corr_matrix = df[selected].corr().abs()
            final_selected = []
            for col in selected:
                # 检查与已选因子的相关性
                high_corr = False
                for sel in final_selected:
                    if corr_matrix.loc[col, sel] > self.config.max_corr_threshold:
                        high_corr = True
                        # 保留 IC 更高的因子
                        if abs(ic_stats[col]["mean_ic"]) > abs(ic_stats[sel]["mean_ic"]):
                            final_selected.remove(sel)
                            final_selected.append(col)
                        break
                if not high_corr:
                    final_selected.append(col)
            selected = final_selected
        
        logger.info(f"共线性过滤: {len(selected)} 因子")
        
        # 保存 IC 统计信息
        self.feature_ic_stats = ic_stats
        
        return selected if selected else feature_cols

    def _fill_missing_features(self, df: pd.DataFrame, feature_cols: Sequence[str], fit: bool) -> pd.DataFrame:
        date_col = self.config.date_col
        if fit:
            self.feature_medians = {}

        if date_col in df.columns:
            for col in feature_cols:
                df[col] = df.groupby(date_col)[col].transform(lambda s: s.fillna(s.median()))

        for col in feature_cols:
            if fit:
                median = float(df[col].median()) if df[col].notna().any() else 0.0
                self.feature_medians[col] = median
            fill_value = self.feature_medians.get(col, 0.0)
            df[col] = df[col].fillna(fill_value)
        return df
