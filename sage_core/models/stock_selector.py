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

    # 列名配置
    date_col: str = "trade_date"
    code_col: str = "ts_code"
    price_col: str = "close"

    # 特征
    feature_cols: Optional[Tuple[str, ...]] = None
    rule_weights: Optional[Dict[str, float]] = None

    # 模型参数
    lgbm_params: Dict[str, object] = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
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
        train_df = train_df.dropna(subset=feature_cols)

        if train_df.empty:
            raise ValueError("训练数据为空，请检查输入数据与标签窗口")

        self.feature_cols = feature_cols
        X = train_df[feature_cols].values
        y = train_df["label"].values

        if self.config.model_type == "rule":
            self._fit_rule(train_df, feature_cols)
        elif self.config.model_type == "lgbm":
            self._fit_lgbm(X, y)
        elif self.config.model_type == "xgb":
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
        df_features = df_features.dropna(subset=feature_cols)

        if df_features.empty:
            raise ValueError("预测数据为空，请检查输入数据")

        if self.config.model_type == "rule":
            scores = self._score_rule(df_features, feature_cols)
        else:
            scores = self.model.predict(df_features[feature_cols].values)

        df_result = df_features.copy()
        df_result["score"] = scores
        df_result["rank"] = self._rank_by_date(df_result)
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

        df["ret_5d"] = group[price_col].pct_change(5)
        df["ret_20d"] = group[price_col].pct_change(20)
        df["ret_60d"] = group[price_col].pct_change(60)
        df["ma_20_ratio"] = df[price_col] / group[price_col].transform(lambda s: s.rolling(20).mean())
        df["vol_20d"] = returns.groupby(df[code_col]).rolling(20).std().reset_index(level=0, drop=True)

        if "turnover" in df.columns:
            df["turnover_20d_mean"] = group["turnover"].transform(lambda s: s.rolling(20).mean())
            df["turnover_20d_std"] = group["turnover"].transform(lambda s: s.rolling(20).std())

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

    def _fit_lgbm(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import lightgbm as lgb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("未安装 lightgbm，请先安装依赖") from exc

        params = dict(self.config.lgbm_params)
        dataset = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            params,
            dataset,
            num_boost_round=int(params.get("num_boost_round", 200)),
            valid_sets=[dataset],
            callbacks=[lgb.log_evaluation(period=50)],
        )

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

        exclude = {
            self.config.date_col,
            self.config.code_col,
            self.config.price_col,
            "label",
        }
        if self.config.industry_col:
            exclude.add(self.config.industry_col)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]

    def _validate_columns(self, df: pd.DataFrame, columns: Iterable[str]) -> None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必要字段: {missing}")

    def _rank_by_date(self, df: pd.DataFrame) -> pd.Series:
        date_col = self.config.date_col
        if date_col in df.columns:
            return df.groupby(date_col)["score"].rank(ascending=False, method="first")
        return df["score"].rank(ascending=False, method="first")
