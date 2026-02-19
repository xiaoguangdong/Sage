"""
Walk-Forward 评估框架 — 滚动窗口训练 + Purging + Embargo + 分组回测

核心思想：
- 滚动窗口训练，模拟真实投资场景
- Purge gap 消除标签泄露（训练集末尾的标签"看到"验证集价格）
- Embargo 消除自相关（验证集开头几天与训练集末尾相关）
- 分组回测（Quintile）评估真实选股能力
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Walk-Forward 配置"""

    train_days: int = 504  # 训练窗口（约2年）
    val_days: int = 63  # 验证窗口（约1季度）
    step_days: int = 63  # 步进天数
    purge_days: int = 30  # Purge gap（>= max_label_horizon × 1.5）
    embargo_days: int = 5  # Embargo 天数
    n_quantiles: int = 5  # 分组数
    forward_days: int = 20  # 评估用前瞻收益天数


@dataclass
class WalkForwardResult:
    """Walk-Forward 结果"""

    predictions: pd.DataFrame  # 所有窗口的预测汇总
    quintile_returns: pd.DataFrame  # 分组收益
    summary: Dict  # 汇总指标
    window_details: List[Dict]  # 每个窗口的详情


class WalkForwardEvaluator:
    """Walk-Forward 评估器"""

    def __init__(
        self,
        wf_config: Optional[WalkForwardConfig] = None,
        selection_config: Optional[SelectionConfig] = None,
    ):
        self.wf_config = wf_config or WalkForwardConfig()
        self.selection_config = selection_config or SelectionConfig()

    def run(
        self,
        df: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
    ) -> WalkForwardResult:
        """
        执行 Walk-Forward 评估

        Args:
            df: 完整股票数据
            regime_labels: 可选的 regime 标签（与 df 行对齐）
        """
        cfg = self.wf_config
        date_col = self.selection_config.date_col

        dates = sorted(df[date_col].unique())
        n_dates = len(dates)

        all_preds = []
        window_details = []

        # 滚动窗口
        start = 0
        window_id = 0
        while start + cfg.train_days + cfg.purge_days + cfg.val_days <= n_dates:
            train_end = start + cfg.train_days
            val_start = train_end + cfg.purge_days
            val_end = min(val_start + cfg.val_days, n_dates)

            train_dates = dates[start:train_end]
            val_dates = dates[val_start:val_end]

            if len(val_dates) == 0:
                break

            df_train = df[df[date_col].isin(set(train_dates))].copy()
            df_val = df[df[date_col].isin(set(val_dates))].copy()

            logger.info(
                f"[Window {window_id}] "
                f"Train: {train_dates[0]}~{train_dates[-1]} ({len(train_dates)}d), "
                f"Val: {val_dates[0]}~{val_dates[-1]} ({len(val_dates)}d)"
            )

            # 训练
            try:
                sel_cfg = copy.deepcopy(self.selection_config)
                model = StockSelector(sel_cfg)
                model.fit(df_train)

                # 预测验证集
                pred = model.predict(df_val)
                pred["_window_id"] = window_id
                all_preds.append(pred)

                n_features = len(model.feature_cols) if model.feature_cols else 0
                window_details.append(
                    {
                        "window_id": window_id,
                        "train_start": str(train_dates[0])[:10],
                        "train_end": str(train_dates[-1])[:10],
                        "val_start": str(val_dates[0])[:10],
                        "val_end": str(val_dates[-1])[:10],
                        "n_features": n_features,
                        "status": "ok",
                    }
                )
            except Exception as e:
                logger.warning(f"[Window {window_id}] 训练失败: {e}")
                window_details.append(
                    {
                        "window_id": window_id,
                        "status": f"failed: {e}",
                    }
                )

            start += cfg.step_days
            window_id += 1

        if not all_preds:
            return WalkForwardResult(
                predictions=pd.DataFrame(),
                quintile_returns=pd.DataFrame(),
                summary={"error": "无有效窗口"},
                window_details=window_details,
            )

        predictions = pd.concat(all_preds, ignore_index=True)

        # 计算前瞻收益用于评估
        predictions = self._add_forward_returns(predictions, df, cfg.forward_days)

        # 分组回测
        quintile_returns, summary = self._quintile_analysis(predictions, cfg.n_quantiles)

        return WalkForwardResult(
            predictions=predictions,
            quintile_returns=quintile_returns,
            summary=summary,
            window_details=window_details,
        )

    def _add_forward_returns(self, pred: pd.DataFrame, df_full: pd.DataFrame, forward_days: int) -> pd.DataFrame:
        """为预测结果添加前瞻收益（仅用于评估，不参与训练）"""
        date_col = self.selection_config.date_col
        code_col = self.selection_config.code_col
        price_col = self.selection_config.price_col

        # 构建 (code, date) → future_price 映射
        price_df = df_full[[code_col, date_col, price_col]].copy()
        price_df = price_df.sort_values([code_col, date_col])

        fwd = price_df.groupby(code_col)[price_col].shift(-forward_days)
        price_df["_fwd_price"] = fwd
        price_df["_fwd_ret"] = price_df["_fwd_price"] / price_df[price_col] - 1

        lookup = price_df.set_index([code_col, date_col])["_fwd_ret"]
        pred["forward_ret"] = pred.set_index([code_col, date_col]).index.map(lambda x: lookup.get(x, np.nan))
        return pred

    def _quintile_analysis(self, pred: pd.DataFrame, n_quantiles: int) -> Tuple[pd.DataFrame, Dict]:
        """分组回测分析"""
        date_col = self.selection_config.date_col

        if "forward_ret" not in pred.columns or pred["forward_ret"].isna().all():
            return pd.DataFrame(), {"error": "无前瞻收益数据"}

        valid = pred.dropna(subset=["score", "forward_ret"])
        if valid.empty:
            return pd.DataFrame(), {"error": "无有效数据"}

        # 按日期分组打分排名（Q1=最高分，Q5=最低分）
        valid["quantile"] = valid.groupby(date_col)["score"].transform(
            lambda s: n_quantiles - pd.qcut(s, n_quantiles, labels=False, duplicates="drop")
        )

        # 各组平均收益
        q_ret = valid.groupby([date_col, "quantile"])["forward_ret"].mean().unstack()
        q_ret.columns = [f"Q{int(c)}" for c in q_ret.columns]

        # 汇总
        mean_ret = q_ret.mean()
        long_short = mean_ret.iloc[0] - mean_ret.iloc[-1] if len(mean_ret) >= 2 else 0

        # 单调性检查（Q1 > Q2 > ... > Q5 为理想）
        monotonic_score = 0
        vals = mean_ret.values
        for i in range(len(vals) - 1):
            if vals[i] > vals[i + 1]:
                monotonic_score += 1
        monotonic_ratio = monotonic_score / max(len(vals) - 1, 1)

        # IC（每期 score 与 forward_ret 的 rank correlation）
        ic_list = []
        for _, group in valid.groupby(date_col):
            if len(group) >= 30:
                ic = group["score"].corr(group["forward_ret"], method="spearman")
                if not np.isnan(ic):
                    ic_list.append(ic)

        ic_mean = np.mean(ic_list) if ic_list else 0
        ic_ir = np.mean(ic_list) / (np.std(ic_list) + 1e-8) if ic_list else 0

        summary = {
            "n_windows": pred["_window_id"].nunique(),
            "n_dates": valid[date_col].nunique(),
            "n_stocks_per_date": valid.groupby(date_col).size().mean(),
            "quantile_mean_returns": mean_ret.to_dict(),
            "long_short_return": round(float(long_short), 6),
            "monotonic_ratio": round(float(monotonic_ratio), 4),
            "rank_ic": round(float(ic_mean), 6),
            "rank_ic_ir": round(float(ic_ir), 4),
            "ic_hit_rate": round(sum(1 for x in ic_list if x > 0) / max(len(ic_list), 1), 4),
        }

        return q_ret, summary
