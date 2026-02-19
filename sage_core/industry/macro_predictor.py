#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
宏观经济预测模型

功能：
1. 基于NBS和Tushare数据进行行业景气度预测
2. 输出行业机会列表、景气度评分、场景分类
3. 识别四种场景：复苏(RECOVERY)、大涨(BOOM)、衰退(RECESSION)、震荡(NEUTRAL)
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from sage_core.utils.column_normalizer import normalize_industry_columns


class MacroPredictor:
    """
    宏观经济预测模型

    职责：
    1. 识别具有宏观机会的行业
    2. 计算行业景气度评分（0-100）
    3. 判断行业场景（复苏/大涨/衰退/震荡）
    4. 输出关键指标供选股模型参考
    """

    def __init__(self, config_path: str = "config/sw_nbs_mapping.yaml", data_delay_days: int = 2):
        """
        初始化预测模型

        Args:
            config_path: 行业映射配置文件路径
        """
        self.config_path = config_path
        self.mapping_config = self._load_mapping()
        self.thresholds = self._get_thresholds()
        self.data_delay_days = max(0, int(data_delay_days))

    def _load_mapping(self) -> Dict:
        """加载行业映射配置"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _get_thresholds(self) -> Dict:
        """
        获取场景分类的阈值配置

        Returns:
            Dict: 各场景的阈值
        """
        return {
            "systemic_recession": {"credit_growth": 9.5, "pmi": 48.5},  # 社融增速低于此值  # PMI低于此值
            "boom": {
                "pb_percentile": 80,  # PB分位数高于此值
                "turnover_rate": 0.08,  # 换手率高于此值
                "rps_120": 90,  # 相对强度高于此值
            },
            "recovery": {
                "ppi_yoy": -2,  # PPI同比高于此值
                "pb_percentile": 60,  # PB分位数低于此值
                "inventory_cleared": True,  # 库存出清
            },
            "recession": {
                "ppi_yoy": -5,  # PPI同比低于此值
                "fai_yoy": 0,  # FAI增速低于此值
                "rev_yoy": 0,  # 营收增速低于此值
            },
        }

    def predict(
        self,
        date: str,
        macro_data: pd.DataFrame,
        industry_data: pd.DataFrame,
        northbound_data: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        预测指定日期的宏观情况

        Args:
            date: 预测日期（格式：YYYY-MM-DD）
            macro_data: 宏观数据（CPI、PPI、PMI、收益率等）
            industry_data: 行业数据（PPI、FAI、Output等）
            northbound_data: 北向资金数据（可选）

        Returns:
            Dict: 预测结果
        """
        macro_data = self._prepare_macro_data(macro_data)
        industry_data = self._prepare_industry_data(industry_data)
        if northbound_data is not None:
            northbound_data = self._prepare_northbound_data(northbound_data)

        # 1. 判断全行业衰退（系统风险）
        systemic_scenario = self._check_systemic_recession(macro_data, date)

        if systemic_scenario["is_recession"]:
            return {
                "date": date,
                "systemic_scenario": "SYSTEMIC RECESSION",
                "opportunity_industries": [],
                "risk_level": "HIGH",
                "message": systemic_scenario["message"],
                "summary": systemic_scenario["message"],
            }

        # 2. 为每个行业计算景气度评分和场景
        opportunity_industries = []

        # 获取申万行业列表
        sw_industries = self.mapping_config["sw_to_nbs"].keys()

        for sw_industry in sw_industries:
            # 获取该行业的指标
            industry_metrics = self._get_industry_metrics(sw_industry, date, industry_data, macro_data, northbound_data)

            if industry_metrics is None:
                continue

            # 计算景气度评分
            boom_score = self._calculate_boom_score(industry_metrics)

            # 判断场景
            scenario = self._classify_scenario(industry_metrics)

            # 只保留机会行业（复苏或大涨）
            if scenario in ["RECOVERY", "RECOVERY (STRONG)", "BOOM / BUBBLE"]:
                opportunity_industries.append(
                    {
                        "industry": sw_industry,
                        "industry_code": self._get_sw_code(sw_industry),
                        "boom_score": boom_score,
                        "scenario": scenario,
                        "characteristics": self._get_characteristics(industry_metrics),
                        "key_indicators": industry_metrics,
                    }
                )

        # 3. 按景气度评分排序
        opportunity_industries.sort(key=lambda x: x["boom_score"], reverse=True)

        # 4. 返回结果
        return {
            "date": date,
            "systemic_scenario": "NORMAL",
            "opportunity_industries": opportunity_industries,
            "risk_level": self._get_risk_level(opportunity_industries),
            "summary": self._generate_summary(opportunity_industries),
        }

    def _check_systemic_recession(self, macro_data: pd.DataFrame, date: str) -> Dict:
        """
        检查是否全行业衰退（系统风险）

        Args:
            macro_data: 宏观数据
            date: 日期

        Returns:
            Dict: 包含is_recession和message
        """
        # 获取最近的数据
        target_date = pd.to_datetime(date) - pd.Timedelta(days=self.data_delay_days)

        # 找到最近可用的宏观数据
        recent_data = macro_data[macro_data["date"] <= target_date]
        if len(recent_data) == 0:
            return {"is_recession": False, "message": "无宏观数据"}

        latest_data = recent_data.iloc[-1]

        # 检查社融增速
        credit_growth = latest_data.get("credit_growth", None)
        pmi_value = latest_data.get("pmi", None)

        reasons = []
        if credit_growth is not None and credit_growth < self.thresholds["systemic_recession"]["credit_growth"]:
            reasons.append(f"社融增速({credit_growth:.2f}%)低于阈值")

        if pmi_value is not None and pmi_value < self.thresholds["systemic_recession"]["pmi"]:
            reasons.append(f"PMI({pmi_value:.2f})低于阈值")

        if reasons:
            return {"is_recession": True, "message": "；".join(reasons)}

        return {"is_recession": False, "message": ""}

    def _get_industry_metrics(
        self,
        sw_industry: str,
        date: str,
        industry_data: pd.DataFrame,
        macro_data: pd.DataFrame,
        northbound_data: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict]:
        """
        获取指定行业的指标

        Args:
            sw_industry: 申万行业名称
            date: 日期
            industry_data: 行业数据
            macro_data: 宏观数据
            northbound_data: 北向资金数据

        Returns:
            Dict: 行业指标
        """
        target_date = pd.to_datetime(date) - pd.Timedelta(days=self.data_delay_days)

        # 从industry_data中获取该行业的数据
        ind_data = industry_data[industry_data["sw_industry"] == sw_industry]
        ind_data = ind_data[ind_data["date"] <= target_date]

        if len(ind_data) == 0:
            return None

        latest_ind_data = ind_data.iloc[-1]

        # 从macro_data中获取宏观数据
        macro = macro_data[macro_data["date"] <= target_date]
        if len(macro) == 0:
            return None

        latest_macro = macro.iloc[-1]

        def _safe(value, default):
            if value is None:
                return default
            try:
                if pd.isna(value):
                    return default
            except Exception:
                pass
            return value

        ppi_value = latest_ind_data.get("sw_ppi_yoy", None)
        if ppi_value is None or (isinstance(ppi_value, float) and pd.isna(ppi_value)):
            ppi_value = latest_ind_data.get("ppi_yoy", None)

        # 构建指标字典
        metrics = {
            # 价格指标
            "ppi_yoy": _safe(ppi_value, None),
            "ppi_change": self._calculate_change(ind_data, "sw_ppi_yoy", 2),
            # 投资指标
            "fai_yoy": _safe(latest_ind_data.get("fai_yoy", None), None),
            "fai_change": self._calculate_change(ind_data, "fai_yoy", 3),
            # 库存指标
            "inventory_yoy": _safe(latest_ind_data.get("inventory_yoy", 0), 0),
            "rev_yoy": _safe(latest_ind_data.get("rev_yoy", 0), 0),
            # 估值指标
            "pb_percentile": _safe(latest_ind_data.get("pb_percentile", 50), 50),
            "pe_percentile": _safe(latest_ind_data.get("pe_percentile", 50), 50),
            # 市场情绪指标
            "turnover_rate": _safe(latest_ind_data.get("turnover_rate", 0.02), 0.02),
            "rps_120": _safe(latest_ind_data.get("rps_120", 50), 50),
            # 宏观指标
            "credit_growth": _safe(latest_macro.get("credit_growth", None), None),
            "pmi": _safe(latest_macro.get("pmi", None), None),
            "cpi_yoy": _safe(latest_macro.get("cpi_yoy", None), None),
            "yield_10y": _safe(latest_macro.get("yield_10y", None), None),
        }

        # 北向资金信号
        if northbound_data is not None:
            nb_data = northbound_data[northbound_data["sw_industry"] == sw_industry]
            nb_data = nb_data[nb_data["trade_date"] <= target_date]
            if len(nb_data) > 0:
                latest_nb = nb_data.iloc[-1]
                metrics["northbound_signal"] = latest_nb.get("northbound_signal", 0)
                metrics["industry_ratio"] = latest_nb.get("industry_ratio", 0)
            else:
                metrics["northbound_signal"] = 0
                metrics["industry_ratio"] = 0
        else:
            metrics["northbound_signal"] = 0
            metrics["industry_ratio"] = 0

        return metrics

    def _prepare_macro_data(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        if macro_data is None or macro_data.empty:
            return macro_data
        data = macro_data.copy()
        if "date" not in data.columns:
            if "trade_date" in data.columns:
                data["date"] = pd.to_datetime(data["trade_date"])
            elif "month" in data.columns:
                data["date"] = pd.to_datetime(data["month"], errors="coerce")
            elif "datetime" in data.columns:
                data["date"] = pd.to_datetime(data["datetime"])
        else:
            data["date"] = pd.to_datetime(data["date"])
        return data

    def _prepare_industry_data(self, industry_data: pd.DataFrame) -> pd.DataFrame:
        if industry_data is None or industry_data.empty:
            return industry_data
        data = normalize_industry_columns(industry_data)
        if "date" not in data.columns:
            if "trade_date" in data.columns:
                data["date"] = pd.to_datetime(data["trade_date"])
            elif "month" in data.columns:
                data["date"] = pd.to_datetime(data["month"], errors="coerce")
        else:
            data["date"] = pd.to_datetime(data["date"])
        return data

    def _prepare_northbound_data(self, northbound_data: pd.DataFrame) -> pd.DataFrame:
        data = normalize_industry_columns(northbound_data)
        if "trade_date" in data.columns:
            data["trade_date"] = pd.to_datetime(data["trade_date"])
        return data

    def _calculate_change(self, df: pd.DataFrame, column: str, periods: int) -> float:
        """
        计算变化

        Args:
            df: 数据框
            column: 列名
            periods: 周期数

        Returns:
            float: 变化值
        """
        if column not in df.columns:
            return 0
        if len(df) < periods + 1:
            return 0

        recent = df[column].iloc[-1]
        previous = df[column].iloc[-(periods + 1)]

        if pd.isna(recent) or pd.isna(previous):
            return 0

        return recent - previous

    def _calculate_boom_score(self, metrics: Dict) -> float:
        """
        计算景气度评分（0-100）

        评分维度：
        1. 价格改善 (30%)：PPI回升
        2. 投资扩张 (25%)：CAPEX增加
        3. 库存出清 (20%)：被动去库
        4. 估值优势 (15%)：PB分位数低
        5. 资金关注 (10%)：北向资金流入

        Args:
            metrics: 行业指标

        Returns:
            float: 景气度评分 (0-100)
        """
        score = 0

        # 1. 价格改善 (30%)
        ppi_score = 0
        if metrics["ppi_yoy"] is not None:
            if metrics["ppi_yoy"] > 5:
                ppi_score = 100
            elif metrics["ppi_yoy"] > 0:
                ppi_score = 70
            elif metrics["ppi_yoy"] > -2:
                ppi_score = 50
            elif metrics["ppi_change"] > 0:  # PPI在回升
                ppi_score = 60
            else:
                ppi_score = 20
        score += ppi_score * 0.30

        # 2. 投资扩张 (25%)
        fai_score = 0
        if metrics["fai_yoy"] is not None:
            if metrics["fai_yoy"] > 15:
                fai_score = 100
            elif metrics["fai_yoy"] > 10:
                fai_score = 80
            elif metrics["fai_yoy"] > 5:
                fai_score = 60
            elif metrics["fai_yoy"] > 0:
                fai_score = 40
            else:
                fai_score = 10
        score += fai_score * 0.25

        # 3. 库存出清 (20%)
        inventory_score = 0
        if metrics["inventory_yoy"] < metrics["rev_yoy"] and metrics["inventory_yoy"] < 10:
            inventory_score = 100  # 被动去库
        elif metrics["inventory_yoy"] < metrics["rev_yoy"]:
            inventory_score = 70
        elif metrics["inventory_yoy"] < 15:
            inventory_score = 50
        else:
            inventory_score = 20
        score += inventory_score * 0.20

        # 4. 估值优势 (15%)
        valuation_score = 0
        if metrics["pb_percentile"] < 20:
            valuation_score = 100
        elif metrics["pb_percentile"] < 40:
            valuation_score = 80
        elif metrics["pb_percentile"] < 60:
            valuation_score = 60
        elif metrics["pb_percentile"] < 80:
            valuation_score = 40
        else:
            valuation_score = 10
        score += valuation_score * 0.15

        # 5. 资金关注 (10%)
        money_score = 0
        if metrics["northbound_signal"] > 0:
            money_score = 100
        elif metrics["industry_ratio"] > 0.05:  # 持仓占比>5%
            money_score = 80
        elif metrics["turnover_rate"] > 0.05:
            money_score = 60
        else:
            money_score = 30
        score += money_score * 0.10

        return round(score, 2)

    def _classify_scenario(self, metrics: Dict) -> str:
        """
        分类场景

        Args:
            metrics: 行业指标

        Returns:
            str: 场景标签
        """
        # 1. 判断行业大涨
        if (
            metrics["pb_percentile"] > self.thresholds["boom"]["pb_percentile"]
            and metrics["turnover_rate"] > self.thresholds["boom"]["turnover_rate"]
            and metrics["rps_120"] > self.thresholds["boom"]["rps_120"]
        ):
            return "BOOM / BUBBLE"

        # 2. 判断行业复苏
        ppi_improving = metrics["ppi_yoy"] > self.thresholds["recovery"]["ppi_yoy"]
        inventory_cleared = metrics["inventory_yoy"] < metrics["rev_yoy"] and metrics["inventory_yoy"] < 10
        reasonable_val = metrics["pb_percentile"] < self.thresholds["recovery"]["pb_percentile"]

        # 北向资金信号
        northbound_increasing = metrics.get("northbound_signal", 0) > 0

        if ppi_improving and inventory_cleared and reasonable_val and northbound_increasing:
            return "RECOVERY (STRONG)"
        elif ppi_improving and inventory_cleared and reasonable_val:
            return "RECOVERY"

        # 3. 判断行业衰退
        if (
            metrics["ppi_yoy"] < self.thresholds["recession"]["ppi_yoy"]
            and (metrics["fai_yoy"] is None or metrics["fai_yoy"] < self.thresholds["recession"]["fai_yoy"])
            and (metrics["rev_yoy"] is None or metrics["rev_yoy"] < self.thresholds["recession"]["rev_yoy"])
        ):
            return "INDUSTRY RECESSION"

        # 4. 默认状态
        return "NEUTRAL / MIXED"

    def _get_characteristics(self, metrics: Dict) -> Dict:
        """
        获取行业特点

        Args:
            metrics: 行业指标

        Returns:
            Dict: 行业特点
        """
        return {
            "demand_improving": metrics["ppi_change"] > 0 or metrics["ppi_yoy"] > 0,
            "inventory_cleared": metrics["inventory_yoy"] < metrics["rev_yoy"],
            "valuation_low": metrics["pb_percentile"] < 50,
            "capex_expanding": metrics["fai_yoy"] is not None and metrics["fai_yoy"] > 5,
        }

    def _get_sw_code(self, sw_industry: str) -> str:
        """
        获取申万行业代码

        Args:
            sw_industry: 申万行业名称

        Returns:
            str: 行业代码
        """
        # 这里应该从配置或数据库中获取
        # 简化处理，返回行业名称的拼音首字母
        return sw_industry

    def _get_risk_level(self, opportunity_industries: List[Dict]) -> str:
        """
        获取风险等级

        Args:
            opportunity_industries: 机会行业列表

        Returns:
            str: 风险等级
        """
        if len(opportunity_industries) == 0:
            return "HIGH"
        elif len(opportunity_industries) < 5:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_summary(self, opportunity_industries: List[Dict]) -> str:
        """
        生成摘要

        Args:
            opportunity_industries: 机会行业列表

        Returns:
            str: 摘要
        """
        if len(opportunity_industries) == 0:
            return "暂无明显机会行业"

        recovery_count = sum(1 for ind in opportunity_industries if "RECOVERY" in ind["scenario"])
        boom_count = sum(1 for ind in opportunity_industries if "BOOM" in ind["scenario"])

        summary = f"发现{len(opportunity_industries)}个机会行业"
        if recovery_count > 0:
            summary += f"，其中{recovery_count}个处于复苏期"
        if boom_count > 0:
            summary += f"，{boom_count}个处于大涨期"

        top_industry = opportunity_industries[0]
        summary += f"，景气度最高：{top_industry['industry']}({top_industry['boom_score']}分)"

        return summary


def main():
    """测试预测模型"""
    predictor = MacroPredictor()

    # 创建模拟数据
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="M")

    # 模拟宏观数据
    macro_data = pd.DataFrame(
        {
            "date": dates,
            "credit_growth": np.random.uniform(9, 14, len(dates)),
            "pmi": np.random.uniform(48, 52, len(dates)),
            "cpi_yoy": np.random.uniform(0, 3, len(dates)),
            "yield_10y": np.random.uniform(2.5, 3.5, len(dates)),
        }
    )

    # 模拟行业数据
    industry_data = pd.DataFrame(
        {
            "date": np.repeat(dates, 3),
            "sw_industry": np.tile(["电子", "汽车", "医药"], len(dates)),
            "sw_ppi_yoy": np.random.uniform(-5, 5, len(dates) * 3),
            "fai_yoy": np.random.uniform(-5, 15, len(dates) * 3),
            "inventory_yoy": np.random.uniform(0, 20, len(dates) * 3),
            "rev_yoy": np.random.uniform(-5, 15, len(dates) * 3),
            "pb_percentile": np.random.uniform(10, 90, len(dates) * 3),
            "pe_percentile": np.random.uniform(10, 90, len(dates) * 3),
            "turnover_rate": np.random.uniform(0.01, 0.15, len(dates) * 3),
            "rps_120": np.random.uniform(30, 95, len(dates) * 3),
        }
    )

    # 预测
    result = predictor.predict("2024-12-01", macro_data, industry_data)

    # 输出结果
    print("=" * 80)
    print(f"宏观经济预测结果 - {result['date']}")
    print("=" * 80)
    print(f"系统场景: {result['systemic_scenario']}")
    print(f"风险等级: {result['risk_level']}")
    print(f"摘要: {result['summary']}")
    print()

    if result["opportunity_industries"]:
        print("机会行业列表:")
        print("-" * 80)
        for i, ind in enumerate(result["opportunity_industries"], 1):
            print(f"{i}. {ind['industry']} ({ind['scenario']}) - 景气度: {ind['boom_score']}")
            print("   关键指标:")
            print(f"     - PPI同比: {ind['key_indicators']['ppi_yoy']:.2f}%")
            print(f"     - FAI增速: {ind['key_indicators']['fai_yoy']:.2f}%")
            print(f"     - PB分位: {ind['key_indicators']['pb_percentile']:.1f}%")
            print(f"     - 换手率: {ind['key_indicators']['turnover_rate']:.2%}")
            print(f"   特点: {ind['characteristics']}")
            print()


if __name__ == "__main__":
    main()
