# iFlow GLM 风格模型设计文档

## 一、核心问题定义

在风格切换多变的A股市场中，如何：
1. **找到当前市场的主导逻辑**（不是判断牛熊，而是识别"钱在玩什么"）
2. **执行这套逻辑**（用选股模型筛选符合逻辑的股票）
3. **预测逻辑失效**（在逻辑切换前撤离）
4. **与趋势模型配合**（趋势判断决定是否参与，风格决定参与什么）
5. **系统化交易**（从识别→筛选→执行→风控，全流程自动化）

---

## 二、设计哲学（区别于传统状态机）

### 传统状态机的问题
```python
# 传统方式：预设风格标签
STATE = ["BULL", "BEAR", "SIDEWAYS", "ROTATION"]
if trend > threshold:
    return "BULL"  # ❌ 这是预测，不是观察
```

### 我们的方式：基于"资金流向"的逆向推导
```python
# 我们的方式：观察钱去哪，推导当前逻辑
hot_sectors = find_hot_sectors()  # 哪些板块热
stock_features = analyze_winners(hot_sectors)  # 赢家有什么特征
current_logic = infer_logic_from_features()  # 逆向推导当前逻辑
```

**核心差异**：
- 传统：预设标签→用指标匹配标签
- 我们：观察市场→归纳当前逻辑→验证逻辑持续性

---

## 三、三层架构设计

```
┌─────────────────────────────────────────────────┐
│  第一层：逻辑发现层（观察市场，不预设）        │
│  "市场在奖励什么"                               │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  第二层：逻辑执行层（用选股模型筛选）          │
│  "找到符合当前逻辑的股票"                       │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  第三层：逻辑监控层（预测失效）                  │
│  "这套逻辑还在被奖励吗"                         │
└─────────────────────────────────────────────────┘
```

---

## 四、第一层：逻辑发现层

### 4.1 核心思想：不预设风格，只问三个问题

#### 问题1：哪些板块被"反复选中"？
```python
def find_hot_sectors(market_data, window=20):
    """
    找到最近N个交易日内，被资金反复下注的板块
    不是涨幅最大，而是"持续性热度"
    """
    # 指标1：板块成交额持续性
    sector_turnover_persistence = calc_turnover_persistence(sectors, window)

    # 指标2：板块内个股一致性
    sector_consistency = calc_stock_consistency(sectors, window)

    # 指标3：回调不破位
    sector_resilience = calc_resilience(sectors, window)

    # 综合评分
    hot_sectors = (
        sector_turnover_persistence * 0.4 +
        sector_consistency * 0.3 +
        sector_resilience * 0.3
    ).nlargest(5)

    return hot_sectors
```

#### 问题2：这些板块的"赢家"有什么特征？
```python
def analyze_winners(hot_sectors, market_data, top_n=20):
    """
    分析热门板块中的赢家股票，提取共同特征
    """
    winners = get_top_performers(hot_sectors, top_n)

    # 特征维度1：基本面特征
    fundamental_features = {
        "roe": winners["roe"].mean(),
        "pe": winners["pe"].mean(),
        "market_cap": winners["market_cap"].median(),
    }

    # 特征维度2：技术特征
    technical_features = {
        "momentum": winners["ret_20d"].mean(),
        "volatility": winners["volatility_20d"].mean(),
        "turnover": winners["turnover_rate"].mean(),
    }

    # 特征维度3：筹码特征
    chip_features = {
        "concentration": calc_chip_concentration(winners),
        "institution_holding": calc_institution_holding(winners),
    }

    return {
        "fundamental": fundamental_features,
        "technical": technical_features,
        "chip": chip_features
    }
```

#### 问题3：这些特征是否形成"一致性逻辑"？
```python
def infer_logic_from_features(winners_features, losers_features):
    """
    通过对比赢家和输家的特征，推断当前市场逻辑
    """
    logic_score = {}

    # 逻辑1：基本面驱动
    if winners_features["fundamental"]["roe"] > losers_features["fundamental"]["roe"] * 1.5:
        logic_score["fundamental"] = 0.8

    # 逻辑2：动量驱动
    if winners_features["technical"]["momentum"] > 0:
        logic_score["momentum"] = 0.9

    # 逻辑3：小盘投机
    if winners_features["fundamental"]["market_cap"] < losers_features["fundamental"]["market_cap"]:
        logic_score["small_cap"] = 0.7

    # 逻辑4：机构抱团
    if winners_features["chip"]["institution_holding"] > 0.5:
        logic_score["institution"] = 0.6

    # 选择主导逻辑
    dominant_logic = max(logic_score, key=logic_score.get)

    return {
        "logic_type": dominant_logic,
        "logic_score": logic_score,
        "confidence": logic_score[dominant_logic]
    }
```

### 4.2 输出：当前市场逻辑描述
```python
current_market_logic = {
    "date": "2026-02-10",
    "hot_sectors": ["半导体", "新能源", "军工"],
    "winners_profile": {
        "dominant_logic": "momentum",  # 主导逻辑：动量
        "logic_score": {
            "fundamental": 0.3,
            "momentum": 0.9,  # 动量逻辑最强
            "small_cap": 0.5,
            "institution": 0.4
        },
        "confidence": 0.9  # 高置信度
    },
    "key_characteristics": {
        "avg_ret_20d": 0.15,  # 20日平均收益15%
        "avg_turnover": 0.08,  # 换手率8%
        "avg_pe": 45.0  # PE中位数45
    }
}
```

---

## 五、第二层：逻辑执行层

### 5.1 核心：动态调整选股因子权重

根据第一层识别的当前逻辑，调整选股模型的因子权重：

```python
def dynamic_factor_weights(market_logic):
    """
    根据当前市场逻辑，动态调整因子权重
    """
    base_weights = {
        "momentum": 0.25,
        "quality": 0.25,
        "liquidity": 0.25,
        "risk": 0.25
    }

    # 逻辑1：动量驱动 → 提高动量因子权重
    if market_logic["logic_type"] == "momentum":
        base_weights["momentum"] = 0.5
        base_weights["quality"] = 0.2
        base_weights["liquidity"] = 0.2
        base_weights["risk"] = 0.1

    # 逻辑2：基本面驱动 → 提高质量因子权重
    elif market_logic["logic_type"] == "fundamental":
        base_weights["quality"] = 0.5
        base_weights["momentum"] = 0.2
        base_weights["liquidity"] = 0.2
        base_weights["risk"] = 0.1

    # 逻辑3：小盘投机 → 降低质量因子权重，提高流动性
    elif market_logic["logic_type"] == "small_cap":
        base_weights["momentum"] = 0.4
        base_weights["liquidity"] = 0.4
        base_weights["quality"] = 0.1
        base_weights["risk"] = 0.1

    # 逻辑4：机构抱团 → 平衡权重
    elif market_logic["logic_type"] == "institution":
        base_weights["momentum"] = 0.3
        base_weights["quality"] = 0.3
        base_weights["liquidity"] = 0.2
        base_weights["risk"] = 0.2

    return base_weights
```

### 5.2 与现有选股模型结合

我们已经有8个因子，现在根据市场逻辑重新加权：

```python
def adaptive_stock_scoring(stocks_factors, market_logic):
    """
    自适应股票评分
    """
    # 获取动态权重
    weights = dynamic_factor_weights(market_logic)

    # 计算加权得分
    scores = {}

    for stock, factors in stocks_factors.items():
        score = (
            factors["4w_ret"] * weights["momentum"] +
            factors["12w_ret"] * weights["momentum"] +
            factors["roe_ttm"] * weights["quality"] +
            factors["gross_margin"] * weights["quality"] +
            factors["turnover"] * weights["liquidity"] +
            factors["amt_rank"] * weights["liquidity"] +
            factors["vol_12w"] * weights["risk"] +
            factors["beta"] * weights["risk"]
        )
        scores[stock] = score

    # 归一化到0-100
    scores_normalized = normalize_scores(scores)

    return scores_normalized
```

### 5.3 与趋势模型配合

趋势模型（HMM）判断市场状态：
- 牛市/震荡市：允许参与
- 熊市：降低仓位或空仓

```python
def trend_model_integration(market_state, logic_confidence):
    """
    趋势模型与风格模型集成
    """
    # 趋势模型输出（HMM）
    if market_state == "BEAR":
        return {
            "position_limit": 0.3,  # 熊市最多30%仓位
            "action": "DEFENSIVE"
        }
    elif market_state == "BULL":
        return {
            "position_limit": 0.9,  # 牛市最多90%仓位
            "action": "AGGRESSIVE"
        }
    else:  # SIDEWAYS
        return {
            "position_limit": 0.6,  # 震荡市60%仓位
            "action": "NEUTRAL"
        }
```

---

## 六、第三层：逻辑监控层

### 6.1 核心：监测"逻辑有效性"，而非"价格走势"

```python
def monitor_logic_validity(historical_logic, current_logic, window=20):
    """
    监控当前逻辑是否仍然有效
    """
    # 监测指标1：逻辑收益衰减
    logic_return_decay = calc_logic_return_decay(historical_logic, window)

    # 监测指标2：逻辑漂移
    logic_drift = calc_logic_drift(historical_logic, current_logic)

    # 监测指标3：逻辑内部一致性下降
    logic_consistency = calc_logic_consistency(current_logic)

    # 监测指标4：替代逻辑出现
    emerging_logics = detect_emerging_logics(market_data)

    # 综合评分
    validity_score = (
        (1 - logic_return_decay) * 0.3 +
        (1 - logic_drift) * 0.3 +
        logic_consistency * 0.2 +
        (1 - len(emerging_logics) * 0.1) * 0.2
    )

    return {
        "validity_score": validity_score,
        "logic_return_decay": logic_return_decay,
        "logic_drift": logic_drift,
        "logic_consistency": logic_consistency,
        "emerging_logics": emerging_logics
    }
```

### 6.2 逻辑失效判定

```python
def logic_failure_warning(validity_metrics):
    """
    判定逻辑是否失效
    """
    # 警告级别
    if validity_metrics["validity_score"] > 0.7:
        return {
            "warning_level": "GREEN",
            "action": "CONTINUE",
            "message": "逻辑健康，继续执行"
        }

    elif validity_metrics["validity_score"] > 0.5:
        return {
            "warning_level": "YELLOW",
            "action": "REDUCE",
            "message": "逻辑弱化，建议降仓",
            "position_adjustment": -0.3
        }

    else:
        return {
            "warning_level": "RED",
            "action": "EXIT",
            "message": "逻辑失效，建议清仓",
            "position_adjustment": -1.0
        }
```

### 6.3 逻辑切换检测

```python
def detect_logic_switch(historical_logics, current_logic):
    """
    检测是否出现逻辑切换
    """
    # 比较当前逻辑与历史逻辑的差异
    logic_diff = calc_logic_distance(historical_logics, current_logic)

    if logic_diff > threshold:
        return {
            "switch_detected": True,
            "new_logic": current_logic,
            "old_logic": historical_logics[-1],
            "switch_magnitude": logic_diff
        }
    else:
        return {
            "switch_detected": False
        }
```

---

## 七、完整执行流程

### 7.1 每日执行流程

```python
def daily_execution(date):
    """
    每日执行流程
    """
    # ===== 步骤1：发现当前市场逻辑 =====
    market_data = load_market_data(date)

    hot_sectors = find_hot_sectors(market_data)
    winners_features = analyze_winners(hot_sectors, market_data)
    current_logic = infer_logic_from_features(winners_features, losers_features)

    # ===== 步骤2：趋势模型判断是否参与 =====
    market_state = hmm_model.predict(date)
    trend_signal = trend_model_integration(market_state, current_logic["confidence"])

    if trend_signal["action"] == "DEFENSIVE":
        # 熊市，降低仓位
        return {"action": "REDUCE_POSITION", "target_position": 0.3}

    # ===== 步骤3：用选股模型筛选股票 =====
    stocks_factors = load_stock_factors(date)
    dynamic_weights = dynamic_factor_weights(current_logic)
    stock_scores = adaptive_stock_scoring(stocks_factors, current_logic)

    # 选择Top N股票
    selected_stocks = select_top_stocks(stock_scores, top_n=20)

    # ===== 步骤4：监控逻辑有效性 =====
    historical_logics = load_historical_logics(window=20)
    validity_metrics = monitor_logic_validity(historical_logics, current_logic)
    warning = logic_failure_warning(validity_metrics)

    # ===== 步骤5：检测逻辑切换 =====
    logic_switch = detect_logic_switch(historical_logics, current_logic)

    # ===== 步骤6：生成交易信号 =====
    if warning["warning_level"] == "RED":
        # 逻辑失效，清仓
        return {
            "action": "CLEAR_POSITION",
            "reason": warning["message"],
            "position_adjustment": warning["position_adjustment"]
        }

    elif warning["warning_level"] == "YELLOW":
        # 逻辑弱化，降仓
        return {
            "action": "REDUCE_POSITION",
            "target_position": trend_signal["position_limit"] * 0.7,
            "selected_stocks": selected_stocks,
            "reason": warning["message"]
        }

    elif logic_switch["switch_detected"]:
        # 逻辑切换，重新学习
        return {
            "action": "RELEARN",
            "new_logic": logic_switch["new_logic"],
            "reason": "检测到逻辑切换"
        }

    else:
        # 逻辑健康，正常参与
        return {
            "action": "PARTICIPATE",
            "target_position": trend_signal["position_limit"],
            "selected_stocks": selected_stocks,
            "current_logic": current_logic,
            "validity_score": validity_metrics["validity_score"]
        }
```

### 7.2 每周执行流程（逻辑验证）

```python
def weekly_execution(date):
    """
    每周执行流程：深度验证逻辑
    """
    # 重新计算逻辑特征
    market_data = load_market_data(date, window=60)
    current_logic = discover_market_logic(market_data)

    # 对比上周逻辑
    last_week_logic = load_last_week_logic()

    # 计算逻辑漂移
    drift_score = calc_logic_drift(last_week_logic, current_logic)

    # 如果漂移过大，发出警告
    if drift_score > 2.0:
        return {
            "action": "WARNING",
            "message": "逻辑发生显著漂移，建议重新评估"
        }

    # 更新逻辑历史
    update_logic_history(current_logic)

    return {
        "action": "UPDATE",
        "current_logic": current_logic,
        "drift_score": drift_score
    }
```

---

## 八、与传统状态机的区别

| 维度 | 传统状态机 | iFlow GLM 风格模型 |
|------|-----------|-------------------|
| **输入** | 预设风格标签（牛/熊/震荡） | 观察资金流向，逆向推导逻辑 |
| **输出** | 市场状态标签 | 当前市场逻辑+置信度 |
| **更新** | 固定规则 | 滚动学习，自适应 |
| **预测** | 预测未来状态 | 监控当前逻辑有效性 |
| **执行** | 根据状态调整仓位 | 根据逻辑调整因子权重 |
| **失效检测** | 规则触发 | 逻辑收益衰减+漂移检测 |

---

## 九、关键优势

### 9.1 不预设风格，适应性强
- 不预设"牛市/熊市/震荡"标签
- 通过观察市场，实时发现当前逻辑
- 适应A股风格多变的特性

### 9.2 动态调整，而非固定规则
- 因子权重根据市场逻辑动态调整
- 不是"动量市就用动量因子"，而是"当前是动量逻辑，所以提高动量因子权重"

### 9.3 监控逻辑，而非预测趋势
- 不预测"牛市什么时候结束"
- 监控"当前逻辑是否还在被奖励"
- 逻辑失效→调整/退出

### 9.4 与趋势模型完美配合
- 趋势模型：判断是否参与（牛市/熊市）
- 风格模型：决定参与什么（动量/基本面/小盘）
- 逻辑监控：决定何时退出

### 9.5 可解释性强
- 输出不是"状态标签"，而是"当前逻辑+特征+置信度"
- 可以清楚知道"市场在奖励什么"
- 便于理解和调试

---

## 十、回测验证方案

### 10.1 回测框架

```python
def backtest_style_model(start_date, end_date):
    """
    回测风格模型
    """
    results = []

    for date in date_range(start_date, end_date):
        # 执行每日流程
        signal = daily_execution(date)

        # 记录结果
        results.append({
            "date": date,
            "action": signal["action"],
            "current_logic": signal.get("current_logic"),
            "validity_score": signal.get("validity_score"),
            "selected_stocks": signal.get("selected_stocks"),
            "position": signal.get("target_position")
        })

    # 计算绩效
    metrics = calculate_performance_metrics(results)

    return results, metrics
```

### 10.2 评估指标

1. **逻辑识别准确率**：识别出的逻辑是否与实际情况一致
2. **逻辑切换及时性**：逻辑切换前多久能检测到
3. **选股效果**：基于逻辑筛选的股票表现
4. **风险控制**：逻辑失效时的回撤控制
5. **整体收益**：系统的整体表现

### 10.3 对比基准

- **基准1**：固定因子权重（不调整）
- **基准2**：传统状态机（牛/熊/震荡）
- **基准3**：纯趋势模型（HMM）

---

## 十一、实施计划

### 阶段1：MVP开发（2周）
- 实现逻辑发现层（热点板块识别）
- 实现逻辑执行层（动态因子权重）
- 实现基础逻辑监控

### 阶段2：回测验证（1周）
- 用2020-2026年数据回测
- 对比基准策略
- 验证有效性

### 阶段3：优化迭代（1周）
- 根据回测结果调整参数
- 优化逻辑识别算法
- 改进失效检测机制

### 阶段4：实盘对接（1周）
- 与现有系统对接
- 集成趋势模型
- 实盘测试

---

## 十二、总结

### 核心思想
> **不预设风格，观察市场，发现逻辑，执行逻辑，监控逻辑**

### 与现有系统的关系
- **趋势模型（HMM）**：判断是否参与（牛市/熊市）
- **风格模型（本设计）**：决定参与什么（动量/基本面/小盘）
- **选股模型（8因子）**：筛选符合逻辑的股票
- **逻辑监控**：决定何时退出

### 最终目标
> **在风格切换多变的A股市场中，系统化地找到市场逻辑、执行逻辑、监控逻辑，与趋势模型和选股模型配合，实现持续盈利。**

---

## 附录：关键函数伪代码

### A. 热点板块识别
```python
def find_hot_sectors(market_data, window=20):
    """
    找到热门板块
    """
    # 计算板块指标
    sector_turnover = calc_sector_turnover(market_data, window)
    sector_consistency = calc_sector_consistency(market_data, window)
    sector_resilience = calc_sector_resilience(market_data, window)

    # 综合评分
    hot_sectors = (
        sector_turnover * 0.4 +
        sector_consistency * 0.3 +
        sector_resilience * 0.3
    ).nlargest(5)

    return hot_sectors
```

### B. 逻辑推断
```python
def infer_logic_from_features(winners, losers):
    """
    推断当前市场逻辑
    """
    logic_scores = {}

    # 对比赢家和输家特征
    if winners["roe"] > losers["roe"] * 1.5:
        logic_scores["fundamental"] = 0.8

    if winners["momentum"] > 0:
        logic_scores["momentum"] = 0.9

    # 选择主导逻辑
    dominant_logic = max(logic_scores, key=logic_scores.get)

    return {
        "logic_type": dominant_logic,
        "logic_score": logic_scores,
        "confidence": logic_scores[dominant_logic]
    }
```

### C. 逻辑有效性监控
```python
def monitor_logic_validity(historical_logics, current_logic):
    """
    监控逻辑有效性
    """
    # 计算逻辑收益衰减
    return_decay = calc_return_decay(historical_logics)

    # 计算逻辑漂移
    drift = calc_logic_drift(historical_logics, current_logic)

    # 计算逻辑一致性
    consistency = calc_logic_consistency(current_logic)

    # 综合评分
    validity_score = (
        (1 - return_decay) * 0.3 +
        (1 - drift) * 0.3 +
        consistency * 0.2 +
        0.2  # 基础分
    )

    return {
        "validity_score": validity_score,
        "return_decay": return_decay,
        "drift": drift,
        "consistency": consistency
    }
```

---

*文档版本：v1.0*
*创建日期：2026-02-10*
*作者：iFlow GLM*