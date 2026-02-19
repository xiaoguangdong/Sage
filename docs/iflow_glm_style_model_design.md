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

## 五、深度建模：从观察到重构

### 5.1 当前模型的三大问题

#### 问题1：观察不够仔细敏感

**当前做法**：
```python
# 只看涨幅Top 20%和Bottom 20%
winners = date_data_sorted.head(top_20pct_count)
losers = date_data_sorted.tail(top_20pct_count)
```

**问题**：
- 只看表面的涨跌
- 没有看资金流向的结构
- 没有看筹码集中度的变化
- 没有看不同市值段的资金偏好

#### 问题2：指标重构没做

**当前做法**：
```python
# 只是调整因子权重
if logic_type == "momentum":
    base_weights["momentum"] = 0.5  # 放大动量权重
```

**问题**：
- 这只是"放大"，不是"重构"
- 动量驱动≠提高动量权重
- 不同逻辑需要完全不同的选股方法论

#### 问题3：输出定义问题

**当前输出**：
```python
{
    "logic_type": "momentum",
    "validity_score": 0.9,  # 强弱指标
    "position": 0.6         # 仓位建议
}
```

**问题**：
- 这些都是"强弱指标"，不是"区分指标"
- 无法回答"什么时候该用哪种选股逻辑"

---

### 5.2 重新思考：市场逻辑的本质是什么？

#### 核心定义

**市场逻辑 = 资金在玩什么游戏**

不是"涨不涨"，而是：
- 资金在聚集还是分散？
- 资金在追涨还是抄底？
- 资金在做长线还是短线？
- 资金在博弈哪些板块？

#### 四种底层逻辑（重新定义）

| 逻辑类型 | 核心特征 | 资金行为 | 选股方法 |
|---------|---------|---------|---------|
| **资金聚集** | 筹码快速集中 | 机构加仓 | 跟踪机构动向 |
| **板块轮动** | 热点快速切换 | 游资接力 | 抓龙头、追热点 |
| **情绪反转** | 超跌/超涨 | 散户博弈 | 做情绪反人性 |
| **结构分化** | 大小盘分化 | 风格切换 | 做风格配置 |

---

### 5.3 观察层的重构：更仔细、更敏感

#### 第一层：资金流向观察（微观）

**观察什么**：
```python
def observe_capital_flow(date_data):
    """
    观察资金流向结构
    """
    # 1. 按市值分层观察
    cap_segments = {
        "large_cap": date_data[date_data['total_mv'] > 500000000000],   # 大盘股
        "mid_cap": date_data[(date_data['total_mv'] > 100000000000) & (date_data['total_mv'] <= 500000000000)],  # 中盘股
        "small_cap": date_data[date_data['total_mv'] <= 100000000000]    # 小盘股
    }

    # 2. 计算各层级的资金净流入
    for segment, data in cap_segments.items():
        # 资金净流入 = (今日成交额 - 均值成交额) / 均值成交额
        data['net_flow'] = (data['amount'] - data['amount'].rolling(20).mean()) / data['amount'].rolling(20).mean()

    # 3. 观察资金偏好的变化
    capital_preference = {
        "large_cap_preference": cap_segments["large_cap"]['net_flow'].mean(),
        "mid_cap_preference": cap_segments["mid_cap"]['net_flow'].mean(),
        "small_cap_preference": cap_segments["small_cap"]['net_flow'].mean()
    }

    return capital_preference
```

#### 第二层：筹码结构观察（中观）

**观察什么**：
```python
def observe_chip_structure(date_data):
    """
    观察筹码结构变化
    """
    # 1. 筹码集中度（使用成交额分布）
    amount_distribution = date_data['amount'].values
    gini_coefficient = calculate_gini(amount_distribution)

    # 2. 筹码聚集度（前20%成交额占比）
    top_20pct_amount = date_data.nlargest(int(len(date_data) * 0.2), 'amount')['amount'].sum()
    total_amount = date_data['amount'].sum()
    concentration_ratio = top_20pct_amount / total_amount

    # 3. 换手率结构
    turnover_segments = {
        "high_turnover": date_data[date_data['turnover_rate'] > 0.1],  # 高换手
        "medium_turnover": date_data[(date_data['turnover_rate'] > 0.05) & (date_data['turnover_rate'] <= 0.1)],
        "low_turnover": date_data[date_data['turnover_rate'] <= 0.05]
    }

    return {
        "gini_coefficient": gini_coefficient,
        "concentration_ratio": concentration_ratio,
        "turnover_structure": turnover_segments
    }
```

#### 第三层：板块轮动观察（宏观）

**观察什么**：
```python
def observe_sector_rotation(date_data, sector_map):
    """
    观察板块轮动节奏
    """
    # 1. 计算各板块的表现
    sector_performance = {}
    for sector, stocks in sector_map.items():
        sector_data = date_data[date_data['ts_code'].isin(stocks)]
        if len(sector_data) > 0:
            sector_performance[sector] = {
                "avg_return": sector_data['pct_chg'].mean(),
                "amount": sector_data['amount'].sum(),
                "stock_count": len(sector_data),
                "up_ratio": (sector_data['pct_chg'] > 0).sum() / len(sector_data)
            }

    # 2. 识别热门板块（成交额Top 3）
    hot_sectors = sorted(sector_performance.items(), key=lambda x: x[1]['amount'], reverse=True)[:3]

    # 3. 识别轮动速度（热门板块的变化）
    # （需要历史数据，这里省略）

    return {
        "sector_performance": sector_performance,
        "hot_sectors": hot_sectors
    }
```

---

### 5.4 选股逻辑的重构：不是调整权重，是重新设计

#### 逻辑1：资金聚集逻辑

**核心思想**：跟踪主力资金动向

**选股指标（重构版）**：
```python
def select_stocks_capital_aggregation(date_data, cap_flow_observation):
    """
    资金聚集逻辑选股
    """
    selected_stocks = []

    # 指标1：机构持仓增加（需要数据）
    # indicator1 = institution_holding_change > threshold

    # 指标2：成交额持续放大
    date_data['amount_trend'] = date_data['amount'].rolling(5).mean() / date_data['amount'].rolling(20).mean()
    indicator2 = date_data['amount_trend'] > 1.2

    # 指标3：换手率适中（2-8%）
    indicator3 = (date_data['turnover_rate'] > 0.02) & (date_data['turnover_rate'] < 0.08)

    # 指标4：价格稳步上涨（不是暴涨）
    date_data['price_trend'] = date_data['close'].pct_change(20)
    indicator4 = (date_data['price_trend'] > 0.05) & (date_data['price_trend'] < 0.2)

    # 综合筛选
    selected = date_data[indicator2 & indicator3 & indicator4].copy()

    # 排序：按成交额趋势和价格趋势综合评分
    selected['score'] = (
        selected['amount_trend'] * 0.6 +
        selected['price_trend'] * 0.4
    )

    return selected.sort_values('score', ascending=False)
```

#### 逻辑2：板块轮动逻辑

**核心思想**：抓龙头、追热点

**选股指标（重构版）**：
```python
def select_stocks_sector_rotation(date_data, sector_observation):
    """
    板块轮动逻辑选股
    """
    hot_sectors = sector_observation['hot_sectors']

    selected_stocks = []

    for sector, performance in hot_sectors:
        # 获取该板块的股票
        sector_stocks = sector_map[sector]
        sector_data = date_data[date_data['ts_code'].isin(sector_stocks)].copy()

        # 指标1：板块内涨幅Top 3
        sector_data['sector_rank'] = sector_data['pct_chg'].rank(ascending=False)
        indicator1 = sector_data['sector_rank'] <= 3

        # 指标2：成交额大
        indicator2 = sector_data['amount'] > sector_data['amount'].quantile(0.7)

        # 指标3：换手率高（但不是异常高）
        indicator3 = (sector_data['turnover_rate'] > 0.05) & (sector_data['turnover_rate'] < 0.2)

        # 指标4：上涨日占比高
        indicator4 = sector_data['up_ratio'] > 0.6

        # 综合筛选
        selected = sector_data[indicator1 & indicator2 & indicator3 & indicator4].copy()

        # 排序：按涨幅和成交额综合评分
        selected['score'] = (
            selected['pct_chg'] * 0.5 +
            selected['amount'] / selected['amount'].max() * 0.5
        )

        selected_stocks.append(selected)

    # 合并所有板块的选股结果
    final_selected = pd.concat(selected_stocks)

    return final_selected.sort_values('score', ascending=False)
```

#### 逻辑3：情绪反转逻辑

**核心思想**：做情绪反人性

**选股指标（重构版）**：
```python
def select_stocks_sentiment_reversal(date_data, market_sentiment):
    """
    情绪反转逻辑选股
    """
    # 指标1：超跌（20日跌幅 > 20%）
    date_data['drawdown_20d'] = (date_data['close'] - date_data['close'].rolling(20).max()) / date_data['close'].rolling(20).max()
    indicator1 = date_data['drawdown_20d'] < -0.2

    # 指标2：恐慌卖出（换手率异常高）
    indicator2 = date_data['turnover_rate'] > 0.15

    # 指标3：基本面不差（ROE > 0）
    indicator3 = date_data['roe'] > 0

    # 指标4：开始企稳（3日不创新低）
    date_data['local_low'] = date_data['close'] == date_data['close'].rolling(3).min()
    indicator4 = ~date_data['local_low']

    # 综合筛选
    selected = date_data[indicator1 & indicator2 & indicator3 & indicator4].copy()

    # 排序：按跌幅和基本面综合评分
    selected['score'] = (
        abs(selected['drawdown_20d']) * 0.6 +
        selected['roe'] * 0.4
    )

    return selected.sort_values('score', ascending=False)
```

#### 逻辑4：结构分化逻辑

**核心思想**：做风格配置

**选股指标（重构版）**：
```python
def select_stocks_structural_divergence(date_data, cap_flow_observation):
    """
    结构分化逻辑选股
    """
    # 判断当前风格偏好
    if cap_flow_observation['large_cap_preference'] > 0.1:
        # 大盘风格
        target_segment = "large_cap"
    elif cap_flow_observation['small_cap_preference'] > 0.1:
        # 小盘风格
        target_segment = "small_cap"
    else:
        # 中盘风格
        target_segment = "mid_cap"

    # 筛选对应市值段的股票
    if target_segment == "large_cap":
        selected = date_data[date_data['total_mv'] > 500000000000].copy()
    elif target_segment == "small_cap":
        selected = date_data[date_data['total_mv'] <= 100000000000].copy()
    else:
        selected = date_data[(date_data['total_mv'] > 100000000000) & (date_data['total_mv'] <= 500000000000)].copy()

    # 指标1：流动性好
    indicator1 = selected['turnover_rate'] > 0.02

    # 指标2：波动率适中
    selected['volatility_20d'] = selected['pct_chg'].rolling(20).std()
    indicator2 = (selected['volatility_20d'] > 0.01) & (selected['volatility_20d'] < 0.05)

    # 指标3：趋势向上
    selected['trend_20d'] = (selected['close'] - selected['close'].shift(20)) / selected['close'].shift(20)
    indicator3 = selected['trend_20d'] > 0

    # 综合筛选
    selected = selected[indicator1 & indicator2 & indicator3].copy()

    # 排序：按趋势和波动率综合评分
    selected['score'] = (
        selected['trend_20d'] * 0.7 -
        selected['volatility_20d'] * 0.3
    )

    return selected.sort_values('score', ascending=False)
```

---

### 5.5 模型输出的重构：从强弱到区分

#### 新的输出结构

**不再是**：
```python
{
    "logic_type": "momentum",
    "validity_score": 0.9,  # 强弱指标
    "position": 0.6
}
```

**而是**：
```python
{
    # 1. 市场状态（区分指标）
    "market_state": {
        "primary_logic": "capital_aggregation",  # 主导逻辑
        "secondary_logic": "sector_rotation",    # 次要逻辑
        "logic_mix_ratio": 0.7,                  # 逻辑混合比例
        "clarity": 0.85                          # 逻辑清晰度（0-1）
    },

    # 2. 逻辑特征（区分特征）
    "logic_features": {
        "capital_aggregation": {
            "signal_strength": 0.8,
            "persistence": 0.9,
            "risk_level": "MEDIUM"
        },
        "sector_rotation": {
            "signal_strength": 0.6,
            "persistence": 0.4,
            "risk_level": "HIGH"
        }
    },

    # 3. 选股结果（直接输出）
    "selected_stocks": {
        "capital_aggregation": ["600519.SH", "000858.SZ", ...],  # 资金聚集逻辑选出的股票
        "sector_rotation": ["300750.SZ", "002594.SZ", ...],     # 板块轮动逻辑选出的股票
        "mixed": [...]                                        # 混合逻辑选出的股票
    },

    # 4. 执行建议（区分性建议）
    "execution": {
        "strategy": "PRIMARY_LOGIC_DOMINANT",  # 策略类型
        "position_allocation": {
            "capital_aggregation": 0.7,
            "sector_rotation": 0.3
        },
        "risk_control": {
            "stop_loss": -0.05,
            "take_profit": 0.15,
            "max_drawdown": 0.1
        }
    }
}
```

#### 策略类型的定义

```python
STRATEGY_TYPES = {
    "PRIMARY_LOGIC_DOMINANT": "主导逻辑为主，逻辑清晰度高",
    "LOGIC_MIXED": "多种逻辑并存，需要分散配置",
    "LOGIC_TRANSITION": "逻辑切换期，降低仓位",
    "LOGIC_FAILURE": "逻辑失效，清仓观望"
}
```

---

### 5.6 系统架构重构

```
┌─────────────────────────────────────────────────┐
│  观察层（多维度、细粒度）                      │
│  - 资金流向观察（微观）                         │
│  - 筹码结构观察（中观）                         │
│  - 板块轮动观察（宏观）                         │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  逻辑识别层（从观察到推断）                      │
│  - 主导逻辑识别                                 │
│  - 次要逻辑识别                                 │
│  - 逻辑清晰度计算                               │
│  - 逻辑混合比例                                 │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  选股逻辑层（根据逻辑重构）                      │
│  - 资金聚集逻辑 → 机构追踪选股                  │
│  - 板块轮动逻辑 → 龙头追涨选股                  │
│  - 情绪反转逻辑 → 超跌反弹选股                  │
│  - 结构分化逻辑 → 风格配置选股                  │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  逻辑监控层（有效性监控）                        │
│  - 逻辑持续性监控                               │
│  - 逻辑漂移检测                                 │
│  - 逻辑失效预警                                 │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  决策输出层（区分性建议）                        │
│  - 策略类型选择                                 │
│  - 仓位分配建议                                 │
│  - 风险控制参数                                 │
└─────────────────────────────────────────────────┘
```

---

## 六、实施路径

### 阶段1：观察层升级（3天）
- 实现资金流向观察
- 实现筹码结构观察
- 实现板块轮动观察

### 阶段2：选股逻辑重构（5天）
- 实现四种选股逻辑的选股函数
- 每种逻辑有独立的选股指标
- 每种逻辑有独立的评分系统

### 阶段3：逻辑识别层（3天）
- 实现主导逻辑识别
- 实现次要逻辑识别
- 实现逻辑清晰度计算

### 阶段4：决策输出层（2天）
- 实现策略类型判断
- 实现仓位分配逻辑
- 实现风险控制参数

### 阶段5：回测验证（3天）
- 对比新旧模型
- 细分场景测试
- 参数优化

---

## 七、关键改进点总结

### 改进1：观察更细致
- ❌ 之前：只看涨幅Top/Bottom
- ✅ 现在：看资金流向、筹码结构、板块轮动

### 改进2：选股重构
- ❌ 之前：调整因子权重
- ✅ 现在：根据逻辑重新设计选股指标

### 改进3：输出区分
- ❌ 之前：输出强弱指标（validity_score）
- ✅ 现在：输出区分指标（策略类型、逻辑清晰度）

### 改进4：逻辑混合
- ❌ 之前：单一逻辑（momentum/fundamental）
- ✅ 现在：多逻辑混合（主逻辑+次逻辑）

### 改进5：风险控制
- ❌ 之前：简单的仓位调整
- ✅ 现在：完整的止损止盈系统

---

## 八、避免模型退化：从博弈到价值

### 8.1 问题的本质

你的质疑揭示了一个**致命风险**：

> **模型可能退化成"选高波动率小股票"的博弈模型**

#### 当前设计的潜在缺陷

**缺陷1：板块轮动逻辑可能选出妖股**
```python
# 指标1：板块内涨幅Top 3
sector_data['sector_rank'] = sector_data['pct_chg'].rank(ascending=False)
indicator1 = sector_data['sector_rank'] <= 3
```
- 短期暴涨 = 可能是小盘股、妖股、被炒作
- 缺少对公司质量的约束

**缺陷2：资金聚集逻辑可能跟踪游资**
```python
# 指标2：成交额持续放大
date_data['amount_trend'] = date_data['amount'].rolling(5).mean() / date_data['amount'].rolling(20).mean()
indicator2 = date_data['amount_trend'] > 1.2
```
- 成交额放大可能是游资炒作，不一定是机构加仓
- 缺少对资金性质的区分

**缺陷3：情绪反转逻辑可能选出垃圾股**
```python
# 指标1：超跌（20日跌幅 > 20%）
date_data['drawdown_20d'] = (date_data['close'] - date_data['close'].rolling(20).max()) / date_data['close'].rolling(20).max()
indicator1 = date_data['drawdown_20d'] < -0.2
```
- 超跌可能是垃圾股，不是被错杀的好公司
- 缺少对基本面的深度验证

---

### 8.2 解决方案：三层过滤体系

#### 第一层：硬性约束（必须满足）

**约束1：市值门槛**
```python
# 根据市场阶段调整市值门槛
def get_market_cap_threshold(market_state):
    if market_state == "HEALTHY_UP":
        return 50000000000   # 50亿（牛市可以选小票）
    elif market_state == "DISTRIBUTION":
        return 100000000000  # 100亿（风险期只选大盘）
    else:
        return 100000000000  # 默认100亿
```

**约束2：流动性门槛**
```python
# 日均成交额必须 > 5000万
def check_liquidity(stock_data):
    avg_amount_20d = stock_data['amount'].rolling(20).mean()
    return avg_amount_20d > 50000000  # 5000万
```

**约束3：基本面硬约束**
```python
def check_fundamental_hard(stock_data):
    # ROE > 0
    indicator1 = stock_data['roe'] > 0

    # 净利润 > 0（近4个季度）
    indicator2 = stock_data['net_profit_ttm'] > 0

    # 经营性现金流 > 0（避免造假）
    indicator3 = stock_data['operating_cash_flow'] > 0

    # 负债率 < 70%（避免高杠杆）
    indicator4 = stock_data['debt_ratio'] < 0.7

    return indicator1 & indicator2 & indicator3 & indicator4
```

**约束4：风险门槛**
```python
def check_risk(stock_data):
    # 过去60天最大回撤 < 40%（避免崩盘股）
    drawdown_60d = calc_max_drawdown(stock_data, window=60)
    indicator1 = drawdown_60d < 0.4

    # 波动率 < 30%（避免妖股）
    volatility_60d = stock_data['pct_chg'].rolling(60).std()
    indicator2 = volatility_60d < 0.3

    # 财报风险：最近一次审计意见 = 标准无保留
    indicator3 = stock_data['audit_opinion'] == "标准无保留"

    return indicator1 & indicator2 & indicator3
```

#### 第二层：价值评分（加权评分）

**价值评分维度**
```python
def calc_value_score(stock_data):
    """
    计算价值评分（0-100）
    """
    # 1. 盈利能力（30分）
    profit_score = (
        min(stock_data['roe'] / 0.2, 1) * 15 +  # ROE满分20%
        min(stock_data['gross_margin'] / 0.3, 1) * 15  # 毛利率满分30%
    )

    # 2. 成长性（25分）
    growth_score = (
        min(stock_data['revenue_growth_yoy'] / 0.3, 1) * 15 +  # 营收增长30%满分
        min(stock_data['profit_growth_yoy'] / 0.3, 1) * 10    # 利润增长30%满分
    )

    # 3. 财务健康（20分）
    health_score = (
        min((1 - stock_data['debt_ratio']) / 0.7, 1) * 10 +  # 负债率
        min(stock_data['current_ratio'] / 2, 1) * 10          # 流动比率2倍满分
    )

    # 4. 估值安全（25分）
    valuation_score = (
        min(1 / stock_data['pe_ttm'] / 50, 1) * 10 +  # PE 50倍以下
        min(1 / stock_data['pb'] / 3, 1) * 15          # PB 3倍以下
    )

    total_score = profit_score + growth_score + health_score + valuation_score

    return total_score
```

#### 第三层：机构认可度（加分项）

**机构认可度维度**
```python
def calc_institution_score(stock_data):
    """
    计算机构认可度（0-20）
    """
    score = 0

    # 1. 机构持仓比例（10分）
    institution_holding = stock_data['institution_holding_ratio']
    score += min(institution_holding / 0.5, 1) * 10

    # 2. 北向资金流入（5分）
    north_flow = stock_data['north_flow_20d']
    if north_flow > 0:
        score += min(north_flow / 1000000000, 1) * 5  # 10亿流入满分

    # 3. 研报覆盖（5分）
    report_count = stock_data['analyst_report_count_90d']
    score += min(report_count / 10, 1) * 5  # 10篇研报满分

    return score
```

---

### 8.3 重构后的选股逻辑

#### 逻辑1：资金聚集逻辑（重构版）

```python
def select_stocks_capital_aggregation_v2(date_data, cap_flow_observation, market_state):
    """
    资金聚集逻辑选股（价值增强版）
    """
    # 第一步：硬性约束过滤
    cap_threshold = get_market_cap_threshold(market_state)
    filtered = date_data[
        (date_data['total_mv'] > cap_threshold) &
        (date_data['amount'].rolling(20).mean() > 50000000)
    ].copy()

    # 第二步：基本面硬约束
    filtered = filtered[check_fundamental_hard(filtered)].copy()

    # 第三步：风险约束
    filtered = filtered[check_risk(filtered)].copy()

    # 第四步：资金聚集信号
    filtered['amount_trend'] = filtered['amount'].rolling(5).mean() / filtered['amount'].rolling(20).mean()
    filtered['price_trend'] = filtered['close'].pct_change(20)

    # 机构资金特征（不只是成交额放大）
    indicator1 = filtered['amount_trend'] > 1.2
    indicator2 = filtered['price_trend'] > 0.05
    indicator3 = (filtered['turnover_rate'] > 0.02) & (filtered['turnover_rate'] < 0.08)

    # 第五步：价值评分
    filtered['value_score'] = calc_value_score(filtered)
    filtered['institution_score'] = calc_institution_score(filtered)

    # 第六步：综合评分
    filtered['final_score'] = (
        filtered['amount_trend'] * 20 +      # 资金信号
        filtered['price_trend'] * 100 +      # 价格趋势
        filtered['value_score'] * 0.5 +      # 价值评分
        filtered['institution_score'] * 1    # 机构认可
    )

    # 第七步：筛选Top 20
    selected = filtered[
        indicator1 & indicator2 & indicator3
    ].nlargest(20, 'final_score')

    return selected
```

#### 逻辑2：板块轮动逻辑（重构版）

```python
def select_stocks_sector_rotation_v2(date_data, sector_observation, market_state):
    """
    板块轮动逻辑选股（价值增强版）
    """
    hot_sectors = sector_observation['hot_sectors']
    selected_stocks = []

    for sector, performance in hot_sectors:
        # 获取该板块的股票
        sector_stocks = sector_map[sector]
        sector_data = date_data[date_data['ts_code'].isin(sector_stocks)].copy()

        # 第一步：硬性约束过滤
        cap_threshold = get_market_cap_threshold(market_state)
        sector_data = sector_data[sector_data['total_mv'] > cap_threshold].copy()

        # 第二步：基本面硬约束
        sector_data = sector_data[check_fundamental_hard(sector_data)].copy()

        # 第三步：龙头识别（不只是涨幅）
        sector_data['sector_rank_return'] = sector_data['pct_chg'].rank(ascending=False)
        sector_data['sector_rank_amount'] = sector_data['amount'].rank(ascending=False)

        # 龙头 = 涨幅Top 3 + 成交额Top 5
        indicator1 = (sector_data['sector_rank_return'] <= 3) & (sector_data['sector_rank_amount'] <= 5)

        # 第四步：价值评分
        sector_data['value_score'] = calc_value_score(sector_data)

        # 第五步：机构认可
        sector_data['institution_score'] = calc_institution_score(sector_data)

        # 第六步：综合评分
        sector_data['final_score'] = (
            sector_data['pct_chg'] * 50 +         # 涨幅
            sector_data['amount'] / sector_data['amount'].max() * 20 +  # 成交额
            sector_data['value_score'] * 0.5 +   # 价值评分
            sector_data['institution_score'] * 1  # 机构认可
        )

        # 第七步：筛选该板块Top 5
        selected = sector_data[indicator1].nlargest(5, 'final_score')
        selected_stocks.append(selected)

    # 合并所有板块的选股结果
    final_selected = pd.concat(selected_stocks)

    return final_selected.sort_values('final_score', ascending=False)
```

#### 逻辑3：情绪反转逻辑（重构版）

```python
def select_stocks_sentiment_reversal_v2(date_data, market_sentiment, market_state):
    """
    情绪反转逻辑选股（价值增强版）
    """
    # 第一步：硬性约束过滤
    cap_threshold = get_market_cap_threshold(market_state)
    filtered = date_data[date_data['total_mv'] > cap_threshold].copy()

    # 第二步：基本面硬约束（反转逻辑中更严格）
    filtered = filtered[
        (filtered['roe'] > 0.1) &           # ROE > 10%
        (filtered['debt_ratio'] < 0.6) &    # 负债率 < 60%
        (filtered['net_profit_ttm'] > 0)    # 净利润 > 0
    ].copy()

    # 第三步：超跌识别
    filtered['drawdown_20d'] = (
        (filtered['close'] - filtered['close'].rolling(20).max()) /
        filtered['close'].rolling(20).max()
    )

    # 第四步：价值评分（反转逻辑中权重更高）
    filtered['value_score'] = calc_value_score(filtered)

    # 第五步：超跌但有价值
    indicator1 = filtered['drawdown_20d'] < -0.2  # 跌幅 > 20%
    indicator2 = filtered['value_score'] > 60     # 价值评分 > 60

    # 第六步：开始企稳
    filtered['local_low'] = filtered['close'] == filtered['close'].rolling(3).min()
    indicator3 = ~filtered['local_low']

    # 第七步：机构认可（加分项）
    filtered['institution_score'] = calc_institution_score(filtered)

    # 第八步：综合评分
    filtered['final_score'] = (
        abs(filtered['drawdown_20d']) * 50 +  # 跌幅越大，分越高
        filtered['value_score'] * 1 +         # 价值评分
        filtered['institution_score'] * 1.5   # 机构认可（反转逻辑中权重更高）
    )

    # 第九步：筛选Top 15
    selected = filtered[
        indicator1 & indicator2 & indicator3
    ].nlargest(15, 'final_score')

    return selected
```

---

### 8.4 行业价值识别：如何选出真正有价值的公司？

#### 行业价值评分体系

```python
def calc_sector_value_score(sector_data):
    """
    计算行业价值评分（0-100）
    """
    # 1. 行业成长性（30分）
    sector_growth = (
        sector_data['revenue_growth_yoy'].mean() * 50  # 营收增长
    )

    # 2. 行业盈利能力（30分）
    sector_profit = (
        min(sector_data['roe'].mean() / 0.15, 1) * 20 +  # ROE
        min(sector_data['gross_margin'].mean() / 0.25, 1) * 10  # 毛利率
    )

    # 3. 行业集中度（20分）
    # 使用赫芬达尔指数（HHI）衡量集中度
    hhi = sum((market_share ** 2) for market_share in sector_data['market_share'])
    concentration_score = min(hhi / 0.25, 1) * 20  # HHI 0.25满分

    # 4. 行业景气度（20分）
   景气度评分 = (
        min(sector_data['capacity_utilization'].mean() / 0.9, 1) * 10 +  # 产能利用率
        min(sector_data['order_growth'].mean() / 0.3, 1) * 10          # 订单增长
    )

    total_score = sector_growth + sector_profit + concentration_score +景气度评分

    return total_score
```

#### 公司价值相对评分

```python
def calc_relative_value_score(stock_data, sector_data):
    """
    计算公司相对行业的价值评分（0-100）
    """
    # 相对行业的ROE
    roe_relative = stock_data['roe'] / sector_data['roe'].median()

    # 相对行业的毛利率
    margin_relative = stock_data['gross_margin'] / sector_data['gross_margin'].median()

    # 相对行业的营收增长
    growth_relative = stock_data['revenue_growth_yoy'] / sector_data['revenue_growth_yoy'].median()

    # 相对行业的估值
    valuation_relative = sector_data['pe_ttm'].median() / stock_data['pe_ttm']

    # 综合评分
    relative_score = (
        min(roe_relative, 1.5) * 30 +
        min(margin_relative, 1.5) * 25 +
        min(growth_relative, 1.5) * 25 +
        min(valuation_relative, 1.5) * 20
    )

    return relative_score
```

---

### 8.5 风险控制：防止模型退化

#### 风险指标监控

```python
def monitor_model_risk(selected_stocks, benchmark):
    """
    监控模型风险，防止退化
    """
    # 1. 市值分布检查
    avg_mv = selected_stocks['total_mv'].mean()
    if avg_mv < 50000000000:  # 平均市值 < 50亿
        return {
            "risk_level": "HIGH",
            "risk_type": "SMALL_CAP_BIAS",
            "message": "模型可能退化成选小股票"
        }

    # 2. 波动率检查
    avg_volatility = selected_stocks['pct_chg'].rolling(60).std().mean()
    if avg_volatility > 0.25:  # 平均波动率 > 25%
        return {
            "risk_level": "HIGH",
            "risk_type": "HIGH_VOLATILITY_BIAS",
            "message": "模型可能退化成选高波动股票"
        }

    # 3. 基本面检查
    avg_roe = selected_stocks['roe'].mean()
    if avg_roe < 0.05:  # 平均ROE < 5%
        return {
            "risk_level": "HIGH",
            "risk_type": "POOR_QUALITY_BIAS",
            "message": "模型可能选出低质量公司"
        }

    # 4. 与基准的相关性检查
    selected_returns = selected_stocks['pct_chg'].mean()
    benchmark_returns = benchmark['pct_chg']
    correlation = selected_returns.rolling(20).corr(benchmark_returns).iloc[-1]

    if correlation < 0.3:  # 相关性 < 0.3
        return {
            "risk_level": "MEDIUM",
            "risk_type": "LOW_CORRELATION",
            "message": "选股结果与市场脱节"
        }

    return {
        "risk_level": "LOW",
        "risk_type": "NONE",
        "message": "模型运行正常"
    }
```

#### 动态调整机制

```python
def dynamic_adjustment(risk_monitor_result):
    """
    根据风险监控结果动态调整
    """
    if risk_monitor_result["risk_level"] == "HIGH":
        # 高风险：大幅提高市值门槛
        return {
            "action": "ADJUST_PARAMS",
            "new_cap_threshold": 200000000000,  # 提高到200亿
            "new_value_threshold": 70,            # 提高价值门槛到70分
            "position_reduction": 0.5            # 降低仓位50%
        }
    elif risk_monitor_result["risk_level"] == "MEDIUM":
        # 中等风险：适度提高门槛
        return {
            "action": "ADJUST_PARAMS",
            "new_cap_threshold": 100000000000,  # 提高到100亿
            "new_value_threshold": 60,            # 提高价值门槛到60分
            "position_reduction": 0.3            # 降低仓位30%
        }
    else:
        # 低风险：保持原参数
        return {
            "action": "KEEP_PARAMS",
            "new_cap_threshold": None,
            "new_value_threshold": None,
            "position_reduction": 0
        }
```

---

### 8.6 总结：如何避免模型退化

#### 核心原则

1. **硬性约束必须满足**：市值、流动性、基本面、风险
2. **价值评分作为基准**：不是只看价格，必须看价值
3. **机构认可作为加分**：不是游资炒作，是机构认可
4. **风险监控实时反馈**：一旦发现退化，立即调整

#### 对比：退化前 vs 退化后

| 维度 | 退化前（纯博弈） | 退化后（价值增强） |
|------|----------------|------------------|
| 选股标准 | 涨幅、成交额 | 价值、机构认可、涨幅 |
| 市值偏好 | 小盘股（<50亿） | 中大盘股（>100亿） |
| 基本面要求 | 低 | ROE>0，净利润>0 |
| 风险控制 | 弱 | 强（回撤<40%，波动率<30%） |
| 持仓周期 | 短（1-2周） | 中长（1-3个月） |

#### 最终目标

**不只是抓住行业逻辑，还要抓住行业里真正有价值的公司**：

- 行业逻辑 = 抓住趋势
- 公司价值 = 把握确定性
- 两者结合 = 可持续的Alpha

---

*文档版本：v3.0*
*更新日期：2026-02-10*
*作者：iFlow GLM*
