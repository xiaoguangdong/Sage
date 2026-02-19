# 申万-NBS行业映射系统说明

## 概述

本系统实现了将国家统计局（NBS）宏观数据映射到申万SW2021一级行业的功能，解决了宏观数据（统计局口径）与微观数据（股票/申万口径）的对应问题。

## 背景

在量化投研中，"宏观-中观映射"是一个经典且重要的问题：
- **申万行业（SW）**：按上市公司业务划分，用于股票分类和投资分析
- **统计局行业（NBS）**：按国民经济活动性质划分，遵循《国民经济行业分类》GB/T 4754

两者分类标准不同，需要建立映射关系才能将宏观数据应用到行业分析中。

## 解决方案

采用**营收暴露权重矩阵**方法（方案一）：

### 核心逻辑

1. **构建映射权重矩阵**：为每个申万行业定义其在各个NBS行业上的营收暴露度权重
2. **加权合成指标**：根据权重将NBS宏观数据合成申万行业指标

### 公式

```
SW_Indicator_i = Σ(Weight_ij × NBS_Indicator_j)
```

其中：
- `SW_Indicator_i`：申万行业i的合成指标
- `Weight_ij`：申万行业i在NBS行业j上的权重
- `NBS_Indicator_j`：NBS行业j的原始指标

权重满足：Σ(Weight_ij) = 1.0

## 系统架构

### 1. 配置文件

**文件路径**：`config/sw_nbs_mapping.yaml`

**内容**：
- `sw_to_nbs`：申万行业到NBS行业的映射关系和权重
- `metadata`：配置元数据（版本、创建日期、说明等）

**映射示例**：

```yaml
"钢铁":
  - nbs_industry: "黑色金属冶炼和压延加工业"
    weight: 0.70
  - nbs_industry: "黑色金属矿采选业"
    weight: 0.30
```

说明：钢铁行业 = 70%的黑色金属冶炼 + 30%的黑色金属矿采选

### 2. 映射工具类

**文件路径**：`scripts/data/macro/industry_mapper.py`

**主要功能**：
- 加载映射配置
- 模糊匹配NBS行业名称
- 映射NBS数据到申万行业
- 验证映射配置
- 生成映射摘要

**使用方法**：

```python
from scripts.macro.industry_mapper import IndustryMapper

# 初始化映射器
mapper = IndustryMapper()

# 映射NBS数据
sw_data = mapper.map_nbs_data_to_sw(
    nbs_data=nbs_df,
    value_col='ppi_yoy'
)
```

### 3. 批量处理脚本

**文件路径**：`scripts/data/macro/map_nbs_to_sw.py`

**功能**：
- 读取NBS PPI数据
- 使用映射配置将NBS数据映射到申万行业
- 保存映射后的申万行业数据

**使用方法**：

```bash
# 运行映射脚本
python3 scripts/data/macro/map_nbs_to_sw.py
```

**输出文件**：`data/tushare/macro/sw_ppi_yoy_YYYYMM.csv`

**输出格式**：

| sw_industry | sw_ppi_yoy | date | source_nbs_count |
|-------------|------------|------|------------------|
| 钢铁 | -7.01 | 2025-12-01 | 2 |
| 基础化工 | -5.34 | 2025-12-01 | 3 |
| ... | ... | ... | ... |

### 4. 集成脚本

**文件路径**：`scripts/data/macro/fetch_all_macro_data.sh`

**功能**：一键获取所有宏观数据并完成映射

**使用方法**：

```bash
./scripts/data/macro/fetch_all_macro_data.sh
```

**处理流程**：
1. 获取国家统计局数据（NBS）
2. 获取Tushare宏观数据
3. 获取北上资金数据
4. 映射NBS数据到申万行业

## 映射覆盖范围

### 申万SW2021一级行业（31个）

| 行业 | 映射NBS行业数 | 权重总和 |
|------|---------------|----------|
| 农林牧渔 | 5 | 1.0 |
| 基础化工 | 4 | 1.0 |
| 钢铁 | 2 | 1.0 |
| 有色金属 | 2 | 1.0 |
| 电子 | 3 | 1.0 |
| 汽车 | 3 | 1.0 |
| ... | ... | ... |

完整的映射关系请查看 `config/sw_nbs_mapping.yaml`

### NBS行业数据

当前支持的数据类型：
- **PPI（工业生产者出厂价格指数）**：按行业分类的同比数据
- **未来扩展**：固定资产投资、工业产出、企业利润等

## 使用场景

### 1. 行业景气度分析

```python
# 加载申万行业PPI数据
sw_ppi = pd.read_csv('data/tushare/macro/sw_ppi_yoy_202512.csv')

# 分析行业景气度
top_industries = sw_ppi.nsmallest(5, 'sw_ppi_yoy')
bottom_industries = sw_ppi.nlargest(5, 'sw_ppi_yoy')
```

### 2. 行业轮动策略

```python
# 根据PPI变化选择行业
if sw_ppi[sw_ppi['sw_industry'] == '钢铁']['sw_ppi_yoy'].values[0] < -5:
    # 钢铁行业PPI大幅下降，可能预示需求疲软
    # 考虑减仓或规避
```

### 3. 宏观因子构建

```python
# 构建通胀因子
inflation_factor = sw_ppi['sw_ppi_yoy'].mean()

# 构建行业分化因子
industry_divergence = sw_ppi['sw_ppi_yoy'].std()
```

## 数据质量

### 映射覆盖率

- **申万行业数量**：31个
- **成功映射数量**：27个（87%）
- **未映射数量**：4个（商贸零售、社会服务、建筑装饰、非银金融）

未映射原因：这些行业的NBS数据在当前PPI数据集中不存在

### 数据验证

- ✓ 权重总和验证：所有申万行业权重总和为1.0
- ✓ 行业名称验证：所有NBS行业名称有效
- ✓ 数值范围验证：PPI同比数据在合理范围内

## 未来改进方向

### 1. 动态权重矩阵

当前使用静态权重，未来可以升级为动态权重：

```python
# 基于主营业务分部数据动态计算权重
def calculate_dynamic_weights(sw_industry, stocks_data):
    """
    根据成分股的主营业务分部数据计算权重
    """
    # 1. 获取申万行业的所有成分股
    # 2. 获取每只股票的主营业务分部营收数据
    # 3. 将业务分部映射到NBS行业
    # 4. 计算权重矩阵

    return weight_matrix
```

### 2. 多指标支持

扩展支持更多NBS指标：
- 固定资产投资（FAI）
- 工业增加值
- 企业利润
- 用电量、货运量等先行指标

### 3. 历史数据回溯

构建历史时间序列，支持回测分析：

```python
# 构建申万行业PPI时间序列
for year_month in get_available_periods():
    nbs_data = load_nbs_data(year_month)
    sw_data = map_nbs_to_sw(nbs_data)
    save_to_timeseries(sw_data)
```

### 4. 版本管理

支持不同版本的申万行业分类：
- SW 2014
- SW 2021（当前）

## 注意事项

1. **权重更新**：建议每季度根据最新财报数据更新权重
2. **数据延迟**：NBS数据通常有1-2个月的延迟
3. **行业变更**：申万和NBS行业分类可能随时间调整
4. **映射精度**：不同行业的映射精度不同，需要结合实际情况判断

## 参考文献

1. 《国民经济行业分类》GB/T 4754-2017
2. 申万SW2021行业分类标准
3. 《项目遇到的问题与解决方案.md》- 行业映射方案说明

## 联系方式

如有问题或建议，请联系项目维护者。

---

**文档版本**：1.0
**创建日期**：2026-02-11
**最后更新**：2026-02-11
