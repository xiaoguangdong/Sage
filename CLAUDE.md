# Claude 工作流程规范


# Non-negotiables
1. **代码/文档修改** → 2. **更新TODO** → 3. **Git提交** → 4. **Push到远程**

## 核心原则

每次完成任务后，必须按照以下顺序执行：

1. **代码/文档修改** → 2. **更新TODO** → 3. **Git提交** → 4. **Push到远程**


## 详细工作流程

### 1. 完成代码或文档修改
- 实现功能代码
- 修复bug
- 更新文档
- 添加测试

### 2. 更新TODO文档
**文件**：`docs/2.0 Sage股票智能交易平台TODO_LIST.md`

**更新规则**：
- 任务完成：将 `[ ]` 改为 `[x]`
- 发现新问题：添加到对应优先级章节
- 实现状态变化：更新"实现状态总览"表格
- 重大里程碑：更新"关键指标目标"表格

**更新频率**：
- P0任务完成：立即更新
- P1任务完成：当天更新
- 每周五：全面回顾和调整优先级

### 3. Git提交
**提交信息格式**：
```
<type>(<scope>): <subject>

<body>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

**Type类型**：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具/依赖更新
- `perf`: 性能优化
- `style`: 代码格式（不影响功能）

**Scope范围**：
- `选股`: stock_selection相关
- `回测`: backtest相关
- `趋势`: trend相关
- `治理`: governance相关
- `执行`: execution相关
- `数据`: data相关
- `监控`: monitoring相关
- `配置`: config相关

**示例**：
```bash
git add .
git commit -m "$(cat <<'EOF'
docs(TODO): 更新TODO文档，清理重复内容并补充实现状态

- 从706行精简到549行
- 新增实现状态总览表
- 补充关键文件清单
- 明确维护规则和更新频率

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

### 4. Push到远程
```bash
git push origin main
```

**注意事项**：
- 确保在main分支
- 如果push失败，先pull再push
- 重要功能可以创建feature分支

## 特殊情况处理

### 实验性代码
- 创建feature分支：`git checkout -b feature/xxx`
- 完成后合并到main：`git merge feature/xxx`

### 紧急修复
- 创建hotfix分支：`git checkout -b hotfix/xxx`
- 修复后立即合并并push

### 大型重构
- 分阶段提交，每个阶段一个commit
- 每个commit保持功能完整可运行
- 及时push避免丢失工作

## 检查清单

每次任务完成前检查：
- [ ] 代码/文档修改完成
- [ ] TODO文档已更新
- [ ] Git commit已创建（包含Co-Authored-By）
- [ ] 已push到远程仓库
- [ ] 本地工作目录干净（git status无未提交文件）

## 项目特定规则

### 数据文件
- 不提交 `data/` 目录下的数据文件
- 不提交 `models/` 目录下的模型文件
- 不提交 `logs/` 目录下的日志文件
- 确保 `.gitignore` 正确配置

### 配置文件
- 提交 `config/` 目录下的配置模板
- 不提交包含敏感信息的配置（如API密钥）
- 使用 `.env.example` 作为环境变量模板

### 文档
- 每次功能更新必须更新对应文档
- 保持 `docs/` 目录结构清晰
- README.md 保持最新

---

## 项目关键信息

### 项目状态
- **完成度**：约80%
- **核心选股系统**：95%完成，可直接使用
- **最佳模型**：RegimeStockSelector（Sharpe 2.20）
- **代码规模**：18,279行Python代码

### 核心架构
```
sage_core/          # 核心模块（完整可用）
├── stock_selection/    # 选股系统（95%完成）
│   ├── stock_selector.py              # 基础选股器（完整）
│   ├── regime_stock_selector.py       # Regime选股器（完整，Sharpe 2.20）
│   └── multi_alpha_stock_selector.py  # 多策略融合（完整）
├── trend/              # 趋势模型
│   └── trend_model.py                 # TrendModelRuleV2（完整）
├── portfolio/          # 组合管理
│   ├── portfolio_manager.py           # 组合管理器（完整）
│   └── risk_control.py                # 风险控制（完整）
├── governance/         # 治理框架
│   └── strategy_governance.py         # Champion/Challenger（完整）
├── backtest/           # 回测引擎
│   ├── walk_forward.py                # Walk-Forward回测（完整）
│   └── simple_backtest.py             # 简单回测（完整）
└── execution/          # 执行系统
    ├── broker_adapter.py              # 券商适配器（框架完整）
    ├── order_lifecycle.py             # 订单生命周期（完整）
    └── unified_signal_contract.py     # 统一信号契约（完整）
```

### 主要短板（需要优先处理）
1. **买卖点模型**：仅有占位符，无实际实现
2. **交易成本模型**：仅有单边成本率，缺乏滑点和市场冲击
3. **宏观模型集成**：数据处理在scripts/，未集成到sage_core/
4. **行业模型V2**：主线模型V2的四维评分引擎未实现
5. **自动化调度**：配置存在，但DAG调度逻辑不完整

### 关键文件位置
- **TODO文档**：`docs/2.0 Sage股票智能交易平台TODO_LIST.md`
- **总架构文档**：`docs/2.1 Sage股票智能交易平台总架构设计文档.md`
- **配置文件**：`config/base.yaml`、`config/tushare_tasks.yaml`
- **数据目录**：`data/`（不提交到Git）
- **模型目录**：`models/`（不提交到Git）

### 数据源
- **主要数据源**：Tushare Pro
- **补充数据源**：同花顺（概念/板块）、东方财富（研报）、政府网站（政策）
- **数据更新**：需要手动运行下载脚本
- **历史数据**：2020-2025（部分数据需补充2016-2019）

### 回测关键指标
- **Regime选股器**（2024-09 ~ 2026-01）：
  - 累计收益：+80.69%
  - 沪深300基准：+44.88%
  - 超额收益：+35.81%
  - Sharpe比率：2.20
- **防过拟合机制**：IC筛选、Purge/Embargo、Walk-Forward、分组回测

### 当前优先级（P0任务）
1. 完成2016-2019历史数据补充
2. 实现完整的交易成本模型（滑点+市场冲击）
3. 验证回测系统无数据泄露
4. 解决概念数据获取问题（Tushare IP限制）
5. 接入confidence动态仓位到回测
6. 实现个股ATR动态止损
7. 实现组合层面分档降仓
8. 实现买卖点模型基础框架

### Git Hook自动化
- **数据同步**：每次commit自动触发数据同步到外部存储
- **目标路径**：`/Volumes/SPEED/BizData/Stock/Sage/data/`
- **同步工具**：rsync

### 常用命令
```bash
# 数据下载
python scripts/data/tushare_downloader.py

# 历史数据补充
python scripts/data/backfill_missing_data.py --mode daily --start_date 20160101 --end_date 20191231

# 回测
python scripts/backtest/backtest_champion_challenger.py

# 数据质量检查
python scripts/monitoring/check_data_quality.py
```

### 注意事项
1. **每次任务完成必须**：修改代码 → 更新TODO → Git提交 → Push
2. **提交信息必须包含**：Co-Authored-By: Claude Sonnet 4.6
3. **不要提交**：data/、models/、logs/ 目录
4. **TODO文档**：P0任务完成立即更新，P1任务当天更新，每周五全面回顾
5. **模型文件**：虽然不提交，但要确保本地备份

---

**最后更新**：2026-02-18
**适用范围**：Sage股票智能交易平台项目
