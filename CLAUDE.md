# Claude 工作流程规范

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

**最后更新**：2026-02-18
**适用范围**：Sage股票智能交易平台项目
