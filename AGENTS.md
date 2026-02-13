# Sage / 量化与宏观数据项目 - Agent 约定

本仓库以“宏观数据获取/清洗 → 行业映射 → 回测与预测”为主。Agent 在这里工作的目标是：**少猜、可复现、可回滚**，并尽量把约定固化为脚本与文件规范。

## 1) 目录与数据规范（优先遵循现有文档）
- 宏观数据规范以 `scripts/macro/README.md` 为准（文件名、字段、目录结构）。
- 主要数据根目录：`data/tushare/`
  - 宏观：`data/tushare/macro/`
  - 北向：`data/tushare/northbound/`
  - 行业/概念：`data/tushare/sectors/`、`data/tushare/concepts/`
- 行业映射配置：
  - `config/sw_nbs_mapping.yaml`（申万L1）
  - `config/sw_nbs_mapping_l2.yaml`（申万L2）

## 2) 运行方式与环境（尽量显式）
- Python 虚拟环境：
  - `venv/`（当前为 Python 3.14.x）
  - `venv310/`（Python 3.10.x，给兼容性/编译依赖用）
- 运行脚本优先用明确解释器，避免“系统 python”漂移：
  - `./venv/bin/python scripts/macro/...`
  - `./venv310/bin/python ...`（仅在必须时）

## 3) Tushare / 账号密钥（强约束）
- **禁止**在代码、文档、日志里硬编码 token/密码。
- Tushare token 统一从环境变量读取：
  - `TUSHARE_TOKEN`（推荐）或 `TS_TOKEN`
- 仓库内如存在账号文件（例如 `config/joinquant.json`），视为**本地私密配置**：不应提交到版本库；如需共享请提供 `*.example.json` 模板。

## 4) 速率限制与长任务习惯（来自历史会话经验）
- Tushare 常见约束：IP 数量限制（最多 2 IP）+ 请求间隔（脚本中多处已按 40–60s sleep）。
- 长任务一律：
  - 输出重定向到日志（`logs/` 或 `data/tushare/**/xxx.log`）
  - 后台执行用 `nohup ... &`
  - 用 `tail -f` 追踪日志
- 并发抓取要保守：不要为了“更快”擅自提高并发或缩短 sleep，除非先确认接口配额与 IP 限制。

## 5) 常用入口（优先复用现成脚本）
- 一键拉取宏观数据：`./scripts/macro/fetch_all_macro_data.sh`
- 宏观数据完整性检查：`./scripts/macro/check_macro_data.sh`
- 定时任务参考：`scripts/macro/cron_macro_schedule.conf`、`scripts/macro/DEPLOYMENT.md`

## 6) 变更原则（写代码时默认遵守）
- 先读现有脚本/文档再改：尤其是 `scripts/macro/README.md`、`scripts/macro/DEPLOYMENT.md`。
- 只做与任务相关的最小改动；不要顺手重构大范围。
- 任何破坏性操作（`rm -rf`、`pkill -9`、清空数据目录）在未被明确要求时**先问一句**再做。
- 输出文件要可重跑：尽量断点续传、增量更新，不要默认全量重写。
- 搜索/批处理时默认排除 `venv/`、`venv310/`、`data/`、`logs/`（它们不是源码）。

## 7) 数据备份机制（本机）
- 代码进 Git；`data/` 不进 Git（见 `.gitignore`）。
- `data/` 镜像备份到外置盘（默认）：`/Volumes/SPEED/BizData/Stock/Sage/data/`
- 触发方式：
  - Git hooks（提交/切换/合并后同步）：`./scripts/backup/install_git_hooks.sh`
  - macOS launchd 监听 `data/` 变化即同步：`./scripts/backup/install_launchd_watch.sh`
