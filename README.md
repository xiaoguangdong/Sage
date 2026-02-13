# Sage

量化与宏观数据相关脚本与模型实验仓库。

## 快速开始

1. 创建/激活虚拟环境（示例）：
   - `python3 -m venv venv && source venv/bin/activate`
2. 设置数据源 token（不要写进代码/仓库）：
   - `export TUSHARE_TOKEN=...`
3. 宏观数据脚本入口：
   - `./scripts/macro/fetch_all_macro_data.sh`
   - `./scripts/macro/check_macro_data.sh`

## 数据不丢失机制（推荐启用）

本仓库默认不把 `data/` 提交到 Git（见 `.gitignore`），建议使用外置盘进行镜像备份：
- 默认目标：`/Volumes/SPEED/BizData/Stock/Sage/data/`

两种触发方式（二选一或都启用）：
1. **Git 提交触发**（提交/切换分支/拉取后自动同步一次）：
   - `./scripts/backup/install_git_hooks.sh`
2. **数据变更触发（macOS）**：监控 `./data` 有变化就同步：
   - `./scripts/backup/install_launchd_watch.sh`

