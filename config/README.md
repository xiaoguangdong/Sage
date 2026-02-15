# 配置目录说明

## 分层

- `base.yaml`：全局运行时配置（数据根、日志、下载参数）。
- `tushare_tasks.yaml`：统一下载器任务定义。
- `app/`：应用编排层配置（趋势/选股/治理/风控）。
- 其余 `policy_*`、`sw_*`、`nbs_*`：领域映射与规则配置。

## 路径约定

- Tushare 主数据根：`data/tushare/`（通过 `base.yaml -> data.paths.tushare` 控制）。
- 兼容回退目录：`data/raw/tushare/`（仅用于历史数据读取，不作为默认写入）。
