# Data scripts

数据下载、清洗、对齐相关脚本放在这里。

## 运行前准备

- 设置 Tushare Token 环境变量：`export TUSHARE_TOKEN=xxxx`
- 或在项目根目录创建 `.env`：`TUSHARE_TOKEN=xxxx`
- 日志默认输出到：`logs/data/YYYYMMDD_NNN_<script>.log`

## Tushare 下载器（daily_basic / margin）

```bash
python scripts/data/tushare_downloader.py --endpoint daily_basic \
  --start-date 2020-01-01 --end-date 2024-12-31 \
  --output-dir /tmp/sage_data --resume

python scripts/data/tushare_downloader.py --endpoint margin \
  --start-date 2020-01-01 --end-date 2024-12-31 \
  --output-dir /tmp/sage_data --resume
```

输出分片目录：
- `/tmp/sage_data/daily_basic/parts/`
- `/tmp/sage_data/margin/parts/`
