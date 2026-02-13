# Tushare批量下载定时任务

## 问题描述

当前Tushare账号触发IP数量超限（最大2个），需要等待30分钟到1小时后才能继续下载。

## 定时任务配置

已设置在**10:30**自动运行下载任务。

## 定时任务管理

### 查看定时任务
```bash
at -l
```

### 查看定时任务详情
```bash
at -c 1  # 1是任务ID
```

### 取消定时任务
```bash
atrm 1  # 1是任务ID
```

### 重新设置定时任务
```bash
echo "/Users/dongxg/SourceCode/deep_final_kp/scripts/schedule_tushare_download.sh" | at 10:30
```

## 手动运行

如果需要立即运行下载任务：
```bash
/Users/dongxg/SourceCode/deep_final_kp/scripts/schedule_tushare_download.sh
```

## 查看日志

```bash
tail -f /Users/dongxg/SourceCode/deep_final_kp/data/tushare/download_batch.log
```

## 每天定时运行

如果需要每天定时运行，可以添加到crontab：

```bash
crontab -e
```

添加以下内容：
```bash
30 10 * * * /Users/dongxg/SourceCode/deep_final_kp/scripts/schedule_tushare_download.sh
```

## 建议下载时间

- **最佳时间**：10:30之后（避开早盘时段，减少API限流）
- **避开时段**：
  - 9:15-9:30（集合竞价）
  - 9:30-11:30（上午交易时段）
  - 13:00-15:00（下午交易时段）

## 注意事项

1. 每次运行前会自动清空状态文件
2. 已下载的数据会自动跳过（断点续传）
3. 如果遇到IP限制，脚本会自动重试3次
4. 建议每天只运行一次，避免频繁触发限流