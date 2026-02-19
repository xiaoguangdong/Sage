#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tushare数据完整性校验器

功能：
1. 读取tushare_tasks.yaml配置
2. 检查每个任务的数据文件是否存在
3. 对于时间序列数据，检查2016-2026年的数据完整性
4. 输出缺失数据报告
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class DataIntegrityChecker:
    """数据完整性校验器"""

    def __init__(
        self,
        config_path: Path,
        data_root: Path,
        start_year: int = 2016,
        end_year: int = 2026,
    ):
        """初始化

        Args:
            config_path: tushare_tasks.yaml配置文件路径
            data_root: 数据根目录
            start_year: 起始年份
            end_year: 结束年份
        """
        self.config_path = config_path
        self.data_root = data_root
        self.start_year = start_year
        self.end_year = end_year

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.tasks = self.config.get('tasks', {})

    def _find_actual_file(self, expected_path: Path) -> Optional[Path]:
        """查找实际的文件路径（支持多种命名模式）

        Args:
            expected_path: 配置中期望的路径

        Returns:
            实际文件路径，如果不存在返回None
        """
        # 1. 直接检查期望路径
        if expected_path.exists():
            return expected_path

        # 2. 检查 _all 后缀版本
        if expected_path.suffix == '.parquet':
            all_version = expected_path.parent / f"{expected_path.stem}_all.parquet"
            if all_version.exists():
                return all_version

        # 3. 检查同名目录（如 daily.parquet -> daily/ 目录）
        stem = expected_path.stem
        same_name_dir = expected_path.parent / stem
        if same_name_dir.exists() and same_name_dir.is_dir():
            # 检查目录下是否有parquet文件
            parquet_files = list(same_name_dir.glob("*.parquet"))
            if parquet_files:
                return same_name_dir

        # 4. 检查按年份分割的文件（如 daily/daily_2020.parquet）
        if expected_path.parent.exists():
            # 查找同名目录下的年份文件
            parent = expected_path.parent
            year_files = list(parent.glob(f"{stem}_20*.parquet"))
            if year_files:
                return parent

        # 5. 检查父目录下的同名文件
        parent_file = self.data_root / f"{stem}_all.parquet"
        if parent_file.exists():
            return parent_file

        # 6. 检查 data/raw/tushare 目录（旧数据位置）
        raw_path = self.data_root.parent / "raw" / "tushare" / expected_path.relative_to(self.data_root)
        if raw_path.exists():
            return raw_path

        return None

    def check_all(self) -> Dict[str, Dict]:
        """检查所有任务的数据完整性

        Returns:
            检查结果字典
        """
        results = {}

        print(f"开始检查数据完整性 ({self.start_year}-{self.end_year})")
        print("=" * 80)

        for task_name, task_config in self.tasks.items():
            print(f"\n检查任务: {task_name}")
            result = self._check_task(task_name, task_config)
            results[task_name] = result

        return results

    def _check_task(self, task_name: str, task_config: Dict) -> Dict:
        """检查单个任务的数据完整性

        Args:
            task_name: 任务名称
            task_config: 任务配置

        Returns:
            检查结果
        """
        output_path = self.data_root / task_config['output']
        mode = task_config.get('mode', 'single')

        result = {
            'task_name': task_name,
            'mode': mode,
            'output_path': str(output_path),
            'file_exists': output_path.exists(),
            'missing_data': [],
            'status': 'unknown',
        }

        # 1. 检查文件是否存在（支持多种命名模式）
        actual_path = self._find_actual_file(output_path)

        if actual_path is None:
            result['status'] = 'missing_file'
            print(f"  ❌ 文件不存在: {output_path}")
            return result

        # 更新为实际路径
        output_path = actual_path
        result['actual_path'] = str(actual_path)
        result['file_exists'] = True

        # 2. 读取数据（合并所有能找到的数据源）
        try:
            dfs = []
            sources = []

            if actual_path.is_dir():
                # 读取目录下所有parquet文件
                parquet_files = list(actual_path.glob("*.parquet"))
                if not parquet_files:
                    result['status'] = 'empty_directory'
                    print(f"  ❌ 目录为空: {actual_path}")
                    return result
                for pf in parquet_files:
                    dfs.append(pd.read_parquet(pf))
                sources.append(f"目录({len(parquet_files)}个文件)")
            else:
                dfs.append(pd.read_parquet(actual_path))
                sources.append(actual_path.name)

            # 同时查找 _all 版本和分片目录，合并更完整的数据
            stem = Path(task_config['output']).stem
            parent = self.data_root / Path(task_config['output']).parent

            # 查找 _all 文件
            all_file = parent / f"{stem}_all.parquet"
            if all_file.exists() and str(all_file) != str(actual_path):
                dfs.append(pd.read_parquet(all_file))
                sources.append(f"{all_file.name}")

            # 查找同名分片目录
            split_dir = parent / stem
            if split_dir.exists() and split_dir.is_dir() and str(split_dir) != str(actual_path):
                split_files = list(split_dir.glob("*.parquet"))
                if split_files:
                    for sf in split_files:
                        dfs.append(pd.read_parquet(sf))
                    sources.append(f"{stem}/({len(split_files)}个分片)")

            # 合并去重
            df = pd.concat(dfs, ignore_index=True)
            dedup_keys = task_config.get('dedup_keys')
            if dedup_keys and all(k in df.columns for k in dedup_keys):
                df = df.drop_duplicates(subset=dedup_keys, keep='last')

            result['record_count'] = len(df)
            result['sources'] = sources
            print(f"  ✅ 数据来源: {', '.join(sources)}，记录数: {len(df):,}")
        except Exception as e:
            result['status'] = 'read_error'
            result['error'] = str(e)
            print(f"  ❌ 读取失败: {e}")
            return result

        # 3. 根据模式检查数据完整性
        if mode == 'single':
            # 单次下载任务，只检查是否有数据
            result['status'] = 'ok' if len(df) > 0 else 'empty'

        elif mode == 'date_range':
            # 日期范围任务，检查时间覆盖
            result = self._check_date_range(df, task_config, result)

        elif mode == 'year_quarters':
            # 季度任务，检查季度覆盖
            result = self._check_year_quarters(df, task_config, result)

        elif mode == 'list':
            # 列表任务，检查时间覆盖（如果有时间字段）
            result = self._check_list_mode(df, task_config, result)

        return result

    def _check_date_range(self, df: pd.DataFrame, task_config: Dict, result: Dict) -> Dict:
        """检查日期范围任务的数据完整性

        检查三个维度：
        1. 起始日期是否覆盖目标起始
        2. 结束日期是否覆盖当前日期（而非遥远的未来）
        3. 中间是否有年份空洞
        """
        # 查找日期字段
        date_field = None
        for field in ['trade_date', 'ann_date', 'end_date']:
            if field in df.columns:
                date_field = field
                break

        if date_field is None:
            result['status'] = 'no_date_field'
            print(f"  ⚠️  未找到日期字段")
            return result

        # 转换日期格式
        df[date_field] = pd.to_datetime(df[date_field], format='%Y%m%d', errors='coerce')

        # 获取数据时间范围
        min_date = df[date_field].min()
        max_date = df[date_field].max()

        result['min_date'] = min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else None
        result['max_date'] = max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else None

        print(f"  📅 时间范围: {result['min_date']} ~ {result['max_date']}")

        if pd.isna(min_date) or pd.isna(max_date):
            result['status'] = 'invalid_dates'
            return result

        # 用当前日期作为结束目标（而非 end_year 年底，未来数据不可能有）
        target_start = datetime(self.start_year, 1, 1)
        now = datetime.now()
        target_end = min(datetime(self.end_year, 12, 31), now - timedelta(days=7))

        issues = []

        # 1. 检查起始日期
        if min_date > target_start + timedelta(days=30):
            issues.append(f"缺少早期数据: {target_start.year}年初 ~ {min_date.strftime('%Y-%m-%d')}")

        # 2. 检查结束日期
        if max_date < target_end:
            issues.append(f"缺少近期数据: {max_date.strftime('%Y-%m-%d')} ~ {target_end.strftime('%Y-%m-%d')}")

        # 3. 检查中间年份空洞
        years_with_data = set(df[date_field].dt.year.dropna().unique())
        expected_years = set(range(max(self.start_year, min_date.year), min(self.end_year, max_date.year) + 1))
        missing_years = sorted(expected_years - years_with_data)
        if missing_years:
            issues.append(f"中间年份空洞: {', '.join(str(y) for y in missing_years)}")

        # 4. 按年统计记录数
        year_counts = df[date_field].dt.year.value_counts().sort_index()
        result['year_counts'] = {int(y): int(c) for y, c in year_counts.items()}

        if issues:
            result['status'] = 'incomplete'
            result['missing_data'] = issues
            for issue in issues:
                print(f"  ⚠️  {issue}")
        else:
            result['status'] = 'ok'
            print(f"  ✅ 数据完整")

        return result

    def _check_year_quarters(self, df: pd.DataFrame, task_config: Dict, result: Dict) -> Dict:
        """检查季度任务的数据完整性

        Args:
            df: 数据DataFrame
            task_config: 任务配置
            result: 当前结果

        Returns:
            更新后的结果
        """
        # 查找季度字段
        period_field = None
        for field in ['end_date', 'period', 'f_ann_date']:
            if field in df.columns:
                period_field = field
                break

        if period_field is None:
            result['status'] = 'no_period_field'
            print(f"  ⚠️  未找到季度字段")
            return result

        # 提取已有的季度
        df['period_str'] = df[period_field].astype(str).str[:8]
        existing_periods = set(df['period_str'].unique())

        # 生成目标季度列表
        start_year = task_config.get('start_year', self.start_year)
        end_year = task_config.get('end_year', self.end_year)
        quarters = task_config.get('quarters', ['0331', '0630', '0930', '1231'])

        target_periods = []
        for year in range(max(start_year, self.start_year), min(end_year, self.end_year) + 1):
            for quarter in quarters:
                target_periods.append(f"{year}{quarter}")

        # 检查缺失的季度
        missing_periods = [p for p in target_periods if p not in existing_periods]

        result['target_periods'] = len(target_periods)
        result['existing_periods'] = len(existing_periods)
        result['missing_periods'] = missing_periods

        if missing_periods:
            result['status'] = 'incomplete'
            result['missing_data'] = missing_periods
            print(f"  ⚠️  缺失季度: {len(missing_periods)}/{len(target_periods)}")
            print(f"     {', '.join(missing_periods[:5])}{'...' if len(missing_periods) > 5 else ''}")
        else:
            result['status'] = 'ok'
            print(f"  ✅ 季度完整: {len(target_periods)}/{len(target_periods)}")

        return result

    def _check_list_mode(self, df: pd.DataFrame, task_config: Dict, result: Dict) -> Dict:
        """检查列表模式任务的数据完整性

        Args:
            df: 数据DataFrame
            task_config: 任务配置
            result: 当前结果

        Returns:
            更新后的结果
        """
        # 列表模式任务，如果有时间字段则检查时间覆盖
        if 'start_field' in task_config and 'end_field' in task_config:
            return self._check_date_range(df, task_config, result)
        else:
            # 没有时间字段，只检查是否有数据
            result['status'] = 'ok' if len(df) > 0 else 'empty'
            print(f"  ✅ 数据存在")
            return result

    def generate_report(self, results: Dict[str, Dict]) -> str:
        """生成检查报告

        Args:
            results: 检查结果

        Returns:
            报告文本
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Tushare数据完整性检查报告")
        report_lines.append(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"目标年份: {self.start_year}-{self.end_year}")
        report_lines.append("=" * 80)

        # 统计
        total = len(results)
        ok_count = sum(1 for r in results.values() if r['status'] == 'ok')
        incomplete_count = sum(1 for r in results.values() if r['status'] == 'incomplete')
        missing_count = sum(1 for r in results.values() if r['status'] == 'missing_file')
        error_count = sum(1 for r in results.values() if r['status'] in ['read_error', 'invalid_dates'])

        report_lines.append(f"\n总任务数: {total}")
        report_lines.append(f"  ✅ 完整: {ok_count}")
        report_lines.append(f"  ⚠️  不完整: {incomplete_count}")
        report_lines.append(f"  ❌ 文件缺失: {missing_count}")
        report_lines.append(f"  ❌ 读取错误: {error_count}")

        # 详细信息
        report_lines.append("\n" + "=" * 80)
        report_lines.append("详细信息")
        report_lines.append("=" * 80)

        # 按状态分组
        status_groups = defaultdict(list)
        for task_name, result in results.items():
            status_groups[result['status']].append((task_name, result))

        # 1. 文件缺失
        if status_groups['missing_file']:
            report_lines.append("\n【文件缺失】")
            for task_name, result in status_groups['missing_file']:
                report_lines.append(f"  - {task_name}: {result['output_path']}")

        # 2. 数据不完整
        if status_groups['incomplete']:
            report_lines.append("\n【数据不完整】")
            for task_name, result in status_groups['incomplete']:
                report_lines.append(f"\n  {task_name}:")
                if result.get('min_date') and result.get('max_date'):
                    report_lines.append(f"    实际范围: {result['min_date']} ~ {result['max_date']}  ({result.get('record_count', '?'):,} 条)")
                if result.get('year_counts'):
                    years_str = ', '.join(f"{y}:{c:,}" for y, c in sorted(result['year_counts'].items()))
                    report_lines.append(f"    按年分布: {years_str}")
                if result.get('missing_periods'):
                    report_lines.append(f"    缺失季度: {len(result['missing_periods'])}个")
                    report_lines.append(f"    {', '.join(result['missing_periods'][:10])}")
                    if len(result['missing_periods']) > 10:
                        report_lines.append(f"    ... 还有 {len(result['missing_periods']) - 10} 个")
                elif result.get('missing_data'):
                    for missing in result['missing_data']:
                        report_lines.append(f"    - {missing}")

        # 3. 读取错误
        if status_groups['read_error'] or status_groups['invalid_dates']:
            report_lines.append("\n【读取错误】")
            for task_name, result in list(status_groups['read_error']) + list(status_groups['invalid_dates']):
                report_lines.append(f"  - {task_name}: {result.get('error', result['status'])}")

        # 4. 完整数据（简要列出）
        if status_groups['ok']:
            report_lines.append(f"\n【数据完整】({len(status_groups['ok'])}个任务)")
            for task_name, result in status_groups['ok'][:5]:
                report_lines.append(f"  ✅ {task_name}")
            if len(status_groups['ok']) > 5:
                report_lines.append(f"  ... 还有 {len(status_groups['ok']) - 5} 个任务")

        report_lines.append("\n" + "=" * 80)

        return "\n".join(report_lines)


def main():
    """主函数"""
    config_path = ROOT / "config/tushare_tasks.yaml"
    data_root = ROOT / "data/tushare"

    checker = DataIntegrityChecker(
        config_path=config_path,
        data_root=data_root,
        start_year=2016,
        end_year=2026,
    )

    # 执行检查
    results = checker.check_all()

    # 生成报告
    report = checker.generate_report(results)
    print("\n" + report)

    # 保存报告
    report_path = ROOT / "logs/data/data_integrity_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
