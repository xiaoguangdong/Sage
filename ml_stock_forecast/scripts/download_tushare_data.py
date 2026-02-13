"""
下载Tushare数据脚本
下载以下数据：
1. moneyflow_hsgt - 北向资金
2. margin - 融资融券
3. fina_indicator - 财务指标
4. daily_basic - 日线基本面指标
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Optional
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_stock_forecast.data.data_provider import DataProvider

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_moneyflow_hsgt(
    provider: DataProvider,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    下载北向资金数据
    
    Args:
        provider: DataProvider实例
        start_date: 开始日期
        end_date: 结束日期（默认为今天）
        
    Returns:
        北向资金DataFrame
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"开始下载北向资金数据: {start_date} ~ {end_date}")
    
    try:
        df = provider.get_moneyflow_hsgt_tushare(start_date, end_date)
        
        if df.empty:
            logger.warning("未获取到北向资金数据，可能需要Tushare积分")
            return df
        
        logger.info(f"成功下载北向资金数据: {len(df)} 条记录")
        return df
        
    except Exception as e:
        logger.error(f"下载北向资金数据失败: {e}")
        return pd.DataFrame()


def download_margin_data(
    provider: DataProvider,
    stock_code: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    下载融资融券数据
    
    Args:
        provider: DataProvider实例
        stock_code: 股票代码，如 600000.SH
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        融资融券DataFrame
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"开始下载融资融券数据: {stock_code}, {start_date} ~ {end_date}")
    
    try:
        df = provider.get_margin_tushare(stock_code, start_date, end_date)
        
        if df.empty:
            logger.warning(f"未获取到融资融券数据: {stock_code}")
            return df
        
        logger.info(f"成功下载融资融券数据: {stock_code}, {len(df)} 条记录")
        return df
        
    except Exception as e:
        logger.error(f"下载融资融券数据失败: {stock_code}, {e}")
        return pd.DataFrame()


def download_fina_indicator(
    provider: DataProvider,
    stock_code: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    下载财务指标数据
    
    Args:
        provider: DataProvider实例
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        财务指标DataFrame
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"开始下载财务指标数据: {stock_code}, {start_date} ~ {end_date}")
    
    try:
        df = provider.get_fina_indicator_tushare(stock_code, start_date, end_date)
        
        if df.empty:
            logger.warning(f"未获取到财务指标数据: {stock_code}")
            return df
        
        logger.info(f"成功下载财务指标数据: {stock_code}, {len(df)} 条记录")
        return df
        
    except Exception as e:
        logger.error(f"下载财务指标数据失败: {stock_code}, {e}")
        return pd.DataFrame()


def download_daily_basic(
    provider: DataProvider,
    stock_code: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    下载日线基本面指标数据
    
    Args:
        provider: DataProvider实例
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        日线基本面DataFrame
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"开始下载日线基本面数据: {stock_code}, {start_date} ~ {end_date}")
    
    try:
        df = provider.get_daily_basic_tushare(stock_code, start_date, end_date)
        
        if df.empty:
            logger.warning(f"未获取到日线基本面数据: {stock_code}")
            return df
        
        logger.info(f"成功下载日线基本面数据: {stock_code}, {len(df)} 条记录")
        return df
        
    except Exception as e:
        logger.error(f"下载日线基本面数据失败: {stock_code}, {e}")
        return pd.DataFrame()


def download_all_data_for_stock(
    provider: DataProvider,
    baostock_code: str,
    tushare_code: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    output_dir: str = "data/tushare"
) -> dict:
    """
    下载单个股票的所有Tushare数据
    
    Args:
        provider: DataProvider实例
        baostock_code: Baostock代码，如 sh.600000
        tushare_code: Tushare代码，如 600000.SH
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
        
    Returns:
        包含所有数据的字典
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"开始下载股票数据: {baostock_code} ({tushare_code})")
    logger.info(f"时间范围: {start_date} ~ {end_date}")
    logger.info(f"{'='*70}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "baostock_code": baostock_code,
        "tushare_code": tushare_code,
        "start_date": start_date,
        "end_date": end_date,
        "margin": None,
        "fina_indicator": None,
        "daily_basic": None
    }
    
    # 下载融资融券数据
    margin_df = download_margin_data(provider, tushare_code, start_date, end_date)
    if not margin_df.empty:
        output_path = os.path.join(output_dir, f"margin_{baostock_code.replace('.', '_')}_{start_date}_{end_date}.parquet")
        margin_df.to_parquet(output_path, index=False)
        results["margin"] = output_path
        logger.info(f"✓ 融资融券数据已保存: {output_path}")
    
    # 下载财务指标数据
    fina_df = download_fina_indicator(provider, tushare_code, start_date, end_date)
    if not fina_df.empty:
        output_path = os.path.join(output_dir, f"fina_indicator_{baostock_code.replace('.', '_')}_{start_date}_{end_date}.parquet")
        fina_df.to_parquet(output_path, index=False)
        results["fina_indicator"] = output_path
        logger.info(f"✓ 财务指标数据已保存: {output_path}")
    
    # 下载日线基本面数据
    basic_df = download_daily_basic(provider, tushare_code, start_date, end_date)
    if not basic_df.empty:
        output_path = os.path.join(output_dir, f"daily_basic_{baostock_code.replace('.', '_')}_{start_date}_{end_date}.parquet")
        basic_df.to_parquet(output_path, index=False)
        results["daily_basic"] = output_path
        logger.info(f"✓ 日线基本面数据已保存: {output_path}")
    
    return results


def download_moneyflow_data(
    provider: DataProvider,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    output_dir: str = "data/tushare"
) -> str:
    """
    下载北向资金数据并保存
    
    Args:
        provider: DataProvider实例
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
        
    Returns:
        保存的文件路径
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"开始下载北向资金数据")
    logger.info(f"时间范围: {start_date} ~ {end_date}")
    logger.info(f"{'='*70}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载北向资金数据
    moneyflow_df = download_moneyflow_hsgt(provider, start_date, end_date)
    
    if moneyflow_df.empty:
        logger.warning("未获取到北向资金数据")
        return ""
    
    # 保存数据
    output_path = os.path.join(output_dir, f"moneyflow_hsgt_{start_date}_{end_date}.parquet")
    moneyflow_df.to_parquet(output_path, index=False)
    
    logger.info(f"✓ 北向资金数据已保存: {output_path}")
    logger.info(f"  数据条数: {len(moneyflow_df)}")
    logger.info(f"  时间范围: {moneyflow_df['trade_date'].min()} ~ {moneyflow_df['trade_date'].max()}")
    
    return output_path


def main():
    """主函数"""
    # 配置
    TUSHARE_TOKEN = "2bcc0e9feb650d9862330a9743e5cc2e6469433c4d1ea0ce2d79371e"  # 用户提供
    
    START_DATE = "2020-01-01"
    END_DATE = "2025-12-31"
    OUTPUT_DIR = "/Users/dongxg/SourceCode/deep_final_kp/data/tushare"
    
    logger.info("="*70)
    logger.info("Tushare数据下载工具")
    logger.info("="*70)
    
    # 创建数据提供者
    logger.info("\n初始化数据提供者...")
    provider = DataProvider(tushare_token=TUSHARE_TOKEN)
    
    # 测试连接
    logger.info("测试Tushare连接...")
    if provider.ts_pro is None:
        logger.error("Tushare Pro初始化失败，请检查token是否正确")
        logger.error("尝试安装tushare: pip install tushare")
        return
    
    logger.info("✓ Tushare Pro连接成功")
    
    # 下载北向资金数据
    try:
        moneyflow_path = download_moneyflow_data(
            provider,
            start_date=START_DATE,
            end_date=END_DATE,
            output_dir=OUTPUT_DIR
        )
        
        if moneyflow_path:
            logger.info(f"\n✓ 北向资金数据下载完成")
            logger.info(f"  文件路径: {moneyflow_path}")
    except Exception as e:
        logger.error(f"下载北向资金数据时出错: {e}")
    
    # 下载示例股票的数据
    test_stocks = [
        {"baostock": "sh.600000", "tushare": "600000.SH", "name": "浦发银行"},
        {"baostock": "sh.600036", "tushare": "600036.SH", "name": "招商银行"},
        {"baostock": "sz.000001", "tushare": "000001.SZ", "name": "平安银行"},
    ]
    
    logger.info(f"\n下载示例股票数据...")
    for stock in test_stocks:
        try:
            results = download_all_data_for_stock(
                provider,
                baostock_code=stock["baostock"],
                tushare_code=stock["tushare"],
                start_date=START_DATE,
                end_date=END_DATE,
                output_dir=OUTPUT_DIR
            )
            
            logger.info(f"\n{stock['name']} ({stock['baostock']}) 下载结果:")
            logger.info(f"  融资融券: {'✓' if results['margin'] else '✗'}")
            logger.info(f"  财务指标: {'✓' if results['fina_indicator'] else '✗'}")
            logger.info(f"  日线基本面: {'✓' if results['daily_basic'] else '✗'}")
            
        except Exception as e:
            logger.error(f"下载{stock['name']}数据时出错: {e}")
    
    # 关闭连接
    provider.close()
    
    logger.info("\n" + "="*70)
    logger.info("数据下载完成！")
    logger.info(f"数据保存目录: {OUTPUT_DIR}")
    logger.info("="*70)


if __name__ == "__main__":
    main()