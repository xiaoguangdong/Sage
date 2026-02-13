"""
数据加载器
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from utils.column_normalizer import normalize_security_columns

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir: str):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # 确保目录存在
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_baostock_data(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        加载Baostock数据
        
        Args:
            stock_code: 股票代码（如 'sh.600000'）
            
        Returns:
            DataFrame或None（如果文件不存在）
        """
        file_path = self.raw_dir / f"{stock_code}.parquet"
        
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return None
            
        try:
            df = pd.read_parquet(file_path)
            df['stock'] = stock_code
            df = normalize_security_columns(df, inplace=False)
            logger.info(f"成功加载Baostock数据: {stock_code}, 行数: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"加载Baostock数据失败: {stock_code}, 错误: {e}")
            return None
    
    def load_all_baostock_data(self, stock_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        加载所有Baostock数据
        
        Args:
            stock_codes: 股票代码列表（可选；为空则加载raw目录全部parquet）
            
        Returns:
            合并后的DataFrame
        """
        if stock_codes is None:
            stock_files = list(self.raw_dir.glob("*.parquet"))
            stock_codes = [f.stem for f in stock_files]
            if not stock_codes:
                logger.warning(f"raw目录下未找到parquet文件: {self.raw_dir}")
                return pd.DataFrame()
        
        dfs = []
        
        for code in stock_codes:
            df = self.load_baostock_data(code)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            logger.warning("没有加载到任何Baostock数据")
            return pd.DataFrame()
            
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"成功加载 {len(dfs)} 只股票的数据，总行数: {len(combined_df)}")
        
        return combined_df
    
    def save_processed_data(self, df: pd.DataFrame, name: str):
        """
        保存处理后的数据
        
        Args:
            df: 要保存的DataFrame
            name: 文件名
        """
        file_path = self.processed_dir / f"{name}.parquet"
        
        try:
            df.to_parquet(file_path)
            logger.info(f"成功保存处理后的数据: {name}, 行数: {len(df)}")
        except Exception as e:
            logger.error(f"保存处理后的数据失败: {name}, 错误: {e}")
            raise
    
    def load_processed_data(self, name: str) -> Optional[pd.DataFrame]:
        """
        加载处理后的数据
        
        Args:
            name: 文件名
            
        Returns:
            DataFrame或None（如果文件不存在）
        """
        file_path = self.processed_dir / f"{name}.parquet"
        
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return None
            
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"成功加载处理后的数据: {name}, 行数: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"加载处理后的数据失败: {name}, 错误: {e}")
            return None
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        检查数据质量
        
        Args:
            df: 要检查的DataFrame
            
        Returns:
            数据质量报告字典
        """
        report = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'date_range': None,
            'stock_codes': df['stock'].unique().tolist() if 'stock' in df.columns else [],
        }
        
        # 检查日期范围
        date_col = 'trade_date' if 'trade_date' in df.columns else 'date' if 'date' in df.columns else None
        if date_col:
            dates = pd.to_datetime(df[date_col], errors='coerce')
            report['date_range'] = {
                'start': dates.min(),
                'end': dates.max(),
                'days': (dates.max() - dates.min()).days if dates.notna().any() else None
            }
        
        # 检查数值型数据的统计信息
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        logger.info(f"数据质量检查完成: 总行数={report['total_rows']}")
        
        return report


if __name__ == "__main__":
    # 测试数据加载器
    logging.basicConfig(level=logging.INFO)
    
    loader = DataLoader("/Users/dongxg/SourceCode/deep_final_kp/data")
    
    # 获取Baostock数据目录中的股票代码
    baostock_dir = Path("/Users/dongxg/SourceCode/deep_final_kp/data/baostock")
    stock_files = list(baostock_dir.glob("*.parquet"))
    stock_codes = [f.stem for f in stock_files[:10]]  # 测试前10只股票
    
    print(f"测试加载 {len(stock_codes)} 只股票...")
    df = loader.load_all_baostock_data(stock_codes)
    
    if not df.empty:
        print(f"\n数据预览:")
        print(df.head())
        
        print(f"\n数据质量报告:")
        report = loader.check_data_quality(df)
        for key, value in report.items():
            print(f"{key}: {value}")
