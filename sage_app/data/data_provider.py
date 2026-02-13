"""
数据提供商接口管理器
统一管理多个数据源的接口调用
"""
import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import baostock as bs

logger = logging.getLogger(__name__)


class DataProvider:
    """统一数据接口管理器"""
    
    def __init__(self, tushare_token: str = None):
        """
        初始化数据提供商
        
        Args:
            tushare_token: Tushare Pro API token
        """
        self.tushare_token = tushare_token
        self.ts_pro = None
        self.bs = None
        
        # 初始化Baostock
        self._init_baostock()
        
        # 初始化Tushare（如果提供了token）
        if tushare_token:
            self._init_tushare()
    
    def _init_baostock(self):
        """初始化Baostock"""
        try:
            lg = bs.login()
            if lg.error_code != '0':
                logger.error(f"Baostock登录失败: {lg.error_msg}")
                self.bs = None
            else:
                logger.info("Baostock登录成功")
                self.bs = bs
        except Exception as e:
            logger.error(f"Baostock初始化异常: {e}")
            self.bs = None
    
    def _init_tushare(self):
        """初始化Tushare Pro"""
        try:
            import tushare as ts
            self.ts_pro = ts.pro_api(self.tushare_token)
            logger.info("Tushare Pro初始化成功")
        except ImportError:
            logger.warning("tushare未安装，请运行: pip install tushare")
            self.ts_pro = None
        except Exception as e:
            logger.error(f"Tushare Pro初始化异常: {e}")
            self.ts_pro = None
    
    # ==================== Baostock接口 ====================
    
    def get_stock_daily_baostock(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取股票日线数据（Baostock）
        
        Args:
            code: 股票代码，如 sh.600000
            start_date: 开始日期，格式 2020-01-01
            end_date: 结束日期，格式 2025-12-31
            
        Returns:
            DataFrame包含OHLCV数据
        """
        if not self.bs:
            logger.error("Baostock未初始化")
            return pd.DataFrame()
        
        try:
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"  # 后复权
            )
            
            if rs.error_code != '0':
                logger.error(f"Baostock查询失败: {rs.error_msg}")
                return pd.DataFrame()
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 转换数据类型
            numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 
                          'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"获取Baostock数据异常: {e}")
            return pd.DataFrame()
    
    def get_index_daily_baostock(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数日线数据（Baostock）
        
        Args:
            code: 指数代码，如 sh.000300
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame包含OHLCV数据
        """
        if not self.bs:
            logger.error("Baostock未初始化")
            return pd.DataFrame()
        
        try:
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3"  # 不复权
            )
            
            if rs.error_code != '0':
                logger.error(f"Baostock指数查询失败: {rs.error_msg}")
                return pd.DataFrame()
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 转换数据类型
            numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 
                          'turn', 'pctChg']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"获取Baostock指数数据异常: {e}")
            return pd.DataFrame()
    
    def get_all_stocks_baostock(self) -> pd.DataFrame:
        """
        获取所有股票列表（Baostock）
        
        Returns:
            DataFrame包含股票列表
        """
        if not self.bs:
            logger.error("Baostock未初始化")
            return pd.DataFrame()
        
        try:
            rs = bs.query_all_stock(day=datetime.now().strftime("%Y-%m-%d"))
            
            if rs.error_code != '0':
                logger.error(f"Baostock股票列表查询失败: {rs.error_msg}")
                return pd.DataFrame()
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            return df
            
        except Exception as e:
            logger.error(f"获取股票列表异常: {e}")
            return pd.DataFrame()
    
    # ==================== Tushare Pro接口 ====================
    
    def get_fina_indicator_tushare(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取财务指标数据（Tushare Pro）
        
        Args:
            ts_code: 股票代码，如 600000.SH
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame包含财务指标
        """
        if not self.ts_pro:
            logger.warning("Tushare Pro未初始化，无法获取财务数据")
            return pd.DataFrame()
        
        try:
            df = self.ts_pro.fina_indicator(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                logger.info(f"获取{ts_code}财务指标成功: {len(df)}条记录")
            
            return df if df is not None else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取Tushare财务指标异常: {e}")
            return pd.DataFrame()
    
    def get_daily_basic_tushare(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取每日基本面指标（Tushare Pro）
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame包含PE、PB等估值指标
        """
        if not self.ts_pro:
            logger.warning("Tushare Pro未初始化，无法获取估值数据")
            return pd.DataFrame()
        
        try:
            df = self.ts_pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                logger.info(f"获取{ts_code}估值数据成功: {len(df)}条记录")
            
            return df if df is not None else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取Tushare估值数据异常: {e}")
            return pd.DataFrame()
    
    def get_moneyflow_hsgt_tushare(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取沪深港通资金流向（Tushare Pro）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame包含北向资金数据
        """
        if not self.ts_pro:
            logger.warning("Tushare Pro未初始化，无法获取北向资金数据")
            return pd.DataFrame()
        
        try:
            df = self.ts_pro.moneyflow_hsgt(
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                logger.info(f"获取北向资金数据成功: {len(df)}条记录")
            
            return df if df is not None else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取北向资金数据异常: {e}")
            return pd.DataFrame()
    
    def get_margin_tushare(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取融资融券数据（Tushare Pro）
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame包含融资融券数据
        """
        if not self.ts_pro:
            logger.warning("Tushare Pro未初始化，无法获取融资融券数据")
            return pd.DataFrame()
        
        try:
            df = self.ts_pro.margin(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                logger.info(f"获取{ts_code}融资融券数据成功: {len(df)}条记录")
            
            return df if df is not None else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取融资融券数据异常: {e}")
            return pd.DataFrame()
    
    # ==================== 新浪/腾讯接口 ====================
    
    def get_realtime_quote_sina(self, codes: List[str]) -> Dict:
        """
        获取实时行情（新浪）
        
        Args:
            codes: 股票代码列表，如 ['sh000300', 'sz000001']
            
        Returns:
            包含实时行情的字典
        """
        try:
            # 格式化代码
            code_str = ",".join(codes)
            url = f"https://qt.gtimg.cn/q={code_str}"
            
            response = requests.get(url, timeout=5)
            response.encoding = 'gbk'
            
            if response.status_code == 200:
                content = response.text
                # 解析返回的数据
                result = {}
                items = content.strip().split('\n')
                for item in items:
                    if '=' in item and not item.startswith('var'):
                        code, data = item.split('=', 1)
                        data = data.strip('"')
                        if data:
                            result[code] = data
                
                return result
            
            return {}
            
        except Exception as e:
            logger.error(f"获取新浪实时行情异常: {e}")
            return {}
    
    # ==================== 通用方法 ====================
    
    def calculate_market_breadth(self, date: str, stock_list: List[str] = None) -> Dict:
        """
        计算市场广度指标
        
        Args:
            date: 日期，格式 2025-12-30
            stock_list: 股票列表，如果为None则获取所有股票
            
        Returns:
            包含市场广度指标的字典
        """
        if stock_list is None:
            # 获取所有股票列表
            stocks_df = self.get_all_stocks_baostock()
            if stocks_df.empty:
                return {}
            stock_list = stocks_df['code'].tolist()
        
        # 统计涨跌家数
        up_count = 0
        down_count = 0
        flat_count = 0
        total_volume = 0
        up_volume = 0
        down_volume = 0
        
        for code in stock_list[:100]:  # 限制数量避免超时
            try:
                df = self.get_stock_daily_baostock(code, date, date)
                if df.empty:
                    continue
                
                pct_chg = df['pctChg'].iloc[0]
                volume = df['volume'].iloc[0]
                
                total_volume += volume
                
                if pct_chg > 0:
                    up_count += 1
                    up_volume += volume
                elif pct_chg < 0:
                    down_count += 1
                    down_volume += volume
                else:
                    flat_count += 1
                    
            except Exception as e:
                logger.warning(f"获取{code}数据失败: {e}")
                continue
        
        total = up_count + down_count + flat_count
        
        return {
            "date": date,
            "up_count": up_count,
            "down_count": down_count,
            "flat_count": flat_count,
            "total": total,
            "up_ratio": up_count / total if total > 0 else 0,
            "down_ratio": down_count / total if total > 0 else 0,
            "volume_ratio_up": up_volume / total_volume if total_volume > 0 else 0,
            "volume_ratio_down": down_volume / total_volume if total_volume > 0 else 0
        }
    
    def close(self):
        """关闭连接"""
        if self.bs:
            try:
                bs.logout()
                logger.info("Baostock已登出")
            except:
                pass


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("="*70)
    print("数据提供商测试")
    print("="*70)
    
    # 创建数据提供商（不提供Tushare token）
    provider = DataProvider()
    
    # 测试Baostock
    print("\n测试Baostock接口...")
    
    # 获取股票日线数据
    print("\n1. 获取股票日线数据 (sh.600000)...")
    df_stock = provider.get_stock_daily_baostock(
        "sh.600000",
        "2025-12-01",
        "2025-12-31"
    )
    print(f"获取到 {len(df_stock)} 条数据")
    if not df_stock.empty:
        print(df_stock.head())
    
    # 获取指数日线数据
    print("\n2. 获取指数日线数据 (sh.000300)...")
    df_index = provider.get_index_daily_baostock(
        "sh.000300",
        "2025-12-01",
        "2025-12-31"
    )
    print(f"获取到 {len(df_index)} 条数据")
    if not df_index.empty:
        print(df_index.head())
    
    # 获取股票列表
    print("\n3. 获取股票列表...")
    df_stocks = provider.get_all_stocks_baostock()
    print(f"获取到 {len(df_stocks)} 只股票")
    print(df_stocks.head())
    
    # 计算市场广度
    print("\n4. 计算市场广度 (2025-12-30)...")
    breadth = provider.calculate_market_breadth("2025-12-30")
    print(f"上涨: {breadth.get('up_count', 0)}")
    print(f"下跌: {breadth.get('down_count', 0)}")
    print(f"平盘: {breadth.get('flat_count', 0)}")
    print(f"上涨比例: {breadth.get('up_ratio', 0):.2%}")
    
    # 测试新浪实时行情
    print("\n5. 获取新浪实时行情...")
    realtime = provider.get_realtime_quote_sina(['sh000300', 'sz000001'])
    print(f"获取到 {len(realtime)} 只股票的实时行情")
    for code, data in realtime.items():
        print(f"{code}: {data[:50]}...")
    
    # 关闭连接
    provider.close()
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)