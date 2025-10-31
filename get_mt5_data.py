"""
从MT5获取真实历史数据并计算技术指标
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_mt5():
    """初始化MT5连接"""
    if not mt5.initialize():
        logger.error("MT5初始化失败")
        return False
    
    logger.info(f"MT5版本: {mt5.version()}")
    logger.info(f"账户信息: {mt5.account_info()}")
    return True


def get_historical_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, bars=50000):
    """
    获取历史数据
    
    Args:
        symbol: 交易品种
        timeframe: 时间周期
        bars: K线数量
    """
    logger.info(f"获取 {symbol} {timeframe} 数据, {bars} 根K线...")
    
    # 获取数据
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    
    if rates is None:
        logger.error(f"获取数据失败: {mt5.last_error()}")
        return None
    
    # 转换为DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    logger.info(f"获取成功: {len(df)} 根K线")
    logger.info(f"时间范围: {df['time'].min()} 到 {df['time'].max()}")
    
    return df


def calculate_rsi(data, period=14):
    """计算RSI指标"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi / 100.0  # 归一化到0-1


def calculate_atr_norm(data, period=14, lookback=100):
    """计算ATR标准化值"""
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # 标准化
    atr_mean = atr.rolling(window=lookback).mean()
    atr_std = atr.rolling(window=lookback).std()
    atr_norm = (atr - atr_mean) / (atr_std + 1e-8)
    
    return atr_norm


def calculate_macd(data, fast=12, slow=26, signal=9):
    """计算MACD"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return histogram


def calculate_macd_hist_norm(data, lookback=100):
    """计算MACD柱状图标准化值"""
    hist = calculate_macd(data['close'])
    hist_mean = hist.rolling(window=lookback).mean()
    hist_std = hist.rolling(window=lookback).std()
    hist_norm = (hist - hist_mean) / (hist_std + 1e-8)
    return hist_norm


def calculate_ma(data, period):
    """计算移动平均"""
    return data.rolling(window=period).mean()


def calculate_trend_indicator(data, ma_period=50, lookback=100):
    """计算趋势指标"""
    ma = calculate_ma(data['close'], ma_period)
    distance = (data['close'] - ma) / (ma + 1e-8) * 100
    
    # 标准化
    distance_mean = distance.rolling(window=lookback).mean()
    distance_std = distance.rolling(window=lookback).std()
    trend_norm = (distance - distance_mean) / (distance_std + 1e-8)
    
    return trend_norm


def get_m15_data(symbol="EURUSD", bars=50000):
    """获取M15数据"""
    return get_historical_data(symbol, mt5.TIMEFRAME_M15, bars)


def get_h1_data(symbol="EURUSD", bars=50000):
    """获取H1数据"""
    return get_historical_data(symbol, mt5.TIMEFRAME_H1, bars)


def get_h4_data(symbol="EURUSD", bars=15000):
    """获取H4数据"""
    return get_historical_data(symbol, mt5.TIMEFRAME_H4, bars)


def align_timeframes(m15_df, h1_df, h4_df):
    """
    对齐不同周期的数据
    将H1和H4的指标映射到M15时间框架
    """
    logger.info("对齐多时间框架数据...")
    
    # 确保时间列是datetime类型
    m15_df['time'] = pd.to_datetime(m15_df['time'])
    h1_df['time'] = pd.to_datetime(h1_df['time'])
    h4_df['time'] = pd.to_datetime(h4_df['time'])
    
    # 使用merge_asof进行时间对齐
    # H1数据对齐到M15
    df = pd.merge_asof(
        m15_df.sort_values('time'),
        h1_df[['time', 'H1_RSI', 'H1_Trend_Indicator']].sort_values('time'),
        on='time',
        direction='backward'
    )
    
    # H4数据对齐到M15
    df = pd.merge_asof(
        df.sort_values('time'),
        h4_df[['time', 'H4_RSI', 'H4_Trend_Indicator']].sort_values('time'),
        on='time',
        direction='backward'
    )
    
    return df


def prepare_training_data(symbol="EURUSD"):
    """
    准备完整的训练数据
    
    Returns:
        DataFrame包含所有需要的特征
    """
    # 初始化MT5
    if not initialize_mt5():
        return None
    
    try:
        # 获取M15数据 (主要周期)
        logger.info("="*60)
        logger.info("获取M15数据...")
        m15_df = get_m15_data(symbol, bars=50000)
        if m15_df is None:
            return None
        
        # 计算M15指标
        logger.info("计算M15技术指标...")
        m15_df['M15_RSI'] = calculate_rsi(m15_df['close'], 14)
        m15_df['M15_ATR_norm'] = calculate_atr_norm(m15_df, 14, 100)
        m15_df['M15_MACD_hist_norm'] = calculate_macd_hist_norm(m15_df, 100)
        
        # 获取H1数据
        logger.info("="*60)
        logger.info("获取H1数据...")
        h1_df = get_h1_data(symbol, bars=15000)
        if h1_df is None:
            return None
        
        # 计算H1指标
        logger.info("计算H1技术指标...")
        h1_df['H1_RSI'] = calculate_rsi(h1_df['close'], 14)
        h1_df['H1_Trend_Indicator'] = calculate_trend_indicator(h1_df, 50, 100)
        
        # 获取H4数据
        logger.info("="*60)
        logger.info("获取H4数据...")
        h4_df = get_h4_data(symbol, bars=5000)
        if h4_df is None:
            return None
        
        # 计算H4指标
        logger.info("计算H4技术指标...")
        h4_df['H4_RSI'] = calculate_rsi(h4_df['close'], 14)
        h4_df['H4_Trend_Indicator'] = calculate_trend_indicator(h4_df, 50, 100)
        
        # 对齐时间框架
        logger.info("="*60)
        df = align_timeframes(m15_df, h1_df, h4_df)
        
        # 删除NaN值
        logger.info("清理数据...")
        df = df.dropna()
        
        # 裁剪异常值
        logger.info("裁剪异常值...")
        df['M15_ATR_norm'] = df['M15_ATR_norm'].clip(-5, 5)
        df['M15_MACD_hist_norm'] = df['M15_MACD_hist_norm'].clip(-5, 5)
        df['H1_Trend_Indicator'] = df['H1_Trend_Indicator'].clip(-5, 5)
        df['H4_Trend_Indicator'] = df['H4_Trend_Indicator'].clip(-5, 5)
        
        # 保存数据
        output_file = f"{symbol}_processed.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"✓ 数据已保存: {output_file}")
        
        # 数据统计
        logger.info("="*60)
        logger.info("数据统计:")
        logger.info(f"总行数: {len(df)}")
        logger.info(f"时间范围: {df['time'].min()} 到 {df['time'].max()}")
        logger.info(f"数据列: {list(df.columns)}")
        logger.info("="*60)
        
        # 显示前几行
        logger.info("\n数据预览:")
        logger.info(df[['time', 'close', 'M15_RSI', 'M15_ATR_norm', 
                        'H1_RSI', 'H4_RSI']].head(10))
        
        return df
        
    finally:
        mt5.shutdown()
        logger.info("MT5连接已关闭")


if __name__ == "__main__":
    logger.info("开始获取MT5真实历史数据...")
    df = prepare_training_data("EURUSD")
    
    if df is not None:
        logger.info("\n✅ 数据准备完成!")
        logger.info("可以运行 train.py 开始训练")
    else:
        logger.error("\n❌ 数据获取失败")
