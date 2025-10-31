"""
外汇强化学习交易环境
基于Gymnasium标准接口实现的专业级交易环境
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForexTradingEnv(gym.Env):
    """
    外汇交易强化学习环境
    
    状态空间: 10维向量 (7个市场指标 + 3个持仓状态)
    动作空间: 4个离散动作 (Hold, Open Long, Open Short, Close)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        leverage: int = 20,
        trade_size_lots: float = 0.1,
        spread_cost_pips: float = 2.0,
        max_position_hold_steps: int = 480,
        max_drawdown_pct: float = 0.50
    ):
        """
        初始化交易环境
        
        Args:
            df: 包含所需技术指标的DataFrame
            initial_balance: 初始账户余额(美元)
            leverage: 杠杆倍数
            trade_size_lots: 交易手数(0.1表示0.1标准手)
            spread_cost_pips: 点差+手续费(点)
            max_position_hold_steps: 最长持仓步数
            max_drawdown_pct: 最大回撤百分比
        """
        super(ForexTradingEnv, self).__init__()
        
        # 账户参数
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.lot_size_standard = 100000  # 1标准手
        self.trade_size_lots = trade_size_lots
        
        # 交易成本
        self.spread_cost_pips = spread_cost_pips
        self.pip_value = 10 * trade_size_lots  # EURUSD: 每点价值 = 10美元/标准手 * 手数
        
        # 风险控制
        self.max_position_hold_steps = max_position_hold_steps
        self.max_drawdown_pct = max_drawdown_pct
        
        # 数据处理
        self.df = df.copy()
        self.max_steps = len(df) - 1
        self.warmup_steps = 100  # 预热步数
        
        # 验证数据完整性
        required_columns = [
            'close', 'M15_RSI', 'M15_ATR_norm', 'M15_MACD_hist_norm',
            'H1_RSI', 'H1_Trend_Indicator', 'H4_RSI', 'H4_Trend_Indicator'
        ]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame缺少必需列: {missing_cols}")
        
        # 定义状态空间 (10维) - 增加容差避免边界问题
        self.observation_space = spaces.Box(
            low=np.array([0, -10, -10, 0, -10, 0, -10, 0, -20, 0], dtype=np.float32),
            high=np.array([1, 10, 10, 1, 10, 1, 10, 2, 20, 1], dtype=np.float32),
            shape=(10,),
            dtype=np.float32
        )
        
        # 定义动作空间 (4个离散动作)
        # 0: Hold, 1: Open Long, 2: Open Short, 3: Close
        self.action_space = spaces.Discrete(4)
        
        # 初始化状态变量
        self._reset_state()
        
        logger.info(f"ForexTradingEnv初始化成功 - 数据长度: {len(df)}, 初始余额: ${initial_balance}")
    
    def _reset_state(self):
        """重置环境状态"""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_balance = self.initial_balance
        
        self.position_state = 0  # 0=空仓, 1=多单, 2=空单
        self.entry_price = 0.0
        self.position_pnl = 0.0
        self.steps_since_trade = 0
        
        self.current_step = self.warmup_steps
        self.total_trades = 0
        self.winning_trades = 0
    
    def reset(
        self,
        seed: int = None,
        options: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境到初始状态
        
        Returns:
            observation: 初始状态观测
            info: 额外信息字典
        """
        super().reset(seed=seed)
        self._reset_state()
        
        observation = self._get_observation()
        info = self._get_info()
        
        logger.info("环境已重置")
        return observation, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步交易动作
        
        Args:
            action: 动作编号 (0-3)
            
        Returns:
            observation: 新状态观测
            reward: 奖励值
            terminated: 是否因失败终止
            truncated: 是否因时间耗尽终止
            info: 额外信息
        """
        # 验证动作有效性
        if not self.action_space.contains(action):
            raise ValueError(f"无效动作: {action}")
        
        # 获取当前市场数据
        current_price = self.df.iloc[self.current_step]['close']
        previous_pnl = self.position_pnl
        
        # 执行动作
        action_cost = 0.0
        invalid_action = False
        
        if action == 1 and self.position_state == 0:  # 开多单
            self.position_state = 1
            self.entry_price = current_price
            self.steps_since_trade = 0
            action_cost = self.spread_cost_pips * self.pip_value
            self.total_trades += 1
            logger.debug(f"开多单 @ {current_price:.5f}")
            
        elif action == 2 and self.position_state == 0:  # 开空单
            self.position_state = 2
            self.entry_price = current_price
            self.steps_since_trade = 0
            action_cost = self.spread_cost_pips * self.pip_value
            self.total_trades += 1
            logger.debug(f"开空单 @ {current_price:.5f}")
            
        elif action == 3 and self.position_state != 0:  # 平仓
            # 计算最终盈亏
            if self.position_state == 1:  # 平多
                pips = (current_price - self.entry_price) * 10000
            else:  # 平空
                pips = (self.entry_price - current_price) * 10000
            
            realized_pnl = pips * self.pip_value - action_cost
            self.balance += realized_pnl
            self.equity = self.balance
            
            if realized_pnl > 0:
                self.winning_trades += 1
            
            logger.debug(f"平仓 @ {current_price:.5f}, 盈亏: ${realized_pnl:.2f}, 点数: {pips:.1f}")
            
            # 重置持仓
            self.position_state = 0
            self.entry_price = 0.0
            self.position_pnl = 0.0
            self.steps_since_trade = 0
            
        else:  # Hold或无效动作
            if action in [1, 2] and self.position_state != 0:
                invalid_action = True
                logger.debug(f"无效动作: 尝试在持仓时开仓")
            elif action == 3 and self.position_state == 0:
                invalid_action = True
                logger.debug(f"无效动作: 尝试在空仓时平仓")
        
        # 更新浮动盈亏(如果持仓中)
        if self.position_state != 0:
            if self.position_state == 1:  # 多单
                pips = (current_price - self.entry_price) * 10000
            else:  # 空单
                pips = (self.entry_price - current_price) * 10000
            
            self.position_pnl = pips * self.pip_value
            self.equity = self.balance + self.position_pnl
            self.steps_since_trade += 1
        
        # 计算奖励
        reward = self._calculate_reward(
            action, 
            action_cost, 
            invalid_action, 
            previous_pnl
        )
        
        # 检查终止条件
        terminated, truncated, reward = self._check_termination(reward)
        
        # 移动到下一步
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
            # 如果还有持仓,强制平仓
            if self.position_state != 0:
                self.balance += self.position_pnl
                self.position_state = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(
        self,
        action: int,
        action_cost: float,
        invalid_action: bool,
        previous_pnl: float
    ) -> float:
        """
        计算奖励值
        
        奖励设计原则:
        1. 基于盈亏变化的百分比
        2. 惩罚交易成本
        3. 惩罚无效动作
        4. 惩罚过长持仓
        5. 严重亏损额外惩罚
        """
        # 基础奖励: 盈亏变化百分比
        delta_pnl = self.position_pnl - previous_pnl
        base_reward = (delta_pnl / self.initial_balance) * 100
        
        # 交易成本惩罚
        if action in [1, 2]:
            cost_penalty = -(action_cost / self.initial_balance) * 100
            base_reward += cost_penalty
        
        # 无效动作惩罚
        if invalid_action:
            base_reward -= 0.1
        
        # 持仓时长惩罚(鼓励快速决策)
        if self.position_state != 0:
            time_penalty = -0.005 * (self.steps_since_trade / 100)
            base_reward += time_penalty
        
        # 风险调整(严重亏损额外惩罚)
        if self.position_state != 0:
            current_atr = self.df.iloc[self.current_step]['M15_ATR_norm']
            loss_threshold = -3 * abs(current_atr) * self.pip_value * 100
            if self.position_pnl < loss_threshold:
                base_reward -= 1.0
        
        return base_reward
    
    def _check_termination(
        self,
        reward: float
    ) -> Tuple[bool, bool, float]:
        """
        检查终止条件
        
        Returns:
            terminated: 是否因失败终止
            truncated: 是否因时间耗尽终止
            reward: 修改后的奖励(如果触发惩罚)
        """
        terminated = False
        truncated = False
        
        # 最大回撤检查
        self.peak_balance = max(self.peak_balance, self.equity)
        drawdown_pct = (self.peak_balance - self.equity) / self.peak_balance
        if drawdown_pct >= self.max_drawdown_pct:
            terminated = True
            reward = -50.0
            logger.warning(f"触发最大回撤止损: {drawdown_pct*100:.2f}%")
        
        # 强制平仓检查(持仓时间过长)
        if self.steps_since_trade >= self.max_position_hold_steps:
            if self.position_state != 0:
                self.balance += self.position_pnl
                self.equity = self.balance
                self.position_state = 0
                self.position_pnl = 0.0
                reward -= 5.0
                logger.warning(f"持仓时间过长,强制平仓")
        
        # 余额检查(亏损70%)
        if self.equity <= self.initial_balance * 0.3:
            terminated = True
            reward = -50.0
            logger.warning(f"账户余额过低: ${self.equity:.2f}")
        
        return terminated, truncated, reward
    
    def _get_observation(self) -> np.ndarray:
        """
        构建状态观测向量
        
        Returns:
            10维float32数组
        """
        try:
            row = self.df.iloc[self.current_step]
            
            # 市场技术指标 (7个特征)
            market_features = [
                float(row['M15_RSI']),
                float(row['M15_ATR_norm']),
                float(row['M15_MACD_hist_norm']),
                float(row['H1_RSI']),
                float(row['H1_Trend_Indicator']),
                float(row['H4_RSI']),
                float(row['H4_Trend_Indicator'])
            ]
            
            # 浮动盈亏归一化
            atr = abs(row['M15_ATR_norm'])
            if atr > 0:
                pnl_norm = self.position_pnl / (atr * self.pip_value * 100)
            else:
                pnl_norm = 0.0
            pnl_norm = np.clip(pnl_norm, -10, 10)
            
            # 持仓时间归一化
            time_norm = min(self.steps_since_trade / self.max_position_hold_steps, 1.0)
            
            # 动态交易状态 (3个特征)
            dynamic_features = [
                float(self.position_state),
                float(pnl_norm),
                float(time_norm)
            ]
            
            observation = np.array(
                market_features + dynamic_features,
                dtype=np.float32
            )
            
            # 静默裁剪观测空间（避免大量警告）
            observation = np.clip(
                observation,
                self.observation_space.low,
                self.observation_space.high
            )
            
            return observation
            
        except Exception as e:
            logger.error(f"构建观测失败: {e}")
            raise
    
    def _get_info(self) -> Dict[str, Any]:
        """
        返回额外信息
        
        Returns:
            包含账户状态的字典
        """
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        return {
            'balance': self.balance,
            'equity': self.equity,
            'position_state': self.position_state,
            'position_pnl': self.position_pnl,
            'current_step': self.current_step,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'peak_balance': self.peak_balance,
            'drawdown_pct': (self.peak_balance - self.equity) / self.peak_balance
        }
    
    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            info = self._get_info()
            print(f"\n{'='*60}")
            print(f"步数: {info['current_step']} | 持仓: {['空仓', '多单', '空单'][info['position_state']]}")
            print(f"余额: ${info['balance']:.2f} | 权益: ${info['equity']:.2f}")
            print(f"浮动盈亏: ${info['position_pnl']:.2f}")
            print(f"交易次数: {info['total_trades']} | 胜率: {info['win_rate']*100:.2f}%")
            print(f"峰值余额: ${info['peak_balance']:.2f} | 回撤: {info['drawdown_pct']*100:.2f}%")
            print(f"{'='*60}\n")
    
    def close(self):
        """清理资源"""
        pass
