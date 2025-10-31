"""
外汇强化学习模型训练脚本
使用PPO算法训练交易策略
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import logging
from pathlib import Path
from datetime import datetime

from forex_env import ForexTradingEnv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# 训练配置参数
# ============================================

# 训练参数
TOTAL_TIMESTEPS = 500000  # 总训练步数
EVAL_FREQ = 10000  # 评估频率
SAVE_FREQ = 50000  # 模型保存频率
N_EVAL_EPISODES = 5  # 评估时运行的回合数

# 环境参数
INITIAL_BALANCE = 10000.0  # 初始账户余额(美元)
LEVERAGE = 20  # 杠杆倍数
TRADE_SIZE_LOTS = 0.1  # 交易手数
SPREAD_COST_PIPS = 2.0  # 点差+手续费(点)
MAX_POSITION_HOLD_STEPS = 480  # 最长持仓步数(480步约=12小时,M15周期)
MAX_DRAWDOWN_PCT = 0.50  # 最大回撤百分比

# PPO超参数
LEARNING_RATE = 3e-4  # 学习率
N_STEPS = 2048  # 每次更新收集的步数
BATCH_SIZE = 64  # 小批量大小
N_EPOCHS = 10  # 每次更新的训练轮数
GAMMA = 0.99  # 折扣因子
GAE_LAMBDA = 0.95  # GAE参数
CLIP_RANGE = 0.2  # PPO裁剪范围
ENT_COEF = 0.01  # 熵系数(鼓励探索)
VF_COEF = 0.5  # 价值函数系数
MAX_GRAD_NORM = 0.5  # 梯度裁剪

# 网络架构
POLICY_NETWORK = [256, 256]  # 策略网络隐藏层

# 数据文件路径
DATA_FILE = "EURUSD_processed.csv"

# 输出路径
LOG_DIR = "./logs"
MODEL_DIR = "./models"
BEST_MODEL_PATH = "./models/best_model"
CHECKPOINT_DIR = "./models/checkpoints"


def load_data(file_path: str) -> pd.DataFrame:
    """
    加载并验证训练数据
    
    Args:
        file_path: CSV数据文件路径
        
    Returns:
        pd.DataFrame: 加载的数据
    """
    logger.info(f"加载数据: {file_path}")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"数据加载成功 - 行数: {len(df)}")
    
    # 验证必需的列
    required_columns = [
        'close', 'M15_RSI', 'M15_ATR_norm', 'M15_MACD_hist_norm',
        'H1_RSI', 'H1_Trend_Indicator', 'H4_RSI', 'H4_Trend_Indicator'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据文件缺少必需列: {missing_cols}")
    
    # 检查数据质量
    logger.info("数据列:")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        logger.info(f"  {col}: {null_count} 个空值")
    
    # 删除包含NaN的行
    df_clean = df.dropna()
    if len(df_clean) < len(df):
        logger.warning(f"删除了 {len(df) - len(df_clean)} 行包含空值的数据")
    
    return df_clean


def create_env(df: pd.DataFrame, monitor_file: str = None) -> ForexTradingEnv:
    """
    创建交易环境
    
    Args:
        df: 训练数据
        monitor_file: Monitor日志文件路径
        
    Returns:
        ForexTradingEnv: 交易环境
    """
    env = ForexTradingEnv(
        df=df,
        initial_balance=INITIAL_BALANCE,
        leverage=LEVERAGE,
        trade_size_lots=TRADE_SIZE_LOTS,
        spread_cost_pips=SPREAD_COST_PIPS,
        max_position_hold_steps=MAX_POSITION_HOLD_STEPS,
        max_drawdown_pct=MAX_DRAWDOWN_PCT
    )
    
    if monitor_file:
        env = Monitor(env, monitor_file)
    
    return env


def split_train_test(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    分割训练集和测试集
    
    Args:
        df: 完整数据集
        test_ratio: 测试集比例
        
    Returns:
        train_df, test_df: 训练和测试数据
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    logger.info(f"数据分割 - 训练: {len(train_df)} 行, 测试: {len(test_df)} 行")
    
    return train_df, test_df


def train_model():
    """
    训练PPO模型
    """
    logger.info("=" * 60)
    logger.info("外汇强化学习交易系统 - 模型训练")
    logger.info("=" * 60)
    
    # 创建输出目录
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    df = load_data(DATA_FILE)
    
    # 分割训练集和测试集
    train_df, test_df = split_train_test(df, test_ratio=0.2)
    
    # 创建训练环境
    logger.info("创建训练环境...")
    train_env = create_env(
        train_df,
        monitor_file=f"{LOG_DIR}/train_monitor.csv"
    )
    train_env = DummyVecEnv([lambda: train_env])
    
    # 创建评估环境
    logger.info("创建评估环境...")
    eval_env = create_env(
        test_df,
        monitor_file=f"{LOG_DIR}/eval_monitor.csv"
    )
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # 配置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 创建PPO模型
    logger.info("创建PPO模型...")
    logger.info(f"超参数:")
    logger.info(f"  学习率: {LEARNING_RATE}")
    logger.info(f"  批大小: {BATCH_SIZE}")
    logger.info(f"  训练轮数: {N_EPOCHS}")
    logger.info(f"  收集步数: {N_STEPS}")
    logger.info(f"  折扣因子: {GAMMA}")
    logger.info(f"  网络架构: {POLICY_NETWORK}")
    
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=dict(
            net_arch=dict(pi=POLICY_NETWORK, vf=POLICY_NETWORK)
        ),
        tensorboard_log=LOG_DIR,
        device=device,
        verbose=1
    )
    
    # 设置回调函数
    logger.info("配置回调函数...")
    
    # 评估回调 - 保存最佳模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_PATH,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # 检查点回调 - 定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_forex",
        verbose=1
    )
    
    # 开始训练
    logger.info("=" * 60)
    logger.info(f"开始训练 - 总步数: {TOTAL_TIMESTEPS:,}")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # 保存最终模型
        final_model_path = f"{MODEL_DIR}/ppo_forex_final.zip"
        model.save(final_model_path)
        logger.info(f"最终模型已保存: {final_model_path}")
        
    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
        interrupted_model_path = f"{MODEL_DIR}/ppo_forex_interrupted.zip"
        model.save(interrupted_model_path)
        logger.info(f"中断模型已保存: {interrupted_model_path}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"训练时长: {duration}")
    logger.info(f"最佳模型路径: {BEST_MODEL_PATH}")
    logger.info(f"TensorBoard日志: {LOG_DIR}")
    logger.info("=" * 60)
    
    # 显示训练统计
    logger.info("\n查看训练进度:")
    logger.info(f"  tensorboard --logdir={LOG_DIR}")
    logger.info("\n下一步:")
    logger.info(f"  python export_onnx.py")
    
    # 清理
    train_env.close()
    eval_env.close()


def evaluate_model(model_path: str, episodes: int = 10):
    """
    评估训练好的模型
    
    Args:
        model_path: 模型文件路径
        episodes: 评估回合数
    """
    logger.info(f"评估模型: {model_path}")
    
    # 加载数据
    df = load_data(DATA_FILE)
    _, test_df = split_train_test(df, test_ratio=0.2)
    
    # 创建环境
    env = create_env(test_df)
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 运行评估
    total_rewards = []
    total_profits = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        total_profits.append(info.get('profit_pct', 0))
        
        logger.info(f"回合 {episode + 1}: 奖励={episode_reward:.2f}, "
                   f"收益率={info.get('profit_pct', 0):.2f}%, "
                   f"交易次数={info.get('total_trades', 0)}")
    
    # 统计结果
    logger.info("\n" + "=" * 60)
    logger.info("评估结果:")
    logger.info(f"  平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    logger.info(f"  平均收益率: {np.mean(total_profits):.2f}% ± {np.std(total_profits):.2f}%")
    logger.info(f"  最佳收益率: {np.max(total_profits):.2f}%")
    logger.info(f"  最差收益率: {np.min(total_profits):.2f}%")
    logger.info("=" * 60)
    
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        # 评估模式
        model_path = sys.argv[2] if len(sys.argv) > 2 else f"{BEST_MODEL_PATH}/best_model.zip"
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        evaluate_model(model_path, episodes)
    else:
        # 训练模式
        train_model()
