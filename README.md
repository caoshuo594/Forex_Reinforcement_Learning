# 外汇强化学习交易系统

基于PPO算法和ONNX部署的专业级外汇交易强化学习系统。

## 项目结构

```
RL/
├── forex_rl_tutorial.ipynb  # 📓 完整教学Notebook（推荐从这里开始！）
├── forex_env.py             # 🏗️ 交易环境实现
├── get_mt5_data.py          # 📊 MT5数据获取脚本
├── train.py                 # 🎓 训练脚本
├── export_onnx.py           # 📦 ONNX导出脚本
├── ForexRLTrader.mq5        # 📈 MQL5 EA实现
├── ForexRLTrader.ex5        # 📈 MQL5 EA编译文件
├── requirements.txt         # 📄 Python依赖
├── README.md                # 📝 本文件
├── DEPLOYMENT_REPORT.md     # 📋 部署完整指南
├── CHECKLIST.md             # ✅ 启动检查清单
├── AI_PROMPT.md             # 🤖 AI编程提示词
├── models/                  # 📁 训练模型保存目录
│   ├── best_model/          #    最佳模型
│   ├── checkpoints/         #    训练检查点
│   ├── forex_policy.onnx    #    ONNX模型文件
│   └── forex_policy_spec.json    # 模型规格文档
└── logs/                    # 📁 训练日志和分析
    ├── training_analysis.png     # 训练分析图表
    ├── train_monitor.csv         # 训练监控日志
    └── eval_monitor.csv          # 评估监控日志
```

## 功能特点

### Python环境端

- ✅ 完整的Gymnasium标准交易环境
- ✅ 10维状态空间(7个市场指标 + 3个持仓状态)
- ✅ 4个离散动作(Hold, Open Long, Open Short, Close)
- ✅ 完善的奖励函数设计
- ✅ 风险控制机制(最大回撤、强制平仓)
- ✅ PPO算法训练支持
- ✅ ONNX模型导出和验证
- ✅ 完整的测试套件

### MQL5端

- ✅ ONNX模型加载和推理
- ✅ 技术指标实时计算
- ✅ 自动交易执行
- ✅ 持仓状态管理
- ✅ 完整的错误处理和日志

## 快速开始

### 🎓 方式一：使用Jupyter Notebook（推荐用于学习）

```bash
# 在VS Code中打开
code forex_rl_tutorial.ipynb

# 或在Jupyter中打开
jupyter notebook forex_rl_tutorial.ipynb
```

**优势**：
- 📚 完整的教学说明
- 🔬 可以逐步执行和实验
- 📊 包含可视化分析
- 💡 适合理解整个流程

---

### ⚡ 方式二：使用Python脚本（快速训练）

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 获取数据（可选，如已有EURUSD_processed.csv可跳过）

```bash
python get_mt5_data.py
```

#### 3. 训练模型

```bash
python train.py
```

训练参数可在`train.py`中修改:
- `TOTAL_TIMESTEPS`: 总训练步数(默认500,000)
- `initial_balance`: 初始账户余额(默认$10,000)
- `trade_size_lots`: 交易手数(默认0.1手)

#### 4. 导出ONNX模型

```bash
python export_onnx.py "models\best_model\best_model.zip" "models\forex_policy.onnx"
```

这将生成:
- `forex_policy.onnx` - ONNX模型文件
- `forex_policy_spec.json` - 模型规格书
- `forex_policy_test_cases.json` - 测试用例
- `feature_description.json` - 特征说明

#### 5. 部署到MT5

**方式A：自动部署**
ONNX模型已自动复制到MT5的Files目录

**方式B：手动部署**
1. 将`models/forex_policy.onnx`复制到`MT5安装目录/MQL5/Files/`
2. 将`ForexRLTrader.mq5`复制到`MT5安装目录/MQL5/Experts/`
3. 在MT5中编译并加载EA

**详细部署指南**：查看 `DEPLOYMENT_REPORT.md`  
**启动检查清单**：查看 `CHECKLIST.md`

---

## 📚 文档说明

| 文档 | 说明 |
|------|------|
| `README.md` | 项目总览和快速开始 |
| `DEPLOYMENT_REPORT.md` | 详细的部署指南和配置说明 |
| `CHECKLIST.md` | MT5启动前的检查清单 |
| `AI_PROMPT.md` | AI编程提示词和项目规范 |
| `forex_rl_tutorial.ipynb` | 完整的交互式教学Notebook |

---

## 环境设计

### 状态空间 (10维)

| 索引  | 特征名称                    | 范围        | 说明              |
| --- | ----------------------- | --------- | --------------- |
| 0   | M15_RSI                 | 0-1       | 15分钟RSI指标       |
| 1   | M15_ATR_norm            | -5到5      | 15分钟ATR标准化     |
| 2   | M15_MACD_hist_norm      | -5到5      | MACD柱状图标准化     |
| 3   | H1_RSI                  | 0-1       | 1小时RSI指标        |
| 4   | H1_Trend_Indicator      | -5到5      | 1小时趋势指标        |
| 5   | H4_RSI                  | 0-1       | 4小时RSI指标        |
| 6   | H4_Trend_Indicator      | -5到5      | 4小时趋势指标        |
| 7   | position_state          | 0, 1, 2   | 持仓状态(空仓/多/空)   |
| 8   | floating_pnl_normalized | -10到10    | 浮动盈亏归一化        |
| 9   | holding_time_normalized | 0-1       | 持仓时间归一化(最长480步) |

### 动作空间 (4个离散动作)

| 动作 | 名称         | 执行条件    |
| -- | ---------- | ------- |
| 0  | Hold       | 任何时候    |
| 1  | Open Long  | 仅空仓时    |
| 2  | Open Short | 仅空仓时    |
| 3  | Close      | 仅有持仓时 |

### 奖励函数

奖励设计基于以下原则:
1. **盈亏变化**: 主要奖励来源,归一化为账户余额百分比
2. **交易成本**: 开仓时扣除点差成本
3. **无效动作**: 惩罚无效动作(-0.1)
4. **持仓时长**: 鼓励快速决策,惩罚过长持仓
5. **风险控制**: 严重亏损额外惩罚

### 终止条件

- **最大回撤**: 触发50%最大回撤时终止(奖励-50)
- **余额检查**: 亏损70%时终止(奖励-50)
- **强制平仓**: 持仓超过480步(20天)强制平仓
- **数据耗尽**: 到达数据末尾时truncated

## 数据准备

### 训练数据格式

CSV文件需包含以下列:

```python
required_columns = [
    'close',                  # 收盘价
    'M15_RSI',               # 15分钟RSI (0-1)
    'M15_ATR_norm',          # ATR标准化值
    'M15_MACD_hist_norm',    # MACD柱状图标准化
    'H1_RSI',                # 1小时RSI
    'H1_Trend_Indicator',    # 1小时趋势指标
    'H4_RSI',                # 4小时RSI
    'H4_Trend_Indicator'     # 4小时趋势指标
]
```

### 数据获取示例(MT5)

```python
import MetaTrader5 as mt5
import pandas as pd

# 连接MT5
mt5.initialize()

# 获取历史数据
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0, 10000)
df = pd.DataFrame(rates)

# 计算技术指标...
# (参考train.py中的示例)
```

## 训练建议

### 超参数调优

```python
# PPO超参数
learning_rate = 3e-4      # 学习率
n_steps = 2048            # 每次更新的步数
batch_size = 64           # 批大小
n_epochs = 10             # 每次更新的轮数
gamma = 0.99              # 折扣因子
gae_lambda = 0.95         # GAE参数
```

### 训练监控

使用TensorBoard查看训练进度:

```bash
tensorboard --logdir=./logs
```

关键指标:
- `rollout/ep_rew_mean`: 平均回合奖励
- `train/policy_loss`: 策略网络损失
- `train/value_loss`: 价值网络损失

## ONNX集成说明

### 模型规格

- **输入**: `(1, 10)` float32数组
- **输出**: `(1, 4)` float32数组(动作概率)
- **Opset版本**: 11 (MQL5兼容)
- **动态轴**: 无(固定形状)

### MQL5推理流程

1. 准备10维观测向量
2. 创建输入矩阵 `matrix(1, 10)`
3. 调用 `OnnxRun()`
4. 获取输出矩阵 `matrix(1, 4)`
5. 选择最大概率对应的动作
6. 执行交易动作

### 验证一致性

使用生成的测试用例验证:

```python
# Python端
test_input = [...]  # 来自test_cases.json
output = session.run(None, {'observation': test_input})
print(output)  # 期望输出
```

```mql5
// MQL5端
double input[10] = {...};  // 相同输入
matrix output = RunInference(input);
Print(output);  // 应与Python输出一致(误差<0.001)
```

## 风险控制

### 环境层面

- 最大回撤限制: 50%
- 余额保护: 亏损70%终止
- 持仓时长限制: 最多480步(20天)
- 交易成本: 2点点差已计入

### EA层面

建议添加:
- 每日最大交易次数限制
- 时间段过滤(避开新闻时段)
- 波动率过滤(高波动时暂停)
- 资金管理(根据账户余额调整手数)

## 性能评估

### 评估指标

- **总收益率**: `(final_balance - initial_balance) / initial_balance`
- **最大回撤**: `(peak - valley) / peak`
- **夏普比率**: `mean(returns) / std(returns) * sqrt(252)`
- **胜率**: `winning_trades / total_trades`
- **平均盈亏比**: `avg_win / abs(avg_loss)`

### 回测示例

```python
from stable_baselines3 import PPO

# 加载模型
model = PPO.load("./models/ppo_forex_final")

# 在测试集上评估
evaluate_model(model, test_env, n_episodes=100)
```

## 常见问题

### Q: ONNX模型加载失败?
A: 确认文件在`MQL5/Files/`目录,检查文件名是否正确。

### Q: 推理输出全是0?
A: 检查观测值是否在合理范围内,确认技术指标计算正确。

### Q: 训练不收敛?
A: 尝试调整学习率、增加训练步数、检查奖励函数设计。

### Q: 实盘表现差?
A: 可能过拟合训练数据,增加训练数据多样性,使用更长时间跨度的数据。

## 注意事项

⚠️ **重要提示**:

1. **模拟测试**: 先在模拟账户充分测试
2. **数据质量**: 使用高质量、足够长度的历史数据训练
3. **市场环境**: 模型在不同市场环境下表现可能不同
4. **持续监控**: 定期监控EA性能,必要时重新训练
5. **风险管理**: 不要投入超过承受能力的资金

## 许可证

本项目仅供学习和研究使用。使用者需自行承担交易风险。

## 更新日志

### v1.0.0 (2024-10-30)
- 初始版本发布
- 完整的环境实现
- PPO训练支持
- ONNX导出和MQL5集成
- 完整的测试套件

## 联系方式

如有问题或建议,请提交Issue。

---

**免责声明**: 本软件仅用于教育和研究目的。交易有风险,投资需谨慎。使用本软件进行实盘交易的任何损失,作者不承担责任。
