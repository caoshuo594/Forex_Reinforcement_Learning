# MetaTrader 5 强化学习交易系统开发提示词

## 项目概述
为MetaTrader 5平台开发一个基于强化学习(Reinforcement Learning)的自动交易Expert Advisor(EA)，能够使用ONNX神经网络模型进行实时交易决策。

## 技术要求

### 1. 编程语言与框架
- **MQL5**: MetaTrader 5的原生编程语言，用于EA开发
- **Python**: 用于强化学习模型训练和数据处理
- **ONNX**: 用于神经网络模型的跨平台部署

### 2. 系统架构
项目包含以下核心组件：

#### 2.1 Python训练环境
- `forex_env.py`: Gymnasium环境实现，模拟外汇交易环境
  - 状态空间: 10维观测向量（技术指标+持仓状态）
  - 动作空间: 4个离散动作（Hold/Open Long/Open Short/Close）
  - 奖励函数: 基于盈亏和交易成本
  
- `export_onnx.py`: 将训练好的模型导出为ONNX格式
  - 输入: `observation` - shape [1, 10], dtype float32
  - 输出: `action_probs` - shape [1, 4], dtype float32
  
- `get_mt5_data.py`: 从MT5获取历史行情数据并预处理

#### 2.2 MQL5交易系统
`ForexRLTrader.mq5` - 主EA程序，包含：

**核心功能模块:**

1. **模型加载模块** (OnInit函数)
   - 支持多路径加载: 默认路径 → ONNX_COMMON_FOLDER → 绝对路径 → 缓冲区加载
   - 使用 `OnnxCreate()` API加载.onnx文件
   - 验证模型输入输出维度

2. **观测值准备模块** (PrepareObservation函数)
   - 计算技术指标:
     - RSI (相对强弱指数) - 3个时间周期
     - ATR (真实波动幅度) - 3个时间周期  
     - MACD (移动平均收敛散度) - 3个时间周期
     - Trend (趋势方向) - 3个时间周期
   - 添加持仓状态: 空仓(0) / 多单(1) / 空单(2)
   - 标准化所有特征到合理范围

3. **推理引擎模块** (RunInference函数)
   - **关键实现细节:**
     ```mql5
     vectorf input_vectorf;      // 使用vectorf而非matrixf
     input_vectorf.Resize(10);   // 10维输入向量
     
     vectorf output_vectorf;
     output_vectorf.Resize(4);   // 4维输出向量
     
     // 使用ONNX_NO_CONVERSION标志确保类型匹配
     OnnxRun(model_handle, ONNX_NO_CONVERSION, input_vectorf, output_vectorf);
     ```
   - 从输出概率中选择最优动作

4. **交易执行模块** (ExecuteAction函数)
   - 动作0: Hold - 保持当前状态
   - 动作1: Open Long - 仅在空仓时开多单
   - 动作2: Open Short - 仅在空仓时开空单
   - 动作3: Close - 平掉当前持仓
   - 使用CTrade类执行实际交易

5. **主循环** (OnTick函数)
   - 每N根K线执行一次决策
   - 准备观测值 → 运行推理 → 执行动作

### 3. 关键技术难点与解决方案

#### 难点1: ONNX模型集成
**问题**: MQL5的ONNX API对数据类型和形状要求严格

**解决方案**:
- 1D张量必须使用`vectorf`类型（float），不能用`matrixf`
- 2D张量应使用`matrixf`类型
- 预分配输出容器: `output_vectorf.Resize(OUTPUT_SIZE)`
- 使用`ONNX_NO_CONVERSION`标志避免自动类型转换
- 让OnnxRun自动推断形状，不手动调用`OnnxSetInputShape/OnnxSetOutputShape`

#### 难点2: 模型路径处理
**问题**: MT5的不同运行模式（测试/优化/实盘）使用不同的文件路径

**解决方案**:
```mql5
// 方法1: 直接加载（推荐）
g_model_handle = OnnxCreate(MODEL_FILE, ONNX_DEFAULT);

// 方法2: ONNX_COMMON_FOLDER
g_model_handle = OnnxCreate(MODEL_FILE, ONNX_COMMON_FOLDER);

// 方法3: 绝对路径构建
string abs_path = TerminalInfoString(TERMINAL_DATA_PATH) + 
                  "\\MQL5\\Files\\" + MODEL_FILE;

// 方法4: 缓冲区加载
int file_handle = FileOpen(MODEL_FILE, FILE_READ|FILE_BIN);
uchar model_buffer[];
FileReadArray(file_handle, model_buffer);
g_model_handle = OnnxCreateFromBuffer(model_buffer, ONNX_DEFAULT);
```

#### 难点3: 数据类型转换
**问题**: MQL5使用double精度，ONNX模型使用float32

**解决方案**:
```mql5
// 观测值从double数组转换为vectorf
for(int i = 0; i < INPUT_SIZE; i++)
    input_vectorf[i] = (float)observation[i];

// 输出从vectorf转回double用于后续处理
double max_prob = (double)output_vectorf[0];
```

### 4. 项目文件结构
```
RL/
├── ForexRLTrader.mq5          # MQL5 EA主程序
├── forex_env.py               # Gymnasium交易环境
├── export_onnx.py             # 模型导出脚本
├── get_mt5_data.py            # 数据获取脚本
├── deploy_onnx_to_mt5.ps1     # 部署自动化脚本
├── requirements.txt           # Python依赖
├── README.md                  # 项目说明
├── feature_description.json   # 特征说明文档
├── models/
│   ├── forex_policy.onnx      # ONNX模型文件
│   ├── forex_policy_spec.json # 模型规格说明
│   └── forex_policy_test_cases.json # 测试用例
└── EURUSD_processed.csv       # 历史数据
```

### 5. 开发流程

#### 阶段1: 数据准备
```python
# 1. 获取MT5历史数据
python get_mt5_data.py

# 2. 数据预处理
# - 计算技术指标
# - 标准化特征
# - 保存为CSV格式
```

#### 阶段2: 模型训练
```python
# 1. 定义Gymnasium环境 (forex_env.py)
# 2. 使用Stable-Baselines3或类似库训练PPO/DQN模型
# 3. 导出为ONNX格式
python export_onnx.py
```

#### 阶段3: 模型部署
```powershell
# 复制ONNX模型到MT5目录
.\deploy_onnx_to_mt5.ps1
```

#### 阶段4: 回测与优化
```
1. 在MetaEditor中编译ForexRLTrader.mq5
2. 在MT5策略测试器中进行回测
3. 分析交易结果和日志
4. 调整参数或重新训练模型
```

### 6. 重要参数配置

#### EA参数
```mql5
input int      CheckInterval = 10;     // 每N根K线检查一次
input double   TradeLots = 0.1;        // 交易手数
input int      MaxSpread = 30;         // 最大点差
input string   EA_Comment = "RL_Trader"; // 订单注释

const int INPUT_SIZE = 10;   // 输入维度
const int OUTPUT_SIZE = 4;   // 输出维度（4个动作）
const string MODEL_FILE = "forex_policy.onnx";
```

#### 模型规格
```json
{
  "input": {
    "name": "observation",
    "shape": [1, 10],
    "dtype": "float32",
    "features": [
      "RSI_H1", "RSI_H4", "RSI_D1",
      "ATR_H1", "ATR_H4", "ATR_D1", 
      "MACD_H1", "MACD_H4", "MACD_D1",
      "Position_State"
    ]
  },
  "output": {
    "name": "action_probs",
    "shape": [1, 4],
    "dtype": "float32",
    "actions": ["Hold", "Open Long", "Open Short", "Close"]
  }
}
```

### 7. 调试与验证

#### 关键日志输出
```mql5
Print("模型加载成功, 句柄: ", g_model_handle);
Print("输入数: ", OnnxGetInputCount(g_model_handle));  // 应为1
Print("输出数: ", OnnxGetOutputCount(g_model_handle)); // 应为1
Print("推理结果 - 动作: ", action, ", Probs: [...]");
```

#### 常见错误处理
1. **错误5805 "parameter is empty"**
   - 原因: 使用了matrixf而非vectorf
   - 解决: 改用vectorf类型

2. **编译错误 "variable expected"**
   - 原因: OnnxCreateFromBuffer参数类型错误
   - 解决: 传入uchar数组引用而非string

3. **推理返回false**
   - 检查模型路径是否正确
   - 验证输入/输出维度匹配
   - 查看GetLastError()的错误代码

### 8. 性能优化建议

1. **减少推理频率**: 不要每个tick都推理，设置合理的CheckInterval
2. **移除调试日志**: 生产环境删除Print语句
3. **指标缓存**: 避免重复计算相同的技术指标
4. **模型优化**: 使用量化或剪枝减小模型大小

### 9. 风险管理

#### 必须实现的保护措施
```mql5
// 1. 止损/止盈
g_trade.Buy(TradeLots, _Symbol, price, 
            price - StopLoss*_Point,    // 止损
            price + TakeProfit*_Point,  // 止盈
            EA_Comment);

// 2. 最大回撤保护
if(AccountInfoDouble(ACCOUNT_EQUITY) < InitialEquity * 0.8)
    ExpertRemove();  // 回撤超过20%停止交易

// 3. 持仓时间限制
if(TimeCurrent() - PositionGetInteger(POSITION_TIME) > MaxHoldTime)
    CloseAllPositions();

// 4. 点差过滤
double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
if(spread > MaxSpread)
    return;  // 点差过大不交易
```

### 10. 部署清单

部署前检查项:
- [ ] ONNX模型文件已复制到 `MQL5/Files/` 目录
- [ ] EA已在MetaEditor中成功编译
- [ ] 在策略测试器中完成充分回测
- [ ] 风险管理参数已设置（止损、止盈、最大手数）
- [ ] 已测试极端市场情况（高波动、低流动性）
- [ ] 所有调试日志已移除或注释
- [ ] 实盘前先在模拟账户测试至少一周

## 常见问题FAQ

**Q: 为什么推理结果总是Hold动作?**
A: 这通常是模型过于保守。检查训练时的奖励函数设计，可能需要增加开仓奖励或减少持仓惩罚。

**Q: 如何处理不同货币对?**
A: 需要针对不同货币对重新训练模型，因为各货币对的波动特性不同。或者在特征中加入货币对编码。

**Q: 可以用于实盘交易吗?**
A: 建议先在模拟账户长期测试。强化学习模型对市场环境变化敏感，需要定期重新训练。

**Q: 如何提高模型收益?**
A: 
1. 增加更多有效特征（如订单流、情绪指标）
2. 使用更长的训练数据
3. 调整奖励函数设计
4. 尝试不同的RL算法（PPO/SAC/TD3）

## 参考资源
- [MQL5 ONNX文档](https://www.mql5.com/en/docs/integration/onnx)
- [Gymnasium文档](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [ONNX官方文档](https://onnx.ai/onnx/)

---

## 致AI助手的指导说明

当用户基于此提示词请求开发类似系统时，请注意：

1. **优先使用vectorf**: 对于1D张量输入/输出，始终使用vectorf类型，不要用matrixf
2. **多路径加载**: 实现至少3种模型加载方式以适应不同运行环境
3. **类型匹配**: 确保ONNX模型的float32与MQL5的float类型对应，使用ONNX_NO_CONVERSION
4. **错误处理**: 每个关键步骤都要检查返回值和GetLastError()
5. **模块化设计**: 将观测准备、推理、动作执行分离为独立函数
6. **渐进式开发**: 先确保模型加载成功，再实现推理，最后连接交易逻辑
7. **充分测试**: 提供测试用例和验证步骤

此系统已在MT5策略测试器中验证通过，可作为生产级参考实现。
