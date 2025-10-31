//+------------------------------------------------------------------+
//|                                              ForexRLTrader.mq5   |
//|                                  外汇强化学习交易EA               |
//|                                  使用ONNX模型进行交易决策          |
//+------------------------------------------------------------------+
#property copyright "Forex RL Trader"
#property link      ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

// ============================================
// ONNX模型配置 (根据数据规格书填写)
// ============================================
#define MODEL_FILE "forex_policy.onnx"
#define INPUT_SIZE 10       // 观测空间维度
#define OUTPUT_SIZE 4       // 动作空间维度 (Hold, Long, Short, Close)

// ============================================
// 交易参数
// ============================================
input double TradeLots = 0.1;           // 交易手数
input int MaxSlippagePoints = 30;       // 最大滑点(点)
input string EA_Comment = "RL_Trader";  // EA标识

// ============================================
// 全局变量
// ============================================
long g_model_handle = INVALID_HANDLE;   // ONNX模型句柄
CTrade g_trade;                          // 交易对象
datetime g_last_bar_time = 0;           // 上一根K线时间

// 持仓状态追踪
int g_position_state = 0;               // 0=空仓, 1=多单, 2=空单
double g_entry_price = 0.0;             // 开仓价格
int g_steps_since_trade = 0;            // 持仓步数

//+------------------------------------------------------------------+
//| 初始化函数                                                         |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("========================================");
    Print("外汇强化学习交易EA - 初始化");
    Print("========================================");
    
    // 重置错误代码
    ResetLastError();
    
    // 加载ONNX模型 - 优先直接从文件加载（OnnxCreate）。
    // 注意: OnnxCreateFromBuffer 接受的是 uchar 数组引用，而不是文件名字符串。
    // 在策略测试器/Agent 环境下，模型文件可能位于当前终端的数据路径下的 MQL5\Files 或 Common\Files。
    g_model_handle = OnnxCreate(MODEL_FILE, ONNX_DEFAULT);

    // 如果直接加载失败，尝试从 Common\Files 目录加载（等价于 FILE_COMMON）
    if(g_model_handle == INVALID_HANDLE)
    {
        Print("OnnxCreate 使用默认标志失败，尝试从 Common\\Files 加载...");
        g_model_handle = OnnxCreate(MODEL_FILE, ONNX_COMMON_FOLDER);
    }

    // 如果仍然失败，尝试使用终端数据路径构造绝对路径并从该文件读取缓冲区加载
    if(g_model_handle == INVALID_HANDLE)
    {
        string data_path = TerminalInfoString(TERMINAL_DATA_PATH);
        string abs_path = StringFormat("%s\\MQL5\\Files\\%s", data_path, MODEL_FILE);
        Print("尝试从绝对路径加载模型: ", abs_path);
        int file_handle = FileOpen(abs_path, FILE_READ|FILE_BIN);
        if(file_handle != INVALID_HANDLE)
        {
            // 读取文件内容
            int file_size = (int)FileSize(file_handle);
            uchar buffer[];
            ArrayResize(buffer, file_size);
            FileReadArray(file_handle, buffer, 0, file_size);
            FileClose(file_handle);

            // 从缓冲区创建模型
            g_model_handle = OnnxCreateFromBuffer(buffer, ONNX_DEFAULT);
            if(g_model_handle != INVALID_HANDLE)
                Print("✓ 从绝对路径缓冲区加载ONNX模型");
            else
                Print("❌ 从绝对路径缓冲区加载ONNX模型失败, 错误代码: ", GetLastError());
        }
        else
        {
            Print("无法打开绝对路径模型文件: ", abs_path);

            // 作为最后手段，尝试使用默认的相对 Files 名称打开（原有逻辑）
            int rel_handle = FileOpen(MODEL_FILE, FILE_READ|FILE_BIN);
            if(rel_handle != INVALID_HANDLE)
            {
                int file_size = (int)FileSize(rel_handle);
                uchar buffer[];
                ArrayResize(buffer, file_size);
                FileReadArray(rel_handle, buffer, 0, file_size);
                FileClose(rel_handle);

                g_model_handle = OnnxCreateFromBuffer(buffer, ONNX_DEFAULT);
                if(g_model_handle != INVALID_HANDLE)
                    Print("✓ 从相对 Files 缓冲区加载ONNX模型");
                else
                    Print("❌ 从相对 Files 缓冲区加载ONNX模型失败, 错误代码: ", GetLastError());
            }
        }
    }
    
    if(g_model_handle == INVALID_HANDLE)
    {
        int error = GetLastError();
        Print("❌ 错误: 无法加载ONNX模型 ", MODEL_FILE);
        Print("请确认文件在 MQL5/Files/ 目录下");
        Print("当前数据路径: ", TerminalInfoString(TERMINAL_DATA_PATH));
        Print("模型应在: MQL5\\Files\\", MODEL_FILE);
        Print("错误代码: ", error);
        
        // 尝试列出Files目录内容
        string search = "*.onnx";
        string filename;
        long search_handle = FileFindFirst(search, filename);
        if(search_handle != INVALID_HANDLE)
        {
            Print("在Files目录中找到的ONNX文件:");
            do
            {
                Print("  - ", filename);
            }
            while(FileFindNext(search_handle, filename));
            FileFindClose(search_handle);
        }
        else
        {
            Print("Files目录中没有找到ONNX文件");
        }
        
        return INIT_FAILED;
    }
    
    Print("✓ ONNX模型加载成功: ", MODEL_FILE);
    
    // --- 查询并打印模型元数据（输入/输出形状）
    long in_count = OnnxGetInputCount(g_model_handle);
    long out_count = OnnxGetOutputCount(g_model_handle);
    Print("========================================");
    Print("模型元数据信息:");
    Print("  输入数量: ", in_count);
    Print("  输出数量: ", out_count);
    
    // 打印输入形状详情
    for(int idx=0; idx<in_count; idx++)
    {
        string input_name = OnnxGetInputName(g_model_handle, idx);
        OnnxTypeInfo ti;
        if(OnnxGetInputTypeInfo(g_model_handle, idx, ti))
        {
            if(ti.type == ONNX_TYPE_TENSOR)
            {
                int nd = ArraySize(ti.tensor.dimensions);
                string shape_str = "输入[" + IntegerToString(idx) + "] '" + input_name + "' 形状: [";
                for(int k=0; k<nd; k++)
                {
                    shape_str += IntegerToString(ti.tensor.dimensions[k]);
                    if(k < nd-1) shape_str += ", ";
                }
                shape_str += "]";
                Print("  ", shape_str);
            }
        }
    }
    
    // 打印输出形状详情
    for(int idx2=0; idx2<out_count; idx2++)
    {
        string output_name = OnnxGetOutputName(g_model_handle, idx2);
        OnnxTypeInfo ti2;
        if(OnnxGetOutputTypeInfo(g_model_handle, idx2, ti2))
        {
            if(ti2.type == ONNX_TYPE_TENSOR)
            {
                int nd2 = ArraySize(ti2.tensor.dimensions);
                string shape_str2 = "输出[" + IntegerToString(idx2) + "] '" + output_name + "' 形状: [";
                for(int k2=0; k2<nd2; k2++)
                {
                    shape_str2 += IntegerToString(ti2.tensor.dimensions[k2]);
                    if(k2 < nd2-1) shape_str2 += ", ";
                }
                shape_str2 += "]";
                Print("  ", shape_str2);
            }
        }
    }
    Print("========================================");
    
    // 配置交易对象
    g_trade.SetExpertMagicNumber(20241030);
    g_trade.SetDeviationInPoints(MaxSlippagePoints);
    g_trade.SetTypeFilling(ORDER_FILLING_FOK);
    g_trade.SetAsyncMode(false);
    
    Print("✓ 交易参数配置完成");
    Print("  品种: ", _Symbol);
    Print("  手数: ", TradeLots);
    Print("  最大滑点: ", MaxSlippagePoints, " 点");
    
    // 初始化持仓状态
    UpdatePositionState();
    
    Print("========================================");
    Print("初始化完成,EA已启动");
    Print("========================================\n");
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| 清理函数                                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(g_model_handle != INVALID_HANDLE)
    {
        OnnxRelease(g_model_handle);
        Print("✓ ONNX模型已释放");
    }
    
    Print("EA已停止,原因代码: ", reason);
}

//+------------------------------------------------------------------+
//| Tick函数 - 每次报价更新时调用                                      |
//+------------------------------------------------------------------+
void OnTick()
{
    // 只在新K线产生时交易
    datetime current_bar_time = iTime(_Symbol, PERIOD_CURRENT, 0);
    if(current_bar_time == g_last_bar_time)
        return;
    
    g_last_bar_time = current_bar_time;
    
    // 更新持仓状态
    UpdatePositionState();
    
    // 准备观测数据
    double observation[];
    if(!PrepareObservation(observation))
    {
        Print("❌ 无法准备观测数据");
        return;
    }
    
    // 执行ONNX推理
    int action = RunInference(observation);
    if(action < 0)
    {
        Print("❌ 推理失败");
        return;
    }
    
    // 执行交易动作
    ExecuteAction(action);
    
    // 增加持仓计数
    if(g_position_state != 0)
        g_steps_since_trade++;
}

//+------------------------------------------------------------------+
//| 准备观测数据                                                       |
//+------------------------------------------------------------------+
bool PrepareObservation(double &observation[])
{
    ArrayResize(observation, INPUT_SIZE);
    
    // 计算技术指标
    // 注意: 这里需要与Python训练时的特征计算完全一致
    
    // 特征0: M15_RSI (归一化到0-1)
    double m15_rsi = CalculateRSI(PERIOD_M15, 14, 0) / 100.0;
    
    // 特征1: M15_ATR_norm (标准化)
    double m15_atr_norm = CalculateATRNorm(PERIOD_M15, 14, 100, 0);
    
    // 特征2: M15_MACD_hist_norm
    double m15_macd_norm = CalculateMACDHistNorm(PERIOD_M15, 0);
    
    // 特征3: H1_RSI
    double h1_rsi = CalculateRSI(PERIOD_H1, 14, 0) / 100.0;
    
    // 特征4: H1_Trend_Indicator
    double h1_trend = CalculateTrendIndicator(PERIOD_H1, 50, 0);
    
    // 特征5: H4_RSI
    double h4_rsi = CalculateRSI(PERIOD_H4, 14, 0) / 100.0;
    
    // 特征6: H4_Trend_Indicator
    double h4_trend = CalculateTrendIndicator(PERIOD_H4, 50, 0);
    
    // 特征7: position_state (0=空仓, 1=多单, 2=空单)
    double pos_state = (double)g_position_state;
    
    // 特征8: floating_pnl_normalized
    double pnl_norm = CalculatePnLNorm();
    
    // 特征9: holding_time_normalized (范围0-1)
    double time_norm = MathMin(g_steps_since_trade / 480.0, 1.0);
    
    // 填充观测数组
    observation[0] = m15_rsi;
    observation[1] = m15_atr_norm;
    observation[2] = m15_macd_norm;
    observation[3] = h1_rsi;
    observation[4] = h1_trend;
    observation[5] = h4_rsi;
    observation[6] = h4_trend;
    observation[7] = pos_state;
    observation[8] = pnl_norm;
    observation[9] = time_norm;
    
    // 边界检查
    for(int i = 0; i < INPUT_SIZE; i++)
    {
        if(MathIsValidNumber(observation[i]) == false)
        {
            Print("❌ 观测值无效: index=", i, ", value=", observation[i]);
            return false;
        }
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| 执行ONNX推理                                                      |
//+------------------------------------------------------------------+
int RunInference(const double &observation[])
{
    // 使用 vectorf 输入（10个元素的向量）
    vectorf input_vectorf;
    input_vectorf.Resize(INPUT_SIZE);

    for(int i = 0; i < INPUT_SIZE; i++)
        input_vectorf[i] = (float)observation[i];

    // 创建输出向量（4个动作概率）
    vectorf output_vectorf;
    output_vectorf.Resize(OUTPUT_SIZE);

    // 执行推理 - 使用 ONNX_NO_CONVERSION 确保类型匹配
    bool success = OnnxRun(g_model_handle, ONNX_NO_CONVERSION, input_vectorf, output_vectorf);

    if(!success)
    {
        int err = GetLastError();
        Print("ONNX推理失败, 错误代码: ", err);
        return -1;
    }

    if(output_vectorf.Size() < OUTPUT_SIZE)
    {
        Print("输出向量太小: ", output_vectorf.Size(), ", 期望: ", OUTPUT_SIZE);
        return -1;
    }

    // 找到最大概率对应的动作
    int best_action = 0;
    double max_prob = (double)output_vectorf[0];

    for(int i = 1; i < OUTPUT_SIZE; i++)
    {
        double prob = (double)output_vectorf[i];
        if(prob > max_prob)
        {
            max_prob = prob;
            best_action = i;
        }
    }

    // 打印调试信息
    string probs_str = StringFormat("Probs: [%.3f, %.3f, %.3f, %.3f]",
                                    (double)output_vectorf[0], (double)output_vectorf[1],
                                    (double)output_vectorf[2], (double)output_vectorf[3]);
    Print("推理结果 - 动作: ", best_action, ", ", probs_str);

    return best_action;
}

//+------------------------------------------------------------------+
//| 执行交易动作                                                       |
//+------------------------------------------------------------------+
void ExecuteAction(int action)
{
    string action_names[] = {"Hold", "Open Long", "Open Short", "Close"};
    Print("执行动作: ", action_names[action]);
    
    // 动作0: Hold - 什么都不做
    if(action == 0)
    {
        return;
    }
    
    // 动作1: Open Long - 仅在空仓时
    if(action == 1 && g_position_state == 0)
    {
        double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        if(g_trade.Buy(TradeLots, _Symbol, price, 0, 0, EA_Comment))
        {
            Print("✓ 开多单成功 @ ", price);
            g_position_state = 1;
            g_entry_price = price;
            g_steps_since_trade = 0;
        }
        else
        {
            Print("❌ 开多单失败: ", g_trade.ResultRetcodeDescription());
        }
        return;
    }
    
    // 动作2: Open Short - 仅在空仓时
    if(action == 2 && g_position_state == 0)
    {
        double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        if(g_trade.Sell(TradeLots, _Symbol, price, 0, 0, EA_Comment))
        {
            Print("✓ 开空单成功 @ ", price);
            g_position_state = 2;
            g_entry_price = price;
            g_steps_since_trade = 0;
        }
        else
        {
            Print("❌ 开空单失败: ", g_trade.ResultRetcodeDescription());
        }
        return;
    }
    
    // 动作3: Close - 仅在持仓时
    if(action == 3 && g_position_state != 0)
    {
        CloseAllPositions();
        return;
    }
    
    // 无效动作
    if((action == 1 || action == 2) && g_position_state != 0)
    {
        Print("⚠️  无效动作: 已有持仓,无法开新仓");
    }
    else if(action == 3 && g_position_state == 0)
    {
        Print("⚠️  无效动作: 无持仓,无法平仓");
    }
}

//+------------------------------------------------------------------+
//| 更新持仓状态                                                       |
//+------------------------------------------------------------------+
void UpdatePositionState()
{
    int total = PositionsTotal();
    
    if(total == 0)
    {
        g_position_state = 0;
        g_entry_price = 0.0;
        return;
    }
    
    // 获取第一个持仓(假设只有一个)
    for(int i = 0; i < total; i++)
    {
        if(PositionGetSymbol(i) == _Symbol)
        {
            long type = PositionGetInteger(POSITION_TYPE);
            if(type == POSITION_TYPE_BUY)
                g_position_state = 1;
            else if(type == POSITION_TYPE_SELL)
                g_position_state = 2;
            
            g_entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
            return;
        }
    }
}

//+------------------------------------------------------------------+
//| 平所有持仓                                                         |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
    int total = PositionsTotal();
    for(int i = total - 1; i >= 0; i--)
    {
        if(PositionGetSymbol(i) == _Symbol)
        {
            ulong ticket = PositionGetInteger(POSITION_TICKET);
            if(g_trade.PositionClose(ticket))
            {
                double profit = PositionGetDouble(POSITION_PROFIT);
                Print("✓ 平仓成功, 盈亏: $", profit);
                g_position_state = 0;
                g_entry_price = 0.0;
                g_steps_since_trade = 0;
            }
            else
            {
                Print("❌ 平仓失败: ", g_trade.ResultRetcodeDescription());
            }
        }
    }
}

//+------------------------------------------------------------------+
//| 辅助函数 - 计算RSI                                                 |
//+------------------------------------------------------------------+
double CalculateRSI(ENUM_TIMEFRAMES period, int rsi_period, int shift)
{
    int handle = iRSI(_Symbol, period, rsi_period, PRICE_CLOSE);
    if(handle == INVALID_HANDLE)
        return 50.0;
    
    double buffer[];
    ArraySetAsSeries(buffer, true);
    if(CopyBuffer(handle, 0, shift, 1, buffer) <= 0)
        return 50.0;
    
    IndicatorRelease(handle);
    return buffer[0];
}

//+------------------------------------------------------------------+
//| 辅助函数 - 计算ATR标准化值                                          |
//+------------------------------------------------------------------+
double CalculateATRNorm(ENUM_TIMEFRAMES period, int atr_period, int lookback, int shift)
{
    int handle = iATR(_Symbol, period, atr_period);
    if(handle == INVALID_HANDLE)
        return 0.0;
    
    double buffer[];
    ArraySetAsSeries(buffer, true);
    if(CopyBuffer(handle, 0, shift, lookback, buffer) <= lookback)
        return 0.0;
    
    // 计算标准化: (当前值 - 均值) / 标准差
    double current = buffer[0];
    double mean = 0.0;
    for(int i = 0; i < lookback; i++)
        mean += buffer[i];
    mean /= lookback;
    
    double std = 0.0;
    for(int i = 0; i < lookback; i++)
        std += MathPow(buffer[i] - mean, 2);
    std = MathSqrt(std / lookback);
    
    IndicatorRelease(handle);
    
    if(std > 0)
        return (current - mean) / std;
    return 0.0;
}

//+------------------------------------------------------------------+
//| 辅助函数 - 计算MACD柱状图标准化值                                    |
//+------------------------------------------------------------------+
double CalculateMACDHistNorm(ENUM_TIMEFRAMES period, int shift)
{
    int handle = iMACD(_Symbol, period, 12, 26, 9, PRICE_CLOSE);
    if(handle == INVALID_HANDLE)
        return 0.0;
    
    // 获取柱状图 (buffer 2)
    double hist_buffer[];
    ArraySetAsSeries(hist_buffer, true);
    if(CopyBuffer(handle, 2, shift, 100, hist_buffer) <= 0)
        return 0.0;
    
    double current = hist_buffer[0];
    
    // 简单标准化
    double mean = 0.0;
    for(int i = 0; i < 100; i++)
        mean += hist_buffer[i];
    mean /= 100;
    
    double std = 0.0;
    for(int i = 0; i < 100; i++)
        std += MathPow(hist_buffer[i] - mean, 2);
    std = MathSqrt(std / 100);
    
    IndicatorRelease(handle);
    
    if(std > 0)
        return (current - mean) / std;
    return 0.0;
}

//+------------------------------------------------------------------+
//| 辅助函数 - 计算趋势指标                                             |
//+------------------------------------------------------------------+
double CalculateTrendIndicator(ENUM_TIMEFRAMES period, int ma_period, int shift)
{
    int handle = iMA(_Symbol, period, ma_period, 0, MODE_SMA, PRICE_CLOSE);
    if(handle == INVALID_HANDLE)
        return 0.0;
    
    double ma_buffer[];
    ArraySetAsSeries(ma_buffer, true);
    if(CopyBuffer(handle, 0, shift, 100, ma_buffer) <= 0)
        return 0.0;
    
    double close[];
    ArraySetAsSeries(close, true);
    if(CopyClose(_Symbol, period, shift, 100, close) <= 0)
        return 0.0;
    
    // 计算价格与MA的距离,并标准化
    double distances[];
    ArrayResize(distances, 100);
    for(int i = 0; i < 100; i++)
        distances[i] = (close[i] - ma_buffer[i]) / ma_buffer[i] * 100;
    
    double current = distances[0];
    double mean = 0.0;
    for(int i = 0; i < 100; i++)
        mean += distances[i];
    mean /= 100;
    
    double std = 0.0;
    for(int i = 0; i < 100; i++)
        std += MathPow(distances[i] - mean, 2);
    std = MathSqrt(std / 100);
    
    IndicatorRelease(handle);
    
    if(std > 0)
        return (current - mean) / std;
    return 0.0;
}

//+------------------------------------------------------------------+
//| 辅助函数 - 计算盈亏归一化值                                          |
//+------------------------------------------------------------------+
double CalculatePnLNorm()
{
    if(g_position_state == 0)
        return 0.0;
    
    double current_price = (g_position_state == 1) ? 
                          SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                          SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    
    double pips = 0.0;
    if(g_position_state == 1)  // 多单
        pips = (current_price - g_entry_price) * 10000;
    else  // 空单
        pips = (g_entry_price - current_price) * 10000;
    
    double pip_value = 10.0 * TradeLots;
    double pnl = pips * pip_value;
    
    // 用ATR归一化
    double atr = CalculateATRNorm(PERIOD_M15, 14, 100, 0);
    if(MathAbs(atr) > 0)
        return MathMin(MathMax(pnl / (MathAbs(atr) * pip_value * 100), -10.0), 10.0);
    
    return 0.0;
}

//+------------------------------------------------------------------+
