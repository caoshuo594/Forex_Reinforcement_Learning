"""
ONNX导出脚本 - 将训练好的PPO模型导出为ONNX格式供MQL5使用
严格遵循MQL5+ONNX集成标准规范
"""

import torch
import torch.onnx
import numpy as np
import onnx
import onnxruntime as ort
from stable_baselines3 import PPO
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# ONNX模型配置 (固定参数)
# ============================================
BATCH_SIZE = 1
NUM_FEATURES = 10  # 状态空间维度
OPSET_VERSION = 11  # MQL5兼容性最好的版本


class PolicyWrapper(torch.nn.Module):
    """
    包装PPO策略网络,简化为单一输入输出
    """
    def __init__(self, policy_net):
        super(PolicyWrapper, self).__init__()
        self.policy_net = policy_net
    
    def forward(self, obs):
        """
        前向传播
        
        Args:
            obs: 观测向量 (batch_size, num_features)
            
        Returns:
            action_probs: 动作概率 (batch_size, num_actions)
        """
        # 提取特征
        features = self.policy_net.extract_features(obs)
        
        # 获取动作logits
        latent_pi = self.policy_net.mlp_extractor.forward_actor(features)
        action_logits = self.policy_net.action_net(latent_pi)
        
        # 转换为概率分布
        action_probs = torch.softmax(action_logits, dim=-1)
        
        return action_probs


def export_onnx(
    model_path: str,
    output_path: str = "forex_policy.onnx"
) -> bool:
    """
    导出ONNX模型
    
    Args:
        model_path: 训练好的PPO模型路径(.zip)
        output_path: ONNX模型保存路径
        
    Returns:
        是否导出成功
    """
    try:
        logger.info(f"加载模型: {model_path}")
        model = PPO.load(model_path, device='cpu')
        
        # 提取策略网络
        policy = model.policy
        policy.eval()
        policy.to('cpu')
        
        # 包装策略网络
        wrapped_policy = PolicyWrapper(policy)
        wrapped_policy.eval()
        
        # 创建dummy输入
        dummy_input = torch.randn(BATCH_SIZE, NUM_FEATURES, dtype=torch.float32)
        
        logger.info("开始导出ONNX...")
        logger.info(f"输入形状: ({BATCH_SIZE}, {NUM_FEATURES})")
        logger.info(f"总共需要 {BATCH_SIZE * NUM_FEATURES} 个输入值")
        
        # 导出ONNX
        torch.onnx.export(
            wrapped_policy,
            dummy_input,
            output_path,
            input_names=['observation'],
            output_names=['action_probs'],
            dynamic_axes={},  # 不使用动态维度
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        logger.info(f"✓ ONNX模型导出成功: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX导出失败: {e}")
        return False


def validate_onnx(onnx_path: str) -> bool:
    """
    验证ONNX模型
    
    Args:
        onnx_path: ONNX模型路径
        
    Returns:
        是否验证通过
    """
    try:
        logger.info("="*60)
        logger.info("ONNX模型验证")
        logger.info("="*60)
        
        # 步骤1: 检查模型结构
        logger.info("步骤1: 检查模型结构...")
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info("✓ ONNX模型结构有效")
        
        # 步骤2: 列出所有算子
        logger.info("\n步骤2: 检查算子兼容性...")
        operators = set()
        for node in model.graph.node:
            operators.add(node.op_type)
        logger.info(f"✓ 使用的算子: {sorted(operators)}")
        
        # 警告高风险算子
        risky_ops = {'Einsum', 'NonMaxSuppression', 'TopK', 'DynamicQuantizeLinear'}
        found_risky = operators & risky_ops
        if found_risky:
            logger.warning(f"⚠️  警告: 发现高风险算子 {found_risky}")
        else:
            logger.info("✓ 未发现高风险算子")
        
        # 步骤3: 测试推理
        logger.info("\n步骤3: 测试推理...")
        session = ort.InferenceSession(onnx_path)
        
        # 测试多组输入
        test_cases = 5
        for i in range(test_cases):
            test_input = np.random.randn(BATCH_SIZE, NUM_FEATURES).astype(np.float32)
            output = session.run(None, {'observation': test_input})
            
            if i == 0:
                logger.info(f"✓ 测试推理成功")
                logger.info(f"  输入形状: {test_input.shape}")
                logger.info(f"  输出形状: {output[0].shape}")
                logger.info(f"  输出范围: [{np.min(output[0]):.4f}, {np.max(output[0]):.4f}]")
                logger.info(f"  输出示例: {output[0][0]}")
        
        # 步骤4: 生成数据规格书
        logger.info("\n" + "="*60)
        logger.info("【数据规格书 - 请保存此信息用于MQL5集成】")
        logger.info("="*60)
        logger.info(f"模型文件: {Path(onnx_path).name}")
        logger.info(f"输入名称: observation")
        logger.info(f"输入形状: ({BATCH_SIZE}, {NUM_FEATURES})")
        logger.info(f"输入类型: float32")
        logger.info(f"输入总数: {BATCH_SIZE * NUM_FEATURES} 个浮点数")
        logger.info(f"输出名称: action_probs")
        logger.info(f"输出形状: ({BATCH_SIZE}, 4)")
        logger.info(f"输出含义: [Hold概率, OpenLong概率, OpenShort概率, Close概率]")
        logger.info(f"动作选择: 取最大概率对应的索引作为动作")
        logger.info("="*60)
        
        # 保存规格书到JSON
        spec = {
            "model_file": Path(onnx_path).name,
            "input_name": "observation",
            "input_shape": [BATCH_SIZE, NUM_FEATURES],
            "input_type": "float32",
            "input_size": BATCH_SIZE * NUM_FEATURES,
            "output_name": "action_probs",
            "output_shape": [BATCH_SIZE, 4],
            "output_type": "float32",
            "action_mapping": {
                "0": "Hold",
                "1": "Open Long",
                "2": "Open Short",
                "3": "Close Position"
            }
        }
        
        spec_path = onnx_path.replace('.onnx', '_spec.json')
        with open(spec_path, 'w', encoding='utf-8') as f:
            json.dump(spec, f, indent=4, ensure_ascii=False)
        logger.info(f"\n✓ 规格书已保存: {spec_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX验证失败: {e}")
        return False


def generate_test_cases(onnx_path: str, n_cases: int = 10):
    """
    生成测试用例用于交叉验证
    
    Args:
        onnx_path: ONNX模型路径
        n_cases: 测试用例数量
    """
    try:
        logger.info(f"\n生成 {n_cases} 组测试用例用于交叉验证...")
        
        session = ort.InferenceSession(onnx_path)
        
        test_cases = []
        for i in range(n_cases):
            # 生成测试输入
            test_input = np.random.randn(BATCH_SIZE, NUM_FEATURES).astype(np.float32)
            
            # 获取输出
            output = session.run(None, {'observation': test_input})
            action_probs = output[0][0]
            action = int(np.argmax(action_probs))
            
            test_cases.append({
                'case_id': i,
                'input': test_input[0].tolist(),
                'output_probs': action_probs.tolist(),
                'action': action
            })
        
        # 保存测试用例
        test_path = onnx_path.replace('.onnx', '_test_cases.json')
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=4, ensure_ascii=False)
        
        logger.info(f"✓ 测试用例已保存: {test_path}")
        logger.info("  请在MQL5中使用相同输入验证输出一致性")
        logger.info("  允许的误差范围: ±0.001 (浮点精度差异)")
        
    except Exception as e:
        logger.error(f"❌ 生成测试用例失败: {e}")


def export_normalization_params():
    """
    导出归一化参数(如果使用了标准化)
    
    注意: 当前环境设计中未使用额外的标准化,观测值已在环境中归一化
    """
    logger.info("\n归一化参数说明:")
    logger.info("="*60)
    logger.info("当前环境设计中,所有观测值已在环境内部归一化")
    logger.info("MQL5端不需要额外的标准化步骤")
    logger.info("直接将10个特征值填充到输入数组即可")
    logger.info("="*60)
    
    # 特征说明
    feature_desc = {
        "feature_0": "M15_RSI (范围: 0-1)",
        "feature_1": "M15_ATR_norm (范围: -5到5)",
        "feature_2": "M15_MACD_hist_norm (范围: -5到5)",
        "feature_3": "H1_RSI (范围: 0-1)",
        "feature_4": "H1_Trend_Indicator (范围: -5到5)",
        "feature_5": "H4_RSI (范围: 0-1)",
        "feature_6": "H4_Trend_Indicator (范围: -5到5)",
        "feature_7": "position_state (0=空仓, 1=多单, 2=空单)",
        "feature_8": "floating_pnl_normalized (范围: -10到10)",
        "feature_9": "holding_time_normalized (范围: 0-1)"
    }
    
    with open('feature_description.json', 'w', encoding='utf-8') as f:
        json.dump(feature_desc, f, indent=4, ensure_ascii=False)
    
    logger.info("✓ 特征说明已保存: feature_description.json")


def main():
    """主函数"""
    import sys
    
    # 默认路径
    MODEL_PATH = "./models_test/ppo_forex_final.zip"
    ONNX_PATH = "./models/forex_policy.onnx"
    
    # 命令行参数
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        ONNX_PATH = sys.argv[2]
    
    # 检查模型文件是否存在
    if not Path(MODEL_PATH).exists():
        logger.error(f"❌ 模型文件不存在: {MODEL_PATH}")
        logger.info("请先运行 train.py 训练模型")
        logger.info(f"可用模型:")
        for p in Path(".").rglob("*.zip"):
            if "ppo_forex" in str(p):
                logger.info(f"  - {p}")
        return
    
    try:
        # 1. 导出ONNX
        success = export_onnx(MODEL_PATH, ONNX_PATH)
        if not success:
            return
        
        # 2. 验证ONNX
        success = validate_onnx(ONNX_PATH)
        if not success:
            return
        
        # 3. 生成测试用例
        generate_test_cases(ONNX_PATH, n_cases=10)
        
        # 4. 导出归一化参数说明
        export_normalization_params()
        
        logger.info("\n" + "="*60)
        logger.info("ONNX导出完成!")
        logger.info("="*60)
        logger.info(f"✓ ONNX模型: {ONNX_PATH}")
        logger.info(f"✓ 规格书: {ONNX_PATH.replace('.onnx', '_spec.json')}")
        logger.info(f"✓ 测试用例: {ONNX_PATH.replace('.onnx', '_test_cases.json')}")
        logger.info(f"✓ 特征说明: feature_description.json")
        logger.info("\n下一步:")
        logger.info("1. 将ONNX模型复制到MT5的MQL5/Files目录")
        logger.info("2. 参考规格书和测试用例编写MQL5推理代码")
        logger.info("3. 使用测试用例验证MQL5输出一致性")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ 导出过程失败: {e}")
        raise


if __name__ == "__main__":
    main()
