"""
evaluate.py
Feynman-GCPN 模型评估脚本
加载训练好的模型，根据指定的反应式生成费曼图
"""

import torch
import argparse
import os
import sys
import json
import numpy as np

# 添加父目录到路径，确保能导入 rl_training 下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feynman_env import FeynmanDiagramEnv
from models import FeynmanGCPN
from physics_engine import PhysicsConstants
from visualization_bridge import DiagramExporter

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained Feynman-GCPN model')
    
    # 核心参数
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint (.pt file)')
    parser.add_argument('--reaction', type=str, default='e+e->mu+mu',
                      help='Reaction to generate (e.g., "e+e->mu+mu" or "e+gamma->e+gamma")')
    
    # 输出设置
    parser.add_argument('--output', type=str, default='evaluation_result.json',
                      help='Output JSON file path')
    parser.add_argument('--deterministic', action='store_true', default=True,
                      help='Use deterministic policy (argmax) for generation')
    
    # 硬件设置
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (auto/cuda/cpu)')
    
    return parser.parse_args()

def parse_reaction(reaction_str):
    """解析反应字符串，支持 _bar 后缀"""
    if '->' not in reaction_str:
        raise ValueError("Reaction format error: Must contain '->'")
    
    initial_str, final_str = reaction_str.split('->')
    # 去除空格并分割
    initial_particles = [p.strip() for p in initial_str.split('+')]
    final_particles = [p.strip() for p in final_str.split('+')]
    
    return initial_particles, final_particles

def detect_model_config(checkpoint, device):
    """从 Checkpoint 权重中自动推断模型超参数"""
    state_dict = checkpoint['model_state_dict']
    
    # 1. 推断 hidden_dim (从 node_encoder 权重的输出维度)
    # encoder.node_encoder.weight shape is [hidden_dim, node_input_dim]
    if 'encoder.node_encoder.weight' in state_dict:
        hidden_dim = state_dict['encoder.node_encoder.weight'].shape[0]
    else:
        print("⚠️ Cannot detect hidden_dim, using default 128")
        hidden_dim = 128
        
    # 2. 推断 MPNN 层数 (计算 encoder.mp_layers.X 的最大索引)
    max_layer_idx = -1
    for key in state_dict.keys():
        if key.startswith('encoder.mp_layers.'):
            # key format: encoder.mp_layers.0.message_mlp.0.weight
            try:
                layer_idx = int(key.split('.')[2])
                if layer_idx > max_layer_idx:
                    max_layer_idx = layer_idx
            except:
                pass
    
    num_mp_layers = max_layer_idx + 1 if max_layer_idx >= 0 else 3
    
    print(f"🔍 Auto-detected model config: hidden_dim={hidden_dim}, layers={num_mp_layers}")
    return hidden_dim, num_mp_layers

def generate_diagram(model, env, device, deterministic=True, max_steps=50):
    """执行单次生成"""
    state, info = env.reset()
    done = False
    total_reward = 0.0
    actions_taken = []
    
    # Action name mapping for debug
    action_names = ['CONNECT', 'BRANCH', 'SET_TYPE', 'TERMINATE', 'MERGE']
    
    print(f"\n🎨 Generating diagram for: {env.initial_particles} -> {env.final_particles}")
    
    for step in range(max_steps):
        state = state.to(device)
        
        # 获取动作
        action = model.get_action(state, deterministic=deterministic)
        
        # 记录动作
        action_desc = f"{action_names[action['action_type']]} (v{action['vertex_idx']} -> v{action['target_vertex']}, p={action['particle_type']})"
        
        # 执行环境步
        next_state, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        actions_taken.append({
            'step': step + 1,
            'action': action,
            'desc': action_desc,
            'reward': reward,
            'success': reward > -0.1 # 简单的成功判定
        })
        
        print(f"   Step {step+1}: {action_desc} | Reward: {reward:.2f}")
        
        if terminated or truncated:
            print(f"   🛑 Evaluation ended (Terminated: {terminated}, Truncated: {truncated})")
            break
            
        state = next_state
        
    return total_reward, actions_taken

def main():
    args = parse_args()
    
    # 1. 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"🖥️  Using device: {device}")
    
    # 2. 解析反应
    try:
        initial_p, final_p = parse_reaction(args.reaction)
        print(f"🧪 Reaction: {' + '.join(initial_p)} -> {' + '.join(final_p)}")
    except Exception as e:
        print(f"❌ Error parsing reaction: {e}")
        return

    # 3. 加载 Checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    print(f"📂 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 4. 自动配置并初始化模型
    hidden_dim, num_mp_layers = detect_model_config(checkpoint, device)
    
    num_particle_types = len(PhysicsConstants.get_all_particles()) + len(PhysicsConstants.BOSONS)
    
    model = FeynmanGCPN(
        node_input_dim=9,
        edge_input_dim=21,
        hidden_dim=hidden_dim,
        num_mp_layers=num_mp_layers,
        num_action_types=5,
        num_particle_types=num_particle_types,
        max_vertices=10
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded successfully")
    
    # 5. 初始化环境
    # 注意：这里不使用并行环境，只用单个环境进行评估
    env = FeynmanDiagramEnv(
        initial_state=initial_p,
        final_state=final_p,
        max_vertices=10,
        max_steps=50
    )
    
    # 6. 生成图
    with torch.no_grad():
        total_reward, actions = generate_diagram(model, env, device, args.deterministic)
    
    # 7. 导出结果
    shapes = DiagramExporter.env_to_shapes(env)
    
    output_data = {
        "reaction": args.reaction,
        "total_reward": total_reward,
        "steps": len(actions),
        "actions": actions,
        "shapes": shapes
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    DiagramExporter.save_diagram(shapes, args.output, metadata={
        "reward": total_reward,
        "reaction": args.reaction,
        "actions": actions
    })
    
    print("\n" + "="*50)
    print(f"🏆 Final Reward: {total_reward:.2f}")
    print(f"💾 Result saved to: {args.output}")
    print("="*50)
    
    # 打印简单的 ASCII 图示 (如果可能)
    print("\nGraph Structure:")
    env.render()

if __name__ == '__main__':
    main()