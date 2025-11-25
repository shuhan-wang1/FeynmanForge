"""
evaluate.py
Feynman-GCPN 模型评估脚本
加载训练好的模型，根据指定的反应式生成费曼图
支持多图生成：为同一反应生成多种不同的费曼图拓扑
"""

import torch
import torch.nn.functional as F
import argparse
import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple, Set
import hashlib

# 添加父目录到路径，确保能导入 rl_training 下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feynman_env import FeynmanDiagramEnv
from models import FeynmanGCPN
from physics_engine import PhysicsConstants
from visualization_bridge import DiagramExporter

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained Feynman-GCPN model')
    
    # 核心参数
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_step_51200.pt',
                      help='Path to the model checkpoint (.pt file)')
    parser.add_argument('--reaction', type=str, default='e+e_bar->e+e_bar',
                      help='Reaction to generate (e.g., "e+e->mu+mu" or "e+gamma->e+gamma")')
    
    # 多图生成参数
    parser.add_argument('--num-diagrams', type=int, default=1,
                      help='Number of distinct diagrams to generate (default: 1)')
    parser.add_argument('--max-attempts', type=int, default=50,
                      help='Maximum attempts to find distinct diagrams (default: 50)')
    parser.add_argument('--temperature', type=float, default=1.0,
                      help='Sampling temperature for diversity (higher = more diverse)')
    
    # 输出设置
    parser.add_argument('--output', type=str, default='evaluation/result.json',
                      help='Output JSON file path (for single diagram) or base path (for multi)')
    parser.add_argument('--deterministic', action='store_true', default=False,
                      help='Use deterministic policy (argmax) - disable for multi-diagram')
    
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


def compute_diagram_topology_hash(env) -> str:
    """
    计算费曼图的拓扑哈希值
    用于识别不同的图结构（不考虑几何位置，只考虑连接关系）
    
    拓扑由以下定义：
    1. 顶点类型（initial/final/interaction）的数量
    2. 边的连接关系（source->target + particle_type）
    3. 内部传播子的类型
    """
    # 统计顶点类型
    vertex_types = sorted([v['type'] for v in env.vertices])
    
    # 规范化边的表示 (排序以获得一致的哈希)
    edge_signatures = []
    for edge in env.edges:
        # 使用边的连接关系和粒子类型
        sig = f"{edge['source']}->{edge['target']}:{edge['particle_id']}:{'anti' if edge['is_anti'] else 'normal'}"
        edge_signatures.append(sig)
    edge_signatures.sort()
    
    # 计算内部顶点数量（非初态/末态）
    internal_vertices = sum(1 for v in env.vertices if v['type'] not in ['initial', 'final'])
    
    # 组合成拓扑签名
    topo_sig = f"V:{','.join(vertex_types)}|E:{';'.join(edge_signatures)}|I:{internal_vertices}"
    
    # 返回哈希值
    return hashlib.md5(topo_sig.encode()).hexdigest()[:12]


def get_diagram_channel_type(env) -> str:
    """
    尝试识别费曼图的散射道类型 (s/t/u channel)
    基于内部传播子的连接方式
    """
    # 简化版本：基于内部顶点的连接模式
    internal_verts = [v for v in env.vertices if v['type'] not in ['initial', 'final']]
    
    if len(internal_verts) == 0:
        return "contact"  # 接触相互作用
    elif len(internal_verts) == 1:
        # 单顶点 - 可能是 s-channel
        return "s-channel (single vertex)"
    elif len(internal_verts) == 2:
        # 两个顶点 - 可能是 t/u channel
        # 检查是否有水平传播子（s-channel）或交叉传播子（t/u）
        # 简化：根据内部顶点的x坐标判断
        x_coords = [v['x'] for v in internal_verts]
        if abs(x_coords[0] - x_coords[1]) < 50:
            return "t/u-channel (crossed)"
        else:
            return "s-channel (annihilation)"
    else:
        return f"complex ({len(internal_verts)} vertices)"


def extract_vertex_states(env) -> List[Dict]:
    """
    从环境中提取顶点的量子数状态
    用于物理门控
    """
    vertex_states = []
    
    for vertex in env.vertices:
        connected_edges = [env.edges[eid] for eid in vertex['connected_edges']]
        
        incoming = [e for e in connected_edges if e['target'] == vertex['id']]
        outgoing = [e for e in connected_edges if e['source'] == vertex['id']]
        
        state = {
            'charge_in': sum(env._get_charge(e) for e in incoming),
            'charge_out': sum(env._get_charge(e) for e in outgoing),
            'lepton_in': sum(env._get_lepton(e) for e in incoming),
            'lepton_out': sum(env._get_lepton(e) for e in outgoing),
            'baryon_in': sum(env._get_baryon(e) for e in incoming),
            'baryon_out': sum(env._get_baryon(e) for e in outgoing),
            'colors_in': [e['color'] for e in incoming if e['color']],
            'colors_out': [e['color'] for e in outgoing if e['color']]
        }
        
        vertex_states.append(state)
    
    return vertex_states


def detect_model_config(checkpoint, device):
    """从 Checkpoint 权重中自动推断模型超参数"""
    state_dict = checkpoint['model_state_dict']
    
    # 1. 推断 hidden_dim (从 node_encoder 权重的输出维度)
    # encoder.node_encoder.0.weight shape is [hidden_dim, node_input_dim]
    if 'encoder.node_encoder.0.weight' in state_dict:
        hidden_dim = state_dict['encoder.node_encoder.0.weight'].shape[0]
    else:
        print("⚠️ Cannot detect hidden_dim, using default 768")
        hidden_dim = 768
        
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

def generate_diagram(model, env, device, deterministic=True, max_steps=50, verbose=True, apply_physics_gate=True):
    """执行单次图生成
    
    Args:
        model: 训练好的模型
        env: 费曼图环境
        device: 计算设备
        deterministic: 是否使用确定性策略
        max_steps: 最大步数
        verbose: 是否打印详细信息
        apply_physics_gate: 是否应用物理门控
    
    Returns:
        total_reward: 总奖励
        actions_taken: 执行的动作列表
        topo_hash: 图拓扑哈希
        channel_type: 散射道类型
    """
    state, info = env.reset()
    done = False
    total_reward = 0.0
    actions_taken = []
    
    # Action name mapping for debug
    action_names = ['CONNECT', 'BRANCH', 'SET_TYPE', 'TERMINATE', 'MERGE']
    
    if verbose:
        print(f"\n🎨 Generating diagram for: {env.initial_particles} -> {env.final_particles}")
    
    for step in range(max_steps):
        state = state.to(device)
        
        # 获取 action masks 从环境
        action_masks_np = env.get_action_masks()
        action_masks = {k: torch.from_numpy(v).to(device) for k, v in action_masks_np.items()}
        
        # 获取顶点状态用于物理门控
        vertex_states = extract_vertex_states(env)
        
        # DEBUG: Print action masks on first step
        if verbose and step == 0:
            print(f"\n[DEBUG] Action masks on step 0:")
            print(f"  action_type: {action_masks_np['action_type']}")
            print(f"  source_vertex shape: {action_masks_np['source_vertex'].shape}")
            print(f"  target_vertex shape: {action_masks_np['target_vertex'].shape}")
        
        # 获取动作（使用 action_masks 和 physics gate）
        action = model.get_action(
            state, 
            vertex_states=vertex_states,
            action_masks=action_masks, 
            deterministic=deterministic,
            apply_physics_gate=apply_physics_gate
        )
        
        # DEBUG: Print action details and probabilities on first few steps
        if verbose and step < 3:
            with torch.no_grad():
                output = model.forward(state.to(device), vertex_states=vertex_states, action_masks=action_masks, return_value=False)
                print(f"[DEBUG Step {step+1}] Selected action: type={action['action_type']}, vertex_idx={action['vertex_idx']}, target_vertex={action['target_vertex']}, particle={action['particle_type']}")
                print(f"  Source vertex probs: {output['source_vertex_probs'].cpu().numpy()[:6]}")
                print(f"  Target vertex probs: {output['target_vertex_probs'].cpu().numpy()[:6]}")
                print(f"  Target vertex masks: {action_masks_np['target_vertex'][:6]}")
        
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
        
        if verbose:
            print(f"   Step {step+1}: {action_desc} | Reward: {reward:.2f}")
        
        if terminated or truncated:
            if verbose:
                print(f"   🛑 Evaluation ended (Terminated: {terminated}, Truncated: {truncated})")
            break
            
        state = next_state
    
    # 计算图的拓扑哈希和散射道类型
    topo_hash = compute_diagram_topology_hash(env)
    channel_type = get_diagram_channel_type(env)
    
    return total_reward, actions_taken, topo_hash, channel_type


def generate_multiple_diagrams(model, env, device, num_diagrams=5, max_attempts=50, 
                               verbose=True, apply_physics_gate=True):
    """
    生成多种不同拓扑的费曼图
    
    通过随机采样（非确定性策略）来探索不同的图拓扑
    使用拓扑哈希来去重，确保生成的图是不同的
    
    Args:
        model: 训练好的模型
        env: 费曼图环境
        device: 计算设备
        num_diagrams: 目标生成的不同图数量
        max_attempts: 最大尝试次数
        verbose: 是否打印详细信息
        apply_physics_gate: 是否应用物理门控
    
    Returns:
        diagrams: 生成的图列表 [{'reward', 'actions', 'shapes', 'topo_hash', 'channel'}]
    """
    discovered_topos: Set[str] = set()
    diagrams = []
    
    print(f"\n{'='*60}")
    print(f"🎯 Generating {num_diagrams} distinct Feynman diagrams")
    print(f"   Max attempts: {max_attempts}")
    print(f"   Physics Gate: {'Enabled' if apply_physics_gate else 'Disabled'}")
    print(f"{'='*60}")
    
    for attempt in range(max_attempts):
        if len(diagrams) >= num_diagrams:
            break
            
        # 使用非确定性策略来获得多样性
        reward, actions, topo_hash, channel = generate_diagram(
            model, env, device, 
            deterministic=False,  # 关键：使用随机采样
            verbose=False,  # 减少输出
            apply_physics_gate=apply_physics_gate
        )
        
        # 检查是否是新拓扑
        if topo_hash not in discovered_topos:
            discovered_topos.add(topo_hash)
            
            # 获取图的形状数据
            shapes = DiagramExporter.env_to_shapes(env)
            
            diagram_info = {
                'id': len(diagrams) + 1,
                'topo_hash': topo_hash,
                'channel': channel,
                'reward': reward,
                'steps': len(actions),
                'actions': actions,
                'shapes': shapes
            }
            diagrams.append(diagram_info)
            
            print(f"\n✅ Found diagram #{len(diagrams)}: {channel}")
            print(f"   Topo hash: {topo_hash}")
            print(f"   Reward: {reward:.2f}")
            print(f"   Steps: {len(actions)}")
            
            if verbose:
                env.render()
        else:
            if attempt % 10 == 0:
                print(f"   [Attempt {attempt+1}] Found duplicate topology, continuing search...")
    
    print(f"\n{'='*60}")
    print(f"📊 Summary: Found {len(diagrams)} distinct diagrams in {attempt+1} attempts")
    print(f"{'='*60}")
    
    return diagrams

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
        node_input_dim=7,
        edge_input_dim=22,
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
    
    # 6. 生成图（单图或多图模式）
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with torch.no_grad():
        if args.num_diagrams > 1:
            # 多图生成模式
            print(f"\n🔀 Multi-diagram generation mode: {args.num_diagrams} diagrams")
            
            diagrams = generate_multiple_diagrams(
                model, env, device,
                num_diagrams=args.num_diagrams,
                max_attempts=args.max_attempts,
                verbose=True,
                apply_physics_gate=True
            )
            
            # 保存所有图到一个 JSON 文件
            output_data = {
                "reaction": args.reaction,
                "num_diagrams": len(diagrams),
                "diagrams": diagrams
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # 同时为每个图单独保存
            base_path = os.path.splitext(args.output)[0]
            for i, diagram in enumerate(diagrams):
                diagram_path = f"{base_path}_diagram_{i+1}.json"
                DiagramExporter.save_diagram(diagram['shapes'], diagram_path, metadata={
                    "reaction": args.reaction,
                    "diagram_id": diagram['id'],
                    "channel": diagram['channel'],
                    "topo_hash": diagram['topo_hash'],
                    "reward": diagram['reward']
                })
            
            print("\n" + "="*60)
            print(f"🏆 Generated {len(diagrams)} distinct Feynman diagrams!")
            print(f"💾 Combined result saved to: {args.output}")
            print(f"💾 Individual diagrams saved to: {base_path}_diagram_*.json")
            print("="*60)
            
            # 打印各图的散射道类型
            print("\n📋 Diagram Summary:")
            for d in diagrams:
                print(f"   #{d['id']}: {d['channel']} (reward: {d['reward']:.2f})")
            
        else:
            # 单图生成模式（原有逻辑）
            total_reward, actions, topo_hash, channel = generate_diagram(
                model, env, device, args.deterministic,
                apply_physics_gate=True
            )
            
            # 导出结果
            shapes = DiagramExporter.env_to_shapes(env)
            
            output_data = {
                "reaction": args.reaction,
                "total_reward": total_reward,
                "steps": len(actions),
                "topo_hash": topo_hash,
                "channel": channel,
                "actions": actions,
                "shapes": shapes
            }
            
            DiagramExporter.save_diagram(shapes, args.output, metadata={
                "reward": total_reward,
                "reaction": args.reaction,
                "channel": channel,
                "topo_hash": topo_hash,
                "actions": actions
            })
            
            print("\n" + "="*50)
            print(f"🏆 Final Reward: {total_reward:.2f}")
            print(f"📊 Channel Type: {channel}")
            print(f"🔑 Topology Hash: {topo_hash}")
            print(f"💾 Result saved to: {args.output}")
            print("="*50)
            
            # 打印简单的 ASCII 图示 (如果可能)
            print("\nGraph Structure:")
            env.render()

if __name__ == '__main__':
    main()