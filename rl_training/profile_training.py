"""
Profiling script to find the REAL bottleneck
"""

import time
import torch
from torch_geometric.data import Batch
from config import Config
from models import FeynmanGCPN
from parallel_env import make_parallel_envs

def profile_single_rollout_step(num_envs=64):
    """测试单个rollout step的各个部分耗时"""
    
    config = Config()
    
    # 创建环境
    print(f"Creating {num_envs} environments...")
    reactions = config.get_training_reactions()
    first_reaction = reactions[0]
    
    parallel_env = make_parallel_envs(
        initial_state=first_reaction['initial'],
        final_state=first_reaction['final'],
        num_envs=num_envs,
        max_vertices=config.max_vertices,
        max_steps=config.max_steps,
        reward_weights=config.known_laws_reward
    )
    
    # 创建模型
    from particle_utils import get_particle_list
    num_particle_types = len(get_particle_list())
    
    model = FeynmanGCPN(
        node_input_dim=9,
        edge_input_dim=21,
        hidden_dim=128,
        num_mp_layers=3,
        num_action_types=4,
        num_particle_types=num_particle_types,
        max_vertices=config.max_vertices,
        lambda_penalty=config.physics_penalty,
        fixed_dim=config.fixed_dim,
        learnable_dim=config.learnable_dim,
        sparsity_weight=config.sparsity_weight
    ).cuda()
    
    # Reset环境
    states, infos = parallel_env.reset()
    
    # 🚀 OPTIMIZATION: Get precomputed vertex_states
    vertex_states_list = [info['vertex_states'] for info in infos]
    
    print(f"\n{'='*60}")
    print(f"Profiling {num_envs} environments")
    print(f"{'='*60}\n")
    
    # 重复测试10次取平均
    n_iterations = 10
    
    times = {
        'batch_creation': [],
        'vertex_extraction': [],
        'gpu_transfer': [],
        'forward_pass': [],
        'action_sampling': [],
        'env_step': [],
        'total': []
    }
    
    for iteration in range(n_iterations):
        t_total_start = time.time()
        
        # 1. Batch creation
        t_start = time.time()
        batched_states = Batch.from_data_list(states)
        times['batch_creation'].append(time.time() - t_start)
        
        # 2. GPU transfer
        t_start = time.time()
        batched_states = batched_states.to('cuda', non_blocking=True)
        torch.cuda.synchronize()  # 确保传输完成
        times['gpu_transfer'].append(time.time() - t_start)
        
        # 3. 🚀 Vertex extraction (OPTIMIZED!)
        # Old: Nested Python loops extracting from envs
        # New: Instant read from precomputed info dict
        t_start = time.time()
        # vertex_states_list is already available - just measure overhead
        _ = vertex_states_list  # No-op to measure baseline
        times['vertex_extraction'].append(time.time() - t_start)
        
        # 4. 🚀 Forward pass (OPTIMIZED!)
        t_start = time.time()
        with torch.no_grad():
            # MPNN encoder (batch)
            node_embeddings, graph_embeddings = model.encoder(batched_states)
            
            # Ensure graph_embeddings is [num_envs, hidden_dim]
            if graph_embeddings.dim() == 1:
                graph_embeddings = graph_embeddings.unsqueeze(0)
            
            # 🚀 OPTIMIZED: Batched policy head call!
            # This replaces the per-env for-loop (lines 115-122 in old code)
            policy_output = model.policy_head.forward_batch(
                graph_embeddings,
                vertex_states_list=vertex_states_list,
                apply_physics_gate=True
            )
            
            # Also compute value in batch
            values = model.value_head(graph_embeddings)
        torch.cuda.synchronize()
        times['forward_pass'].append(time.time() - t_start)

        
        # 5. Action sampling (简化版)
        t_start = time.time()
        # 模拟action采样
        times['action_sampling'].append(time.time() - t_start)
        
        # 6. Environment step
        t_start = time.time()
        actions_list = [
            {'action_type': 3, 'vertex_idx': 0, 'particle_type': 0, 'target_vertex': 0}
            for _ in range(num_envs)
        ]
        next_states, rewards, terminateds, truncateds, infos = parallel_env.step(actions_list)
        
        # 🚀 Update vertex_states from worker-computed data
        vertex_states_list = [info['vertex_states'] for info in infos]

        times['env_step'].append(time.time() - t_start)
        
        states = next_states
        times['total'].append(time.time() - t_total_start)
    
    # 打印结果
    print("\n时间分析（平均值，单位：毫秒）:")
    print(f"{'组件':<25} {'耗时(ms)':<12} {'占比':<10}")
    print("-" * 50)
    
    total_avg = sum(times['total']) / n_iterations * 1000
    
    for component, time_list in times.items():
        avg_time_ms = sum(time_list) / n_iterations * 1000
        percentage = (avg_time_ms / total_avg * 100) if total_avg > 0 else 0
        print(f"{component:<25} {avg_time_ms:>10.2f}ms  {percentage:>8.1f}%")
    
    print("-" * 50)
    print(f"{'TOTAL':<25} {total_avg:>10.2f}ms  {100:>8.1f}%")
    
    # 估算FPS
    steps_per_sec = 1.0 / (total_avg / 1000)
    fps = steps_per_sec * num_envs
    print(f"\n估算性能:")
    print(f"  Steps/sec: {steps_per_sec:.1f}")
    print(f"  FPS (frames): {fps:.0f}")
    print(f"  每个rollout(512步)需要: {512 / steps_per_sec:.1f}秒")
    
    parallel_env.close()

if __name__ == "__main__":
    import sys
    
    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    profile_single_rollout_step(num_envs)
