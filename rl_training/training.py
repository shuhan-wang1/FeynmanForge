"""
优化的PPO训练循环 - 针对GPU利用率优化

关键优化:
1. 真正的批量GPU推理 (使用PyG Batch)
2. 减少CPU-GPU同步 (批量处理actions)
3. 异步数据传输 (pin_memory + non_blocking)
4. 高效的rollout收集

======================================
BUG FIXES (V8.1):
======================================
BUG 1: vertex_states=None 导致Physics Gate失效
  - 位置: collect_rollout_optimized 的 _batch_forward 调用
  - 修复: 从每个环境提取vertex_states并传入模型
  - 影响: Physics Gate现在能正确计算守恒律违规
  
BUG 2: 没有Multi-task Training  
  - 问题: 所有环境永远只跑muon_decay
  - 修复: 添加_cycle_training_env()方法，每次episode完成后切换反应类型
  - 影响: 模型现在会在6种不同反应间循环训练 (muon_decay → tau_decay → z_to_uu → ...)
  
BUG 3: 模型不支持真正的Batch Forward
  - 问题: policy_head需要vertex_states才能正确计算
  - 修复: _batch_forward中分两步：(1) batch处理MPNN encoder (2) per-env处理policy head
  - 影响: 保持了batch处理的效率，同时正确支持Physics Gate
"""


import torch
import torch.nn as nn
import torch.nn.functional as F  # Missing import!
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
import os
from datetime import datetime
from tqdm import tqdm
import time


class OptimizedRolloutBuffer:
    """
    优化的经验回放缓冲区
    
    改进:
    1. 预分配内存
    2. 支持批量添加
    3. 高效的GPU传输
    """
    
    def __init__(self, num_envs: int, rollout_steps: int, device: str = 'cuda'):
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.device = device
        self.ptr = 0
        
        # 预分配存储 (在CPU上，需要时传到GPU)
        self.states: List[Data] = []
        self.vertex_states_buffer: List[List[Dict]] = []  # ✅ BUG FIX 3a: Store vertex_states
        self.actions = {
            'action_type': np.zeros((rollout_steps, num_envs), dtype=np.int64),
            'vertex_idx': np.zeros((rollout_steps, num_envs), dtype=np.int64),
            'particle_type': np.zeros((rollout_steps, num_envs), dtype=np.int64),
            'target_vertex': np.zeros((rollout_steps, num_envs), dtype=np.int64),
        }
        self.rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        
    def add(
        self,
        states: List[Data],  # List of Data for each env
        actions: Dict[str, np.ndarray],  # {action_type: [num_envs], ...}
        rewards: np.ndarray,  # [num_envs]
        values: np.ndarray,   # [num_envs]
        log_probs: np.ndarray,  # [num_envs]
        dones: np.ndarray,    # [num_envs]
        vertex_states_list: Optional[List[List[Dict]]] = None  # ✅ BUG FIX 3a
    ):
        """批量添加一步的数据"""
        self.states.extend(states)  # 累积所有states
        
        # ✅ BUG FIX 3b: Store vertex_states if provided
        if vertex_states_list is not None:
            self.vertex_states_buffer.extend(vertex_states_list)
        
        step = self.ptr
        self.actions['action_type'][step] = actions['action_type']
        self.actions['vertex_idx'][step] = actions['vertex_idx']
        self.actions['particle_type'][step] = actions['particle_type']
        self.actions['target_vertex'][step] = actions['target_vertex']
        
        self.rewards[step] = rewards
        self.values[step] = values
        self.log_probs[step] = log_probs
        self.dones[step] = dones
        
        self.ptr += 1
    
    def get_tensors(self) -> Dict[str, torch.Tensor]:
        """获取所有数据作为GPU张量"""
        return {
            'actions': {
                k: torch.from_numpy(v[:self.ptr]).to(self.device)
                for k, v in self.actions.items()
            },
            'rewards': torch.from_numpy(self.rewards[:self.ptr]).to(self.device),
            'values': torch.from_numpy(self.values[:self.ptr]).to(self.device),
            'log_probs': torch.from_numpy(self.log_probs[:self.ptr]).to(self.device),
            'dones': torch.from_numpy(self.dones[:self.ptr]).to(self.device),
        }
    
    def get_batched_states(self) -> Batch:
        """获取batched states用于批量评估"""
        return Batch.from_data_list(self.states).to(self.device)
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.states.clear()
        self.vertex_states_buffer.clear()  # ✅ BUG FIX 3c
        # 不需要重置numpy数组，会被覆盖
    
    def __len__(self):
        return self.ptr * self.num_envs


class OptimizedPPOTrainer:
    """
    优化的PPO训练器
    
    关键优化:
    1. 批量前向传播 - 所有环境的state一次推理
    2. 批量action采样 - GPU上完成采样
    3. 异步环境执行 - 在GPU计算时并行执行环境step
    4. 减少同步点 - 最小化.item()调用
    """
    
    def __init__(
        self,
        parallel_env,  # ParallelEnvs 实例
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 2e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.3,
        max_grad_norm: float = 0.5,
        epochs_per_update: int = 6,
        mini_batch_size: int = 256,  # PPO mini-batch大小
        num_envs: int = 8,
        training_reactions: List[Dict] = None,  # BUG FIX 2: Multi-task training
    ):
        self.env = parallel_env
        self.model = model.to(device)
        self.device = device
        self.num_envs = num_envs
        
        # BUG FIX 2: Multi-task training setup
        from config import Config
        self.training_reactions = training_reactions or Config.get_training_reactions()
        self.current_reaction_idx = 0
        self.current_reaction = self.training_reactions[0]
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.epochs_per_update = epochs_per_update
        self.mini_batch_size = mini_batch_size
        
        # Optimizer with gradient scaling for mixed precision
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 可选: 混合精度训练
        self.scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
        self.use_amp = False  # 设为True启用混合精度
        
        # Logging
        self.writer = None
        self.global_step = 0
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -float('inf')
        self.reaction_success_counts = {r['name']: 0 for r in self.training_reactions}
        
    def _cycle_training_env(self):
        """
        BUG FIX 2 & 4: Cycle to next training reaction
        
        IMPORTANT: This should only be called BETWEEN rollouts, not during!
        We don't modify running environments - instead we just track which reaction to use for next reset.
        """
        self.current_reaction_idx = (self.current_reaction_idx + 1) % len(self.training_reactions)
        self.current_reaction = self.training_reactions[self.current_reaction_idx]
        
        print(f"  [Multi-task] Switched to reaction: {self.current_reaction['name']}")
        
    def _reset_env_with_current_reaction(self, env_idx: int):
        """
        Reset a single environment with current reaction
        
        MULTIPROCESSING COMPATIBLE: Uses IPC to set attributes
        """
        # Set initial_state through IPC (worker process will receive it)
        self.env.set_attr('initial_particles', self.current_reaction['initial'], indices=[env_idx])
        self.env.set_attr('final_particles', self.current_reaction['final'], indices=[env_idx])
        
        # Now reset
        return self.env.reset_single(env_idx)

    
    def collect_rollout_optimized(
        self, 
        rollout_steps: int,
    ) -> Tuple[OptimizedRolloutBuffer, Dict]:
        """
        优化的rollout收集
        
        关键: 批量GPU推理 + 并行环境执行
        
        Returns:
            buffer: 填充好的rollout buffer
            stats: 统计信息
        """
        buffer = OptimizedRolloutBuffer(self.num_envs, rollout_steps, self.device)
        
        # 初始化环境
        states, infos = self.env.reset()
        
        # 🚀 OPTIMIZATION: Get precomputed vertex_states from worker processes
        # Workers compute this in parallel during reset() - no main thread overhead!
        vertex_states_list = [info['vertex_states'] for info in infos]
        
        # 追踪episode
        episode_rewards_accum = np.zeros(self.num_envs)
        episode_lengths_accum = np.zeros(self.num_envs, dtype=np.int32)
        completed_episodes = []
        
        self.model.eval()
        
        # 🚀 DOUBLE BUFFERING: Prefetch first batch
        batched_states = Batch.from_data_list(states).to(self.device, non_blocking=True)
        
        for step in range(rollout_steps):
            # ===== ASYNC DOUBLE-BUFFER PIPELINE =====
            # Current batch is already on GPU from previous iteration!
            # We don't recreate it here - massive savings!
            
            with torch.no_grad():
                # GPU forward (using pre-loaded batch)
                # ✅ EARLY TERMINATION PENALTY: Pass infos to get step_counts
                outputs = self._batch_forward(batched_states, vertex_states_list, infos)
                
                # Sample actions on GPU
                actions_tensor, log_probs_tensor = self._batch_sample_actions(outputs)
                values_tensor = outputs['value'].squeeze(-1)
            
            # Transfer to CPU asynchronously
            actions_np = {
                'action_type': actions_tensor['action_type'].cpu().numpy(),
                'vertex_idx': actions_tensor['vertex_idx'].cpu().numpy(),
                'particle_type': actions_tensor['particle_type'].cpu().numpy(),
                'target_vertex': actions_tensor['target_vertex'].cpu().numpy(),
            }
            
            actions_list = [
                {k: int(v[i]) for k, v in actions_np.items()}
                for i in range(self.num_envs)
            ]
            
            # 🚀 Start workers (async, don't wait)
            self.env.step_async(actions_list)
            
            # While workers run, store data to buffer
            buffer.add(
                states=states,
                actions=actions_np,
                rewards=rewards_np if step > 0 else np.zeros(self.num_envs, dtype=np.float32),
                values=values_tensor.cpu().numpy(),
                log_probs=log_probs_tensor.cpu().numpy(),
                dones=dones_np if step > 0 else np.zeros(self.num_envs, dtype=np.float32),
                vertex_states_list=vertex_states_list  # ✅ BUG FIX 3c: Pass vertex_states
            )
            
            # Wait for workers
            next_states, rewards, terminateds, truncateds, infos = self.env.step_wait()
            vertex_states_list = [info['vertex_states'] for info in infos]
            
            rewards_np = np.array(rewards, dtype=np.float32)
            dones_np = np.array([t or tr for t, tr in zip(terminateds, truncateds)], dtype=np.float32)
            
            # 🚀 DOUBLE BUFFER: Prepare NEXT batch while doing other work
            # This overlaps with the loop overhead, so GPU never waits
            if step < rollout_steps - 1:  # Don't prepare batch we won't use
                batched_states = Batch.from_data_list(next_states).to(self.device, non_blocking=True)


            
            # ===== 6. 更新统计 =====
            episode_rewards_accum += rewards_np
            episode_lengths_accum += 1
            
            # 🚀 OPTIMIZED: Batch process done episodes
            done_indices = [i for i in range(self.num_envs) if dones_np[i]]
            
            if done_indices:
                # Collect episode stats
                for i in done_indices:
                    completed_episodes.append({
                        'reward': episode_rewards_accum[i],
                        'length': episode_lengths_accum[i],
                        'reaction': self.current_reaction['name']
                    })
                    episode_rewards_accum[i] = 0
                    episode_lengths_accum[i] = 0
                
                # 🚀 BATCH RESET: Parallel IPC instead of sequential
                # Old: Loop through each done env one-by-one (slow)
                # New: Send all reset commands, then collect all responses (fast)
                
                # Set attributes in parallel
                for idx in done_indices:
                    self.env.set_attr('initial_particles', self.current_reaction['initial'], indices=[idx])
                    self.env.set_attr('final_particles', self.current_reaction['final'], indices=[idx])
                
                # Reset all done envs in parallel
                for idx in done_indices:
                    self.env.remotes[idx].send(('reset', None))
                
                # Collect all results
                for idx in done_indices:
                    status, data = self.env.remotes[idx].recv()
                    if status == 'error':
                        raise RuntimeError(f"Reset failed for env {idx}: {data}")
                    state, info = data
                    next_states[idx] = state
                    vertex_states_list[idx] = info['vertex_states']
            
            states = next_states


        # ✅ BUG FIX 4a: Compute last_values for GAE bootstrap
        # Don't use zeros - use actual value estimates for non-terminal states!
        with torch.no_grad():
            last_batch = Batch.from_data_list(states).to(self.device, non_blocking=True)
            # ✅ EARLY TERMINATION PENALTY: Pass infos for last value computation too
            last_outputs = self._batch_forward(last_batch, vertex_states_list, infos)
            last_values = last_outputs['value'].squeeze(-1).cpu().numpy()
        
        # 统计
        stats = {
            'num_episodes': len(completed_episodes),
            'mean_reward': np.mean([e['reward'] for e in completed_episodes]) if completed_episodes else 0,
            'mean_length': np.mean([e['length'] for e in completed_episodes]) if completed_episodes else 0,
            'total_steps': rollout_steps * self.num_envs,
            'last_values': last_values  # ✅ Pass to compute_gae
        }
        
        # BUG FIX 4: Cycle reaction AFTER rollout completes (not during!)
        # Strategy: Cycle every N completed episodes to ensure balanced training
        episodes_per_reaction = 50  # Adjust based on your needs
        self.total_episodes_count = getattr(self, 'total_episodes_count', 0) + len(completed_episodes)
        
        if self.total_episodes_count >= episodes_per_reaction:
            self._cycle_training_env()
            self.total_episodes_count = 0  # Reset counter
        
        return buffer, stats
    
    def _batch_forward(self, batched_states: Batch, vertex_states_list: Optional[List[List[Dict]]] = None, infos: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        """
        🚀 OPTIMIZED: 批量前向传播
        
        KEY OPTIMIZATION: Uses policy_head.forward_batch() for GPU-parallel processing
        
        Args:
            batched_states: PyG Batch对象 [total_nodes across all graphs]
            vertex_states_list: List of vertex states for each environment (for Physics Gate)
            infos: List of info dicts from environments (contains step_count for early termination penalty)
            
        Returns:
            outputs with policy probabilities for each env
        """
        # 获取batch信息
        batch_size = batched_states.num_graphs
        
        # Step 1: Batch MPNN encoding (efficient)
        if self.use_amp and self.device == 'cuda':
            with torch.amp.autocast('cuda'):
                node_embeddings, graph_embeddings = self.model.encoder(batched_states)
        else:
            node_embeddings, graph_embeddings = self.model.encoder(batched_states)
        
        # Step 2: 🚀 OPTIMIZED - Batch process policy head (NEW!)
        # Old: Loop through each env sequentially (64 GPU kernel launches!)
        # New: Single batched call (1 GPU kernel launch!)
        
        if batch_size > 1:
            # Ensure graph_embeddings is [batch_size, hidden_dim]
            if graph_embeddings.dim() == 1:
                graph_embeddings = graph_embeddings.unsqueeze(0).expand(batch_size, -1)
            
            # ✅ EARLY TERMINATION PENALTY: Extract step_counts from infos
            step_counts = None
            if infos is not None:
                step_counts = torch.tensor(
                    [info['step_count'] for info in infos],
                    dtype=torch.long,
                    device=self.device
                )
            
            # 🚀 KEY OPTIMIZATION: Single batched forward pass!
            # This replaces the sequential for-loop with a batched GPU operation
            policy_output = self.model.policy_head.forward_batch(
                graph_embeddings,
                vertex_states_list=vertex_states_list,
                apply_physics_gate=(vertex_states_list is not None),
                step_counts=step_counts  # ✅ Pass step_counts for early termination penalty
            )
            
            # Batch compute values
            values_batch = self.model.value_head(graph_embeddings)
            
            # Combine outputs
            output = {
                'action_type_probs': policy_output['action_type_probs'],
                'vertex_probs': policy_output['vertex_probs'],
                'particle_probs': policy_output['particle_probs'],
                'value': values_batch,
                'node_embeddings': node_embeddings,
                'graph_embedding': graph_embeddings
            }
            
            # Include gate values if computed
            if 'gate_values' in policy_output:
                output['gate_values'] = policy_output['gate_values']
        else:
            # Single environment - use regular forward
            vertex_states = vertex_states_list[0] if vertex_states_list else None
            step_count = infos[0]['step_count'] if infos else 0
            raw_output = self.model(batched_states, vertex_states=vertex_states, return_value=True, step_count=step_count)
            output = raw_output
        
        return output


    
    def _batch_sample_actions(
        self, 
        outputs: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        在GPU上批量采样actions
        
        Returns:
            actions: Dict of action tensors [num_envs]
            log_probs: Log probabilities [num_envs]
        """
        # 采样 (在GPU上)
        action_type = torch.multinomial(outputs['action_type_probs'], 1).squeeze(-1)
        vertex_idx = torch.multinomial(outputs['vertex_probs'], 1).squeeze(-1)
        particle_type = torch.multinomial(outputs['particle_probs'], 1).squeeze(-1)
        target_vertex = torch.multinomial(outputs['vertex_probs'], 1).squeeze(-1)
        
        actions = {
            'action_type': action_type,
            'vertex_idx': vertex_idx,
            'particle_type': particle_type,
            'target_vertex': target_vertex,
        }
        
        # 计算log probs (批量)
        batch_size = action_type.shape[0] if action_type.dim() > 0 else 1
        
        # 使用gather获取选中action的概率
        if batch_size > 1:
            action_type_log_prob = torch.log(
                outputs['action_type_probs'].gather(1, action_type.unsqueeze(-1)).squeeze(-1) + 1e-8
            )
            vertex_log_prob = torch.log(
                outputs['vertex_probs'].gather(1, vertex_idx.unsqueeze(-1)).squeeze(-1) + 1e-8
            )
            particle_log_prob = torch.log(
                outputs['particle_probs'].gather(1, particle_type.unsqueeze(-1)).squeeze(-1) + 1e-8
            )
        else:
            action_type_log_prob = torch.log(outputs['action_type_probs'][action_type] + 1e-8)
            vertex_log_prob = torch.log(outputs['vertex_probs'][vertex_idx] + 1e-8)
            particle_log_prob = torch.log(outputs['particle_probs'][particle_type] + 1e-8)
        
        total_log_prob = action_type_log_prob + vertex_log_prob + particle_log_prob
        
        return actions, total_log_prob
    
    def compute_gae(
        self,
        rewards: torch.Tensor,  # [steps, num_envs]
        values: torch.Tensor,   # [steps, num_envs]
        dones: torch.Tensor,    # [steps, num_envs]
        last_values: torch.Tensor  # ✅ BUG FIX 4b: Bootstrap values [num_envs]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GAE (Generalized Advantage Estimation)
        
        在GPU上计算
        """
        steps, num_envs = rewards.shape
        advantages = torch.zeros_like(rewards)
        
        gae = torch.zeros(num_envs, device=self.device)
        
        for t in reversed(range(steps)):
            if t == steps - 1:
                # ✅ BUG FIX 4c: Use last_values for bootstrap, not zeros!
                # This correctly handles episodes that are truncated mid-rollout
                next_value = last_values * (1.0 - dones[t])
            else:
                next_value = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update_policy(self, buffer: OptimizedRolloutBuffer, last_values: np.ndarray) -> Dict[str, float]:
        """
        PPO策略更新
        
        使用mini-batch SGD
        
        Args:
            buffer: Rollout buffer with experiences
            last_values: Bootstrap values for GAE [num_envs]  # ✅ BUG FIX 4d
        """
        self.model.train()
        
        # 获取数据
        data = buffer.get_tensors()
        batched_states = buffer.get_batched_states()
        
        # ✅ BUG FIX 4d: Pass last_values to compute_gae
        last_values_tensor = torch.from_numpy(last_values).to(self.device)
        
        # 计算advantages
        advantages, returns = self.compute_gae(
            data['rewards'],
            data['values'],
            data['dones'],
            last_values_tensor
        )
        
        # Flatten for mini-batch training
        # [steps, num_envs] -> [steps * num_envs]
        total_samples = buffer.ptr * self.num_envs
        
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        old_log_probs_flat = data['log_probs'].reshape(-1)
        
        actions_flat = {
            k: v.reshape(-1) for k, v in data['actions'].items()
        }
        
        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        # PPO epochs
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0
        entropy_total = 0
        
        indices = np.arange(total_samples)
        
        for epoch in range(self.epochs_per_update):
            np.random.shuffle(indices)
            
            for start in range(0, total_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, total_samples)
                mb_indices = indices[start:end]
                
                # 获取mini-batch数据
                mb_advantages = advantages_flat[mb_indices]
                mb_returns = returns_flat[mb_indices]
                mb_old_log_probs = old_log_probs_flat[mb_indices]
                mb_actions = {k: v[mb_indices] for k, v in actions_flat.items()}
                
                # 获取对应的states (需要从buffer中索引)
                mb_states = [buffer.states[i] for i in mb_indices]
                mb_batch = Batch.from_data_list(mb_states).to(self.device)
                
                # ✅ BUG FIX 3d: Get corresponding vertex_states for Physics Gate
                mb_vertex_states = None
                if len(buffer.vertex_states_buffer) > 0:
                    mb_vertex_states = [buffer.vertex_states_buffer[i] for i in mb_indices]
                
                # ✅ BUG FIX 3e: Use batched forward for consistency with rollout
                # Issue: model.forward() expects vertex_states: List[Dict] (single env)
                #        but mb_vertex_states is List[List[Dict]] (multiple envs)
                # Solution: Use batched encoder + batched policy_head like in rollout
                if mb_vertex_states is not None and len(mb_states) > 1:
                    # Batched processing with Physics Gate
                    node_embeddings, graph_embeddings = self.model.encoder(mb_batch)
                    
                    # Ensure correct dimensions
                    if graph_embeddings.dim() == 1:
                        graph_embeddings = graph_embeddings.unsqueeze(0).expand(len(mb_states), -1)
                    
                    # Batched policy head with Physics Gate
                    policy_output = self.model.policy_head.forward_batch(
                        graph_embeddings,
                        vertex_states_list=mb_vertex_states,
                        apply_physics_gate=True
                    )
                    
                    # Batched value head
                    values_batch = self.model.value_head(graph_embeddings)
                    
                    # Combine outputs
                    outputs = {
                        'action_type_probs': policy_output['action_type_probs'],
                        'vertex_probs': policy_output['vertex_probs'],
                        'particle_probs': policy_output['particle_probs'],
                        'value': values_batch
                    }
                else:
                    # Single sample or no vertex_states: use standard forward
                    # Note: vertex_states should be List[Dict] for single env
                    vs = mb_vertex_states[0] if mb_vertex_states and len(mb_vertex_states) > 0 else None
                    outputs = self.model(mb_batch, vertex_states=vs, return_value=True)
                
                # 计算新的log probs (简化版，需要根据实际模型调整)
                # 这里假设outputs包含每个样本的概率分布
                new_log_probs = self._compute_log_probs(outputs, mb_actions, len(mb_indices))
                values = outputs['value'].squeeze(-1)
                
                # 计算entropy
                entropy = self._compute_entropy(outputs, len(mb_indices))
                
                # PPO目标
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, mb_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                entropy_total += entropy.item()
        
        num_updates = self.epochs_per_update * (total_samples // self.mini_batch_size + 1)
        
        return {
            'total_loss': total_loss / num_updates,
            'policy_loss': policy_loss_total / num_updates,
            'value_loss': value_loss_total / num_updates,
            'entropy': entropy_total / num_updates,
        }
    
    def _compute_log_probs(
        self, 
        outputs: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        batch_size: int
    ) -> torch.Tensor:
        """计算给定actions的log概率"""
        # 处理可能的维度问题
        if outputs['action_type_probs'].dim() == 1:
            # 单样本情况
            action_type_log_prob = torch.log(outputs['action_type_probs'][actions['action_type']] + 1e-8)
            vertex_log_prob = torch.log(outputs['vertex_probs'][actions['vertex_idx']] + 1e-8)
            particle_log_prob = torch.log(outputs['particle_probs'][actions['particle_type']] + 1e-8)
        else:
            # Batch情况
            action_type_log_prob = torch.log(
                outputs['action_type_probs'].gather(1, actions['action_type'].unsqueeze(-1)).squeeze(-1) + 1e-8
            )
            vertex_log_prob = torch.log(
                outputs['vertex_probs'].gather(1, actions['vertex_idx'].unsqueeze(-1)).squeeze(-1) + 1e-8
            )
            particle_log_prob = torch.log(
                outputs['particle_probs'].gather(1, actions['particle_type'].unsqueeze(-1)).squeeze(-1) + 1e-8
            )
        
        return action_type_log_prob + vertex_log_prob + particle_log_prob
    
    def _compute_entropy(self, outputs: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        """计算策略entropy"""
        def safe_entropy(probs):
            return -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        action_entropy = safe_entropy(outputs['action_type_probs'])
        vertex_entropy = safe_entropy(outputs['vertex_probs'])
        particle_entropy = safe_entropy(outputs['particle_probs'])
        
        return action_entropy + vertex_entropy + particle_entropy
    
    def train(
        self,
        total_timesteps: int,
        rollout_steps: int = 1024,
        log_interval: int = 10,
        save_interval: int = 10000,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
    ):
        """
        主训练循环
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        
        num_updates = total_timesteps // (rollout_steps * self.num_envs)
        
        # BUG FIX 5: Validate training configuration
        if num_updates == 0:
            steps_per_update = rollout_steps * self.num_envs
            print(f"\n{'='*60}")
            print(f"❌ ERROR: Invalid training configuration!")
            print(f"{'='*60}")
            print(f"  Total timesteps: {total_timesteps:,}")
            print(f"  Steps per update: {steps_per_update:,}")
            print(f"  Result: num_updates = {num_updates} (需要至少1个update!)")
            print(f"\n解决方案:")
            print(f"  1. 减少环境数: --num-envs {total_timesteps // (rollout_steps * 2)}")
            print(f"  2. 增加总步数: --steps {steps_per_update * 10:,}")
            print(f"  3. 减少rollout steps: 在run_experiment.py中改为128")
            print(f"{'='*60}\n")
            raise ValueError(f"num_updates = 0! Steps per update ({steps_per_update:,}) exceeds total timesteps ({total_timesteps:,})")
        
        print(f"\n{'='*60}")
        print(f"Starting Optimized PPO Training")
        print(f"{'='*60}")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Rollout steps: {rollout_steps}")
        print(f"  Num envs: {self.num_envs}")
        print(f"  Steps per update: {rollout_steps * self.num_envs:,}")
        print(f"  Total updates: {num_updates:,}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        with tqdm(total=total_timesteps, desc="Training") as pbar:
            for update in range(num_updates):
                # Collect rollout
                t0 = time.time()
                buffer, rollout_stats = self.collect_rollout_optimized(rollout_steps)
                rollout_time = time.time() - t0
                
                # Update policy
                t0 = time.time()
                # ✅ BUG FIX 4e: Pass last_values from rollout_stats
                update_stats = self.update_policy(buffer, rollout_stats['last_values'])
                update_time = time.time() - t0
                
                # Update global step
                steps_this_update = rollout_steps * self.num_envs
                self.global_step += steps_this_update
                
                # Logging
                if update % log_interval == 0:
                    elapsed = time.time() - start_time
                    fps = self.global_step / elapsed
                    
                    # Log to tensorboard
                    self.writer.add_scalar('rollout/mean_reward', rollout_stats['mean_reward'], self.global_step)
                    self.writer.add_scalar('rollout/mean_length', rollout_stats['mean_length'], self.global_step)
                    self.writer.add_scalar('rollout/num_episodes', rollout_stats['num_episodes'], self.global_step)
                    
                    self.writer.add_scalar('train/policy_loss', update_stats['policy_loss'], self.global_step)
                    self.writer.add_scalar('train/value_loss', update_stats['value_loss'], self.global_step)
                    self.writer.add_scalar('train/entropy', update_stats['entropy'], self.global_step)
                    
                    self.writer.add_scalar('time/fps', fps, self.global_step)
                    self.writer.add_scalar('time/rollout_time', rollout_time, self.global_step)
                    self.writer.add_scalar('time/update_time', update_time, self.global_step)
                    
                    # BUG FIX 2: Log current reaction
                    self.writer.add_text('multi_task/current_reaction', self.current_reaction['name'], self.global_step)
                    
                    print(f"\nUpdate {update}/{num_updates}")
                    print(f"  Steps: {self.global_step:,} | FPS: {fps:.0f}")
                    print(f"  Reaction: {self.current_reaction['name']}")  # BUG FIX 2: Show current reaction
                    print(f"  Reward: {rollout_stats['mean_reward']:.2f} | Length: {rollout_stats['mean_length']:.1f}")
                    print(f"  Loss: {update_stats['total_loss']:.4f} | Entropy: {update_stats['entropy']:.4f}")
                    print(f"  Time: rollout={rollout_time:.2f}s, update={update_time:.2f}s")
                
                # Save checkpoint
                if self.global_step % save_interval < steps_this_update:
                    ckpt_path = os.path.join(checkpoint_dir, f'model_{self.global_step}.pt')
                    self.save_checkpoint(ckpt_path)
                    print(f"  Saved checkpoint: {ckpt_path}")
                
                # Track best
                if rollout_stats['mean_reward'] > self.best_reward:
                    self.best_reward = rollout_stats['mean_reward']
                
                # Clear buffer
                buffer.clear()
                
                pbar.update(steps_this_update)
        
        # Final save
        final_path = os.path.join(checkpoint_dir, 'model_final.pt')
        self.save_checkpoint(final_path)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Total steps: {self.global_step:,}")
        print(f"  Best reward: {self.best_reward:.2f}")
        print(f"  Time: {time.time() - start_time:.1f}s")
        print(f"{'='*60}")
        
        self.writer.close()
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_reward': self.best_reward,
        }, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_reward = checkpoint.get('best_reward', -float('inf'))
        print(f"Loaded checkpoint from {path}")
