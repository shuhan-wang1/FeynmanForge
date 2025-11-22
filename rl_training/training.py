"""
PPO Training Loop for Feynman-GCPN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
from tqdm import tqdm
from torch_geometric.data import Batch

from feynman_env import FeynmanDiagramEnv
from models import FeynmanGCPN
from physics_engine import PhysicsConstants


class RolloutBuffer:
    """
    Storage for trajectories collected during rollouts
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.vertex_states = []
    
    def add(
        self,
        state,
        action: Dict,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        vertex_state: List[Dict]
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.vertex_states.append(vertex_state)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.vertex_states.clear()
    
    def get(self):
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'values': self.values,
            'log_probs': self.log_probs,
            'dones': self.dones,
            'vertex_states': self.vertex_states
        }
    
    def __len__(self):
        return len(self.rewards)


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for Feynman-GCPN
    """
    
    def __init__(
        self,
        env,  # 可以是单个环境或ParallelEnvs
        model: FeynmanGCPN,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        epochs_per_update: int = 4,
        batch_size: int = 64,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        num_envs: int = 1,  # 并行环境数量
        use_amp: bool = True  # Enable mixed precision training
    ):
        self.env = env
        self.model = model.to(device)
        self.device = device
        self.num_envs = num_envs
        self.is_parallel = num_envs > 1
        self.use_amp = use_amp and device.type == 'cuda'

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None

        # Buffer
        self.buffer = RolloutBuffer()

        # Logging
        self.writer = None
        self.global_step = 0

        # Best model tracking
        self.best_reward = -float('inf')
        self.best_diagram = None
        self.vis_env = None  # 用于可视化的环境引用

        # Save initial diagram immediately for visualization
        os.makedirs('diagrams', exist_ok=True)
        self._save_current_diagram()

        if self.use_amp:
            print("🚀 Mixed precision training (FP16) enabled!")
    
    def collect_rollout(self, num_steps: int, deterministic: bool = False) -> Dict:
        """
        Collect a rollout of experiences
        支持单环境和并行环境
        
        Args:
            num_steps: Number of steps to collect
            deterministic: Whether to use deterministic policy
            
        Returns:
            Statistics dictionary
        """
        if self.is_parallel:
            return self._collect_rollout_parallel(num_steps, deterministic)
        else:
            return self._collect_rollout_single(num_steps, deterministic)
    
    def _collect_rollout_single(self, num_steps: int, deterministic: bool = False) -> Dict:
        """单环境收集rollout"""
        episode_rewards = []
        episode_lengths = []
        
        state, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_steps):
            # Move state to device (use non_blocking for better GPU utilization)
            state_device = state.to(self.device, non_blocking=True)
            
            # Get vertex states for physics gating
            vertex_states = self._extract_vertex_states()
            
            # Get action and value
            with torch.no_grad():
                output = self.model(state_device, vertex_states, return_value=True)

                if deterministic:
                    action_type = output['action_type_probs'].argmax().item()
                    vertex_idx = output['vertex_probs'].argmax().item()
                    particle_type = output['particle_probs'].argmax().item()

                    # Mask out vertex_idx when selecting target_vertex
                    target_probs = output['vertex_probs'].clone()
                    target_probs[vertex_idx] = 0
                    if target_probs.sum() > 0:
                        target_probs = target_probs / target_probs.sum()
                        target_vertex = target_probs.argmax().item()
                    else:
                        target_vertex = 0

                    action = {
                        'action_type': action_type,
                        'vertex_idx': vertex_idx,
                        'particle_type': particle_type,
                        'target_vertex': target_vertex
                    }

                    # Log prob calculation
                    target_vertex_log_prob = torch.log(target_probs[target_vertex] + 1e-8)
                else:
                    action_type = torch.multinomial(output['action_type_probs'], 1).item()
                    vertex_idx = torch.multinomial(output['vertex_probs'], 1).item()
                    particle_type = torch.multinomial(output['particle_probs'], 1).item()

                    # CRITICAL FIX: Mask out vertex_idx when sampling target_vertex
                    # Prevents MERGE(vertex, vertex) which always fails
                    target_probs = output['vertex_probs'].clone()
                    target_probs[vertex_idx] = 0
                    if target_probs.sum() > 0:
                        target_probs = target_probs / target_probs.sum()
                        target_vertex = torch.multinomial(target_probs, 1).item()
                    else:
                        target_vertex = (vertex_idx + 1) % len(target_probs)

                    action = {
                        'action_type': action_type,
                        'vertex_idx': vertex_idx,
                        'particle_type': particle_type,
                        'target_vertex': target_vertex
                    }

                    # Log prob calculation with masked target_vertex distribution
                    target_vertex_log_prob = torch.log(target_probs[target_vertex] + 1e-8)

                value = output['value'].item()

                # Compute log prob (using masked target_probs for target_vertex)
                action_type_log_prob = torch.log(output['action_type_probs'][action['action_type']] + 1e-8)
                vertex_log_prob = torch.log(output['vertex_probs'][action['vertex_idx']] + 1e-8)
                particle_log_prob = torch.log(output['particle_probs'][action['particle_type']] + 1e-8)
                log_prob = (action_type_log_prob + vertex_log_prob + particle_log_prob + target_vertex_log_prob).item()
            
            # 瓶颈在这里：env.step() 在CPU上串行执行
            # 这是RL的固有限制，环境必须串行执行
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.buffer.add(state, action, reward, value, log_prob, done, vertex_states)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Check if this is the best diagram so far
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    env_ref = self.vis_env if self.vis_env else (self.env.envs[0] if hasattr(self.env, 'envs') else self.env)
                    self.best_diagram = env_ref.get_diagram_json()
                    # Immediately save best diagram for visualization
                    self._save_best_diagram()
                
                # Also save current diagram periodically for live monitoring
                if len(episode_rewards) % 1 == 0:  # 每个episode都保存
                    self._save_current_diagram()
                
                # Reset
                state, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                state = next_state
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0
        }
    
    def _collect_rollout_parallel(self, num_steps: int, deterministic: bool = False) -> Dict:
        """并行环境收集rollout - 真正的批量GPU处理"""
        episode_rewards = []
        episode_lengths = []

        # DEBUG: Track action type distribution
        action_type_counts = [0, 0, 0, 0, 0]  # [CONNECT, BRANCH, SET_TYPE, TERMINATE, MERGE]
        termination_reasons = {'terminated': 0, 'truncated': 0}  # Track why episodes end

        # 重置所有环境
        states, infos = self.env.reset()
        episode_rewards_per_env = [0] * self.num_envs
        episode_lengths_per_env = [0] * self.num_envs

        steps_per_env = num_steps // self.num_envs

        for step in range(steps_per_env):
            # ===== 关键优化：批量GPU处理 =====
            # 1. 批量提取vertex_states
            vertex_states_batch = [
                self._extract_vertex_states_from_env(self.env.envs[env_idx])
                for env_idx in range(self.num_envs)
            ]

            # 2. **NEW: Batch all graphs together for parallel GPU processing**
            actions = []
            values = []
            log_probs = []

            with torch.no_grad():
                # Batch all graph states together using PyG Batch
                try:
                    batched_state = Batch.from_data_list(states).to(self.device, non_blocking=True)

                    # Process entire batch at once (MAJOR SPEEDUP)
                    with autocast('cuda', enabled=self.use_amp):
                        outputs = self._process_batched_states(batched_state, vertex_states_batch)

                    # Unbatch results
                    for env_idx in range(self.num_envs):
                        output = {
                            'action_type_probs': outputs['action_type_probs'][env_idx],
                            'vertex_probs': outputs['vertex_probs'][env_idx],
                            'particle_probs': outputs['particle_probs'][env_idx],
                            'value': outputs['values'][env_idx]
                        }

                        if deterministic:
                            action = {
                                'action_type': output['action_type_probs'].argmax().item(),
                                'vertex_idx': output['vertex_probs'].argmax().item(),
                                'particle_type': output['particle_probs'].argmax().item(),
                                'target_vertex': output['vertex_probs'].argmax().item()
                            }
                        else:
                            action = {
                                'action_type': torch.multinomial(output['action_type_probs'], 1).item(),
                                'vertex_idx': torch.multinomial(output['vertex_probs'], 1).item(),
                                'particle_type': torch.multinomial(output['particle_probs'], 1).item(),
                                'target_vertex': torch.multinomial(output['vertex_probs'], 1).item()
                            }

                        value = output['value'].item()

                        # DEBUG: Print action distribution for first environment periodically
                        if env_idx == 0 and step % 100 == 0 and self.global_step < 5000:
                            action_names = ['CONNECT', 'BRANCH', 'SET_TYPE', 'TERMINATE', 'MERGE']
                            probs = output['action_type_probs']
                            print(f"\n[DEBUG Global Step {self.global_step}, Rollout Step {step}]")
                            print(f"Action probs: {' '.join([f'{action_names[i]}={probs[i]:.3f}' for i in range(5)])}")
                            print(f"  ⚠️  TERMINATE prob: {probs[3]:.4f} (should be < 0.01 with -5.0 bias)")
                            print(f"Chosen: {action_names[action['action_type']]} | Value: {value:.3f}")

                        action_type_log_prob = torch.log(output['action_type_probs'][action['action_type']] + 1e-8)
                        vertex_log_prob = torch.log(output['vertex_probs'][action['vertex_idx']] + 1e-8)
                        particle_log_prob = torch.log(output['particle_probs'][action['particle_type']] + 1e-8)
                        log_prob = (action_type_log_prob + vertex_log_prob + particle_log_prob).item()

                        # DEBUG: Track action types
                        action_type_counts[action['action_type']] += 1

                        actions.append(action)
                        values.append(value)
                        log_probs.append(log_prob)

                except Exception as e:
                    # Fallback to sequential processing if batching fails
                    print(f"⚠️  Batch processing failed, falling back to sequential: {e}")
                    for env_idx in range(self.num_envs):
                        state_device = states[env_idx].to(self.device, non_blocking=True)
                        output = self.model(state_device, vertex_states_batch[env_idx], return_value=True)

                        if deterministic:
                            action = {
                                'action_type': output['action_type_probs'].argmax().item(),
                                'vertex_idx': output['vertex_probs'].argmax().item(),
                                'particle_type': output['particle_probs'].argmax().item(),
                                'target_vertex': output['vertex_probs'].argmax().item()
                            }
                        else:
                            action = {
                                'action_type': torch.multinomial(output['action_type_probs'], 1).item(),
                                'vertex_idx': torch.multinomial(output['vertex_probs'], 1).item(),
                                'particle_type': torch.multinomial(output['particle_probs'], 1).item(),
                                'target_vertex': torch.multinomial(output['vertex_probs'], 1).item()
                            }

                        value = output['value'].item()

                        action_type_log_prob = torch.log(output['action_type_probs'][action['action_type']] + 1e-8)
                        vertex_log_prob = torch.log(output['vertex_probs'][action['vertex_idx']] + 1e-8)
                        particle_log_prob = torch.log(output['particle_probs'][action['particle_type']] + 1e-8)
                        log_prob = (action_type_log_prob + vertex_log_prob + particle_log_prob).item()

                        # DEBUG: Track action types
                        action_type_counts[action['action_type']] += 1

                        actions.append(action)
                        values.append(value)
                        log_probs.append(log_prob)
            
            # 4. 批量存储经验
            for env_idx in range(self.num_envs):
                self.buffer.add(states[env_idx], actions[env_idx], 0, values[env_idx], 
                              log_probs[env_idx], False, vertex_states_batch[env_idx])
            
            # 5. 并行执行所有环境的step（这里是真正的多核并行）
            next_states, rewards, terminateds, truncateds, infos = self.env.step(actions)
            
            # 更新buffer中的reward并处理环境重置
            for env_idx in range(self.num_envs):
                if len(self.buffer.rewards) > 0:
                    self.buffer.rewards[-(self.num_envs - env_idx)] = rewards[env_idx]
                
                episode_rewards_per_env[env_idx] += rewards[env_idx]
                episode_lengths_per_env[env_idx] += 1
                
                done = terminateds[env_idx] or truncateds[env_idx]
                if done:
                    # DEBUG: Track termination reason
                    if terminateds[env_idx]:
                        termination_reasons['terminated'] += 1
                    if truncateds[env_idx]:
                        termination_reasons['truncated'] += 1

                    current_reward = episode_rewards_per_env[env_idx]
                    episode_rewards.append(current_reward)
                    episode_lengths.append(episode_lengths_per_env[env_idx])
                    
                    # 检查并更新最佳奖励
                    if current_reward > self.best_reward:
                        self.best_reward = current_reward
                        # 从对应的环境中获取最佳图结构
                        try:
                            self.best_diagram = self.env.envs[env_idx].get_diagram_json()
                            # 立即保存最佳图
                            self._save_best_diagram()
                        except Exception as e:
                            # 如果无法访问，至少更新数值
                            pass
                    
                    # 定期更新可视化 (只用第0个环境的数据)
                    if env_idx == 0:
                        try:
                            self._save_current_diagram()
                        except:
                            pass
                    
                    episode_rewards_per_env[env_idx] = 0
                    episode_lengths_per_env[env_idx] = 0
                    
                    # 重置已完成的环境并更新状态
                    reset_state, _ = self.env.envs[env_idx].reset()
                    next_states[env_idx] = reset_state
            
            states = next_states

        # DEBUG: Print action type distribution
        if self.global_step < 1000 or self.global_step % 500 == 0:
            action_names = ['CONNECT', 'BRANCH', 'SET_TYPE', 'TERMINATE', 'MERGE']
            total_actions = sum(action_type_counts)
            print(f"\n{'='*80}")
            print(f"[ROLLOUT DEBUG] Global Step {self.global_step}")
            print(f"{'='*80}")
            print(f"Action Type Distribution (total={total_actions}):")
            for i, name in enumerate(action_names):
                count = action_type_counts[i]
                pct = 100.0 * count / total_actions if total_actions > 0 else 0
                marker = " ⚠️  TOO HIGH!" if i == 3 and pct > 2.0 else ""  # Warn if TERMINATE > 2%
                print(f"  {name:12s}: {count:5d} ({pct:5.1f}%){marker}")
            print(f"\nEpisode Terminations:")
            print(f"  TERMINATED (chose TERMINATE action): {termination_reasons['terminated']}")
            print(f"  TRUNCATED (hit max_steps):            {termination_reasons['truncated']}")
            term_rate = 100.0 * termination_reasons['terminated'] / (termination_reasons['terminated'] + termination_reasons['truncated']) if (termination_reasons['terminated'] + termination_reasons['truncated']) > 0 else 0
            print(f"  TERMINATE rate: {term_rate:.1f}% (should be < 10%)")
            print(f"\nEpisode Stats:")
            print(f"  Completed episodes: {len(episode_lengths)}")
            print(f"  Mean length: {np.mean(episode_lengths) if episode_lengths else 0:.2f} (target: 15-30)")
            print(f"  Mean reward: {np.mean(episode_rewards) if episode_rewards else 0:.2f}")
            print(f"{'='*80}\n")

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0
        }
    
    def _process_batched_states(self, batched_state, vertex_states_batch):
        """
        Process a batch of graph states in parallel on GPU

        Args:
            batched_state: PyG Batch object containing all graphs
            vertex_states_batch: List of vertex states for each graph

        Returns:
            Dictionary with batched outputs
        """
        # Process the batched graphs through the model
        # This needs to handle the batched nature properly
        batch_size = len(vertex_states_batch)

        # We'll need to process each graph separately but on GPU in parallel
        # Since vertex_states structure is complex, we process sequentially but keep on GPU
        action_type_probs_list = []
        vertex_probs_list = []
        particle_probs_list = []
        values_list = []

        # Split batched graph back into individual graphs
        # PyG Batch stores the batch assignment in batched_state.batch
        graphs = batched_state.to_data_list()

        for i, (graph, vertex_states) in enumerate(zip(graphs, vertex_states_batch)):
            output = self.model(graph, vertex_states, return_value=True)
            action_type_probs_list.append(output['action_type_probs'])
            vertex_probs_list.append(output['vertex_probs'])
            particle_probs_list.append(output['particle_probs'])
            values_list.append(output['value'])

        # Stack results
        return {
            'action_type_probs': action_type_probs_list,
            'vertex_probs': vertex_probs_list,
            'particle_probs': particle_probs_list,
            'values': values_list
        }

    def _extract_vertex_states_from_env(self, env):
        """从指定环境提取vertex states"""
        vertex_states = []
        for vertex in env.vertices:
            connected_edges = [env.edges[eid] for eid in vertex['connected_edges']]
            incoming = [e for e in connected_edges if e['target'] == vertex['id']]
            outgoing = [e for e in connected_edges if e['source'] == vertex['id']]

            vertex_states.append({
                'incoming': incoming,
                'outgoing': outgoing,
                'position': (vertex['x'], vertex['y'])
            })
        return vertex_states
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation
        
        Returns:
            advantages: [num_steps]
            returns: [num_steps]
        """
        advantages = np.zeros(len(rewards))
        returns = np.zeros(len(rewards))
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update_policy(self) -> Dict:
        """
        Update policy using PPO
        
        Returns:
            Dictionary with loss statistics
        """
        data = self.buffer.get()
        
        if len(data['rewards']) == 0:
            return {}
        
        # Compute advantages
        advantages, returns = self.compute_gae(
            data['rewards'],
            data['values'],
            data['dones']
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        old_log_probs = torch.tensor(data['log_probs'], dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        # Mini-batch updates
        indices = np.arange(len(data['rewards']))
        
        for epoch in range(self.epochs_per_update):
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                # Batch data (use non_blocking for GPU transfer)
                batch_states = [data['states'][i].to(self.device, non_blocking=True) for i in batch_idx]
                batch_actions = [data['actions'][i] for i in batch_idx]
                batch_vertex_states = [data['vertex_states'][i] for i in batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx].to(self.device, non_blocking=True)
                batch_advantages = advantages_tensor[batch_idx].to(self.device, non_blocking=True)
                batch_returns = returns_tensor[batch_idx].to(self.device, non_blocking=True)
                
                # 批量评估动作（并行处理整个batch）
                # 预分配tensor以减少内存分配开销
                batch_size_actual = len(batch_idx)
                batch_log_probs = torch.zeros(batch_size_actual, device=self.device)
                batch_values = torch.zeros(batch_size_actual, device=self.device)
                batch_entropies = torch.zeros(batch_size_actual, device=self.device)

                # Use mixed precision for forward pass
                with autocast('cuda', enabled=self.use_amp):
                    for i, (state, action, vertex_state) in enumerate(zip(batch_states, batch_actions, batch_vertex_states)):
                        action_tensor = {k: torch.tensor(v, device=self.device) for k, v in action.items()}
                        log_prob, value, entropy = self.model.evaluate_actions(state, action_tensor, vertex_state)
                        batch_log_probs[i] = log_prob
                        batch_values[i] = value
                        batch_entropies[i] = entropy

                    # Compute PPO loss
                    ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = 0.5 * ((batch_values - batch_returns) ** 2).mean()

                    # Entropy bonus
                    entropy_loss = -batch_entropies.mean()

                    # Total loss
                    loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize with mixed precision
                self.optimizer.zero_grad(set_to_none=True)  # set_to_none=True更快

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / num_updates if num_updates > 0 else 0,
            'value_loss': total_value_loss / num_updates if num_updates > 0 else 0,
            'entropy': total_entropy / num_updates if num_updates > 0 else 0
        }
    
    def train(
        self,
        total_timesteps: int,
        rollout_steps: int = 2048,
        log_interval: int = 10,
        save_interval: int = 100,
        eval_interval: int = 50,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs'
    ):
        """
        Main training loop
        
        Args:
            total_timesteps: Total number of environment steps
            rollout_steps: Steps to collect before each update
            log_interval: Episodes between logging
            save_interval: Episodes between saving checkpoints
            eval_interval: Episodes between evaluation
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
        """
        # Setup logging
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs('diagrams', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(log_dir, f'feynman_gcpn_{timestamp}'))
        
        # Create initial diagram file so visualization doesn't fail
        self._save_current_diagram()
        
        print("=" * 80)
        print("Starting Feynman-GCPN Training")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Rollout steps: {rollout_steps}")
        env_ref = self.vis_env if self.vis_env else (self.env.envs[0] if hasattr(self.env, 'envs') else self.env)
        print(f"Initial state: {env_ref.initial_particles}")
        print(f"Final state: {env_ref.final_particles}")
        print("=" * 80)
        
        num_updates = total_timesteps // rollout_steps
        update_num = 0
        
        with tqdm(total=total_timesteps, desc="Training") as pbar:
            while self.global_step < total_timesteps:
                # Collect rollout
                rollout_stats = self.collect_rollout(rollout_steps)
                self.global_step += rollout_steps
                
                # Update policy
                update_stats = self.update_policy()
                update_num += 1
                
                # Logging
                if len(rollout_stats['episode_rewards']) > 0:
                    self.writer.add_scalar('train/mean_reward', rollout_stats['mean_reward'], self.global_step)
                    self.writer.add_scalar('train/mean_length', rollout_stats['mean_length'], self.global_step)
                
                if update_stats:
                    self.writer.add_scalar('train/policy_loss', update_stats['policy_loss'], self.global_step)
                    self.writer.add_scalar('train/value_loss', update_stats['value_loss'], self.global_step)
                    self.writer.add_scalar('train/entropy', update_stats['entropy'], self.global_step)
                
                # Console logging
                if update_num % log_interval == 0:
                    print(f"\n[Update {update_num}/{num_updates}] Step: {self.global_step}")
                    if len(rollout_stats['episode_rewards']) > 0:
                        print(f"  Mean Reward: {rollout_stats['mean_reward']:.2f}")
                        print(f"  Mean Length: {rollout_stats['mean_length']:.1f}")
                        print(f"  Best Reward: {self.best_reward:.2f}")
                    if update_stats:
                        print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
                        print(f"  Value Loss: {update_stats['value_loss']:.4f}")
                
                # Save checkpoint
                if update_num % save_interval == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f'model_step_{self.global_step}.pt')
                    self.save_checkpoint(checkpoint_path)
                    print(f"  Saved checkpoint: {checkpoint_path}")
                
                # Save best diagram
                if self.best_diagram is not None:
                    diagram_path = 'diagrams/current_best.json'
                    with open(diagram_path, 'w') as f:
                        json.dump(self.best_diagram, f, indent=2)
                
                pbar.update(rollout_steps)
        
        # Final save
        final_path = os.path.join(checkpoint_dir, 'model_final.pt')
        self.save_checkpoint(final_path)
        print(f"\n{'='*80}")
        print(f"Training complete! Final model saved to {final_path}")
        print(f"Best reward achieved: {self.best_reward:.2f}")
        print(f"{'='*80}")
        
        self.writer.close()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_reward': self.best_reward,
            'best_diagram': self.best_diagram
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_reward = checkpoint.get('best_reward', -float('inf'))
        self.best_diagram = checkpoint.get('best_diagram', None)
        print(f"Loaded checkpoint from {path}")
    
    def _save_best_diagram(self):
        """Save best diagram to JSON file for visualization"""
        if self.best_diagram is not None:
            import json
            os.makedirs('diagrams', exist_ok=True)
            env_ref = self.vis_env if self.vis_env else (self.env.envs[0] if hasattr(self.env, 'envs') else self.env)
            diagram_data = {
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'episode': self.global_step // 2048,  # Approximate episode number
                    'reward': float(self.best_reward),
                    'initial_state': env_ref.initial_particles,
                    'final_state': env_ref.final_particles
                },
                'shapes': self.best_diagram
            }
            with open('diagrams/current_best.json', 'w', encoding='utf-8') as f:
                json.dump(diagram_data, f, indent=2, ensure_ascii=False)
    
    def _save_current_diagram(self):
        """Save current diagram for live monitoring"""
        import json
        os.makedirs('diagrams', exist_ok=True)
        env_ref = self.vis_env if self.vis_env else (self.env.envs[0] if hasattr(self.env, 'envs') else self.env)
        current_diagram = env_ref.get_diagram_json()
        
        # 如果图为空，创建一个占位符显示初态和末态
        if not current_diagram:
            # 创建初态粒子的可视化
            num_initial = len(env_ref.initial_particles)
            num_final = len(env_ref.final_particles)
            y_step = 100
            
            for i, p_id in enumerate(env_ref.initial_particles):
                # 解析反粒子后缀 _bar
                base_id, is_anti = (p_id[:-4], True) if p_id.endswith('_bar') else (p_id, False)
                
                # 反粒子需要从右往左绘制
                if is_anti:
                    p1, p2 = {'x': 150, 'y': 200 + i * y_step}, {'x': 50, 'y': 200 + i * y_step}
                else:
                    p1, p2 = {'x': 50, 'y': 200 + i * y_step}, {'x': 150, 'y': 200 + i * y_step}
                
                current_diagram.append({
                    'id': f'initial_{i}',
                    'type': 'fermion',
                    'p1': p1,
                    'p2': p2,
                    'props': {
                        'particleId': base_id,
                        'isAnti': is_anti,
                        'color': 'none',
                        'category': 'fermion',
                        'group': 'initial'
                    }
                })
            
            for i, p_id in enumerate(env_ref.final_particles):
                # 解析反粒子后缀 _bar
                base_id, is_anti = (p_id[:-4], True) if p_id.endswith('_bar') else (p_id, False)
                
                # 反粒子需要从右往左绘制
                if is_anti:
                    p1, p2 = {'x': 750, 'y': 200 + i * y_step}, {'x': 650, 'y': 200 + i * y_step}
                else:
                    p1, p2 = {'x': 650, 'y': 200 + i * y_step}, {'x': 750, 'y': 200 + i * y_step}
                
                current_diagram.append({
                    'id': f'final_{i}',
                    'type': 'fermion',
                    'p1': p1,
                    'p2': p2,
                    'props': {
                        'particleId': base_id,
                        'isAnti': is_anti,
                        'color': 'none',
                        'category': 'fermion',
                        'group': 'final'
                    }
                })
        
        diagram_data = {
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'episode': self.global_step // 2048,
                'reward': 0.0,
                'initial_state': env_ref.initial_particles,
                'final_state': env_ref.final_particles,
                'num_steps': len(env_ref.action_history)
            },
            'shapes': current_diagram,
            'actions': env_ref.action_history  # NEW: Include action history for visualization
        }
        with open('diagrams/current_diagram.json', 'w', encoding='utf-8') as f:
            json.dump(diagram_data, f, indent=2, ensure_ascii=False)
        
        # 同时保存到current_best.json以便可视化显示
        with open('diagrams/current_best.json', 'w', encoding='utf-8') as f:
            json.dump(diagram_data, f, indent=2, ensure_ascii=False)
    
    def _extract_vertex_states(self) -> List[Dict]:
        """
        Extract quantum number states from current environment
        For physics gating
        """
        vertex_states = []
        
        for vertex in self.env.vertices:
            connected_edges = [self.env.edges[eid] for eid in vertex['connected_edges']]
            
            incoming = [e for e in connected_edges if e['target'] == vertex['id']]
            outgoing = [e for e in connected_edges if e['source'] == vertex['id']]
            
            state = {
                'charge_in': sum(self.env._get_charge(e) for e in incoming),
                'charge_out': sum(self.env._get_charge(e) for e in outgoing),
                'lepton_in': sum(self.env._get_lepton(e) for e in incoming),
                'lepton_out': sum(self.env._get_lepton(e) for e in outgoing),
                'baryon_in': sum(self.env._get_baryon(e) for e in incoming),
                'baryon_out': sum(self.env._get_baryon(e) for e in outgoing),
                'colors_in': [e['color'] for e in incoming if e['color']],
                'colors_out': [e['color'] for e in outgoing if e['color']]
            }
            
            vertex_states.append(state)
        
        return vertex_states


def main():
    """Example training run"""
    
    # Define reaction: e⁻ + e⁺ → μ⁻ + μ⁺ (Bhabha scattering via photon)
    initial_state = ['e', 'e']  # e⁻ and e⁺ (second will be anti)
    final_state = ['mu', 'mu']  # μ⁻ and μ⁺
    
    # Create environment
    env = FeynmanDiagramEnv(
        initial_state=initial_state,
        final_state=final_state,
        max_vertices=10,
        max_steps=50
    )
    
    # Create model
    num_particle_types = len(PhysicsConstants.get_all_particles()) + len(PhysicsConstants.BOSONS)
    
    model = FeynmanGCPN(
        node_input_dim=9,  # Updated from 6 to 9
        edge_input_dim=21,
        hidden_dim=128,
        num_mp_layers=3,
        num_action_types=5,  # Updated from 4 to 5 for ACTION_MERGE
        num_particle_types=num_particle_types,
        max_vertices=10,
        lambda_penalty=5.0
    )
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        model=model,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    # Train
    trainer.train(
        total_timesteps=100000,
        rollout_steps=2048,
        log_interval=10,
        save_interval=100,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    )


if __name__ == '__main__':
    main()
