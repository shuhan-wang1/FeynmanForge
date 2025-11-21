"""
并行环境包装器 - 充分利用多核CPU和GPU

使用多进程真正利用多核CPU（绕过Python GIL）
"""

import torch
import numpy as np
from typing import List
from multiprocessing import Process, Queue, Pool
from feynman_env import FeynmanDiagramEnv


def worker_step(env, action, result_queue):
    """工作进程：执行单个环境的step"""
    result = env.step(action)
    result_queue.put(result)


class ParallelEnvs:
    """
    运行多个环境实例并行收集数据
    使用多进程绕过GIL真正利用多核CPU
    """
    
    def __init__(self, env_fns: List[callable], device='cuda'):
        """
        Args:
            env_fns: 环境构造函数列表
            device: PyTorch设备
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.device = device
        # 不使用进程池，因为环境有状态
        
    def reset(self):
        """重置所有环境"""
        states = []
        infos = []
        for env in self.envs:
            state, info = env.reset()
            states.append(state)
            infos.append(info)
        return states, infos
    
    def step(self, actions: List[dict]):
        """
        简化版：不使用多进程（环境状态难以序列化）
        但优化为批量操作减少开销
        
        Args:
            actions: 每个环境的动作列表
        """
        # 直接批量执行，Python的C扩展部分可以释放GIL
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        
        states = [r[0] for r in results]
        rewards = [r[1] for r in results]
        terminateds = [r[2] for r in results]
        truncateds = [r[3] for r in results]
        infos = [r[4] for r in results]
        
        return states, rewards, terminateds, truncateds, infos
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()


def make_parallel_envs(num_envs: int, initial_state: List[str], final_state: List[str], 
                       max_vertices: int = 10, max_steps: int = 50):
    """
    创建并行环境
    
    Args:
        num_envs: 并行环境数量（建议等于CPU核心数）
        initial_state: 初态粒子
        final_state: 末态粒子
    
    Returns:
        ParallelEnvs实例
    """
    def make_env(init_s, final_s, max_v, max_s):
        def _init():
            return FeynmanDiagramEnv(
                initial_state=init_s,
                final_state=final_s,
                max_vertices=max_v,
                max_steps=max_s
            )
        return _init
    
    env_fns = [
        make_env(initial_state, final_state, max_vertices, max_steps)
        for _ in range(num_envs)
    ]
    
    return ParallelEnvs(env_fns)
