"""
并行环境包装器 - Multiprocessing版本
使用 torch.multiprocessing 绕过 GIL，真正利用多核 CPU
"""

import torch
import torch.multiprocessing as mp
from torch_geometric.data import Data, Batch
import sys
import numpy as np
from multiprocessing.connection import wait
import errno


# ✅ FIX 1: 深度递归CUDA清理 - 防止序列化错误
def ensure_cpu_only(obj):
    """
    递归确保对象中所有张量都在CPU上
    防止 CUDA tensor serialization errors in multiprocessing
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu() if obj.is_cuda else obj
    elif isinstance(obj, Data):
        # PyG Data对象特殊处理
        cpu_data = Data()
        # ✅ FIX: 兼容新版 PyG，obj.keys 可能是方法
        keys = obj.keys() if callable(obj.keys) else obj.keys
        for key in keys:
            attr = getattr(obj, key)
            if isinstance(attr, torch.Tensor):
                setattr(cpu_data, key, attr.cpu() if attr.is_cuda else attr)
            else:
                setattr(cpu_data, key, attr)
        # 保留batch属性（如果存在且不为None）
        if hasattr(obj, 'batch') and obj.batch is not None:
            cpu_data.batch = obj.batch.cpu() if obj.batch.is_cuda else obj.batch
        return cpu_data
    elif isinstance(obj, dict):
        return {k: ensure_cpu_only(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(ensure_cpu_only(item) for item in obj)
    else:
        return obj


# 这是一个给子进程运行的 Worker 函数
def worker(remote, parent_remote, env_fn_wrapper):
    """
    Worker process that runs a single environment
    
    ✅ FIXED: 
    - Deep CUDA cleaning to prevent serialization errors
    - Timeout protection for operations
    - Robust error handling
    """
    import os
    
    # ✅ FIX 2: 禁用子进程中的CUDA（强制CPU-only）
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    parent_remote.close()
    env = None
    
    try:
        # Create environment in worker process
        env = env_fn_wrapper.x()
        
        # Send success signal
        remote.send(('ready', None))
        
        while True:
            try:
                # ✅ FIX 3: Poll with timeout to detect dead pipes
                if not remote.poll(timeout=60):  # 60秒超时
                    continue
                    
                cmd, data = remote.recv()
            except EOFError:
                # Pipe closed by parent - exit gracefully
                break
            except Exception as e:
                # 其他接收错误 - 尝试报告后退出
                try:
                    remote.send(('error', f"Recv error: {e}"))
                except:
                    pass
                break
            
            try:
                if cmd == 'step':
                    state, reward, terminated, truncated, info = env.step(data)
                    
                    # ✅ FIX 1: 深度清理CUDA张量（递归处理嵌套结构）
                    state = ensure_cpu_only(state)
                    info = ensure_cpu_only(info)
                    
                    remote.send(('result', (state, reward, terminated, truncated, info)))
                    
                elif cmd == 'reset':
                    state, info = env.reset()
                    
                    # ✅ FIX 1: 深度清理
                    state = ensure_cpu_only(state)
                    info = ensure_cpu_only(info)
                    
                    remote.send(('result', (state, info)))
                    
                elif cmd == 'close':
                    break
                    
                elif cmd == 'get_attr':
                    result = getattr(env, data)
                    # 清理可能的CUDA张量
                    result = ensure_cpu_only(result)
                    remote.send(('result', result))
                    
                elif cmd == 'set_attr':
                    setattr(env, data[0], data[1])
                    remote.send(('result', None))
                    
                else:
                    raise NotImplementedError(f"Unknown command: {cmd}")
                    
            except Exception as e:
                # Send exception back to parent
                import traceback
                error_msg = f"Worker error in {cmd}: {str(e)}\n{traceback.format_exc()}"
                try:
                    remote.send(('error', error_msg))
                except:
                    # 如果发送错误消息失败，pipe可能已断
                    break
                
    except Exception as e:
        # Environment creation failed
        import traceback
        error_msg = f"Worker init error: {str(e)}\n{traceback.format_exc()}"
        try:
            remote.send(('error', error_msg))
        except:
            pass  # Parent pipe may be closed
            
    finally:
        # Cleanup
        if env is not None and hasattr(env, 'close'):
            try:
                env.close()
            except:
                pass
        try:
            remote.close()
        except:
            pass

# Cloudpickle wrapper to handle lambda functions
class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class ParallelEnvs:
    """
    🚀 MULTIPROCESSING环境包装器 - 绕过GIL限制
    
    关键改进:
    - 使用多进程而非多线程，每个环境在独立进程中运行
    - 真正并行执行CPU密集型环境step
    - 每个进程独立运行，不受GIL限制
    
    预期提速: 在12 vCPU上，从1核 → 接近12核并行 = ~10x
    """
    
    def __init__(self, env_fns, max_workers=None, timeout=30.0, worker_restart_interval=5000):
        """
        env_fns: list of functions that return envs
        timeout: 超时时间（秒），防止死锁
        worker_restart_interval: ✅ Windows共享内存泄漏修复：每N步重启worker
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        self.timeout = timeout
        self.env_fns = env_fns  # ✅ 保存env_fns用于重启
        self.worker_restart_interval = worker_restart_interval
        self.step_counts = [0] * self.num_envs  # ✅ 追踪每个worker的步数
        
        # 使用 spawn (Windows下必须spawn, Linux可用forkserver)
        self.ctx = mp.get_context('spawn')
        
        # Start initial workers
        self._start_workers()
        
        self._current_states = [None] * self.num_envs
        
        # Mock envs attribute for compatibility with existing code
        self.envs = [None] * self.num_envs  # Not directly accessible in multiprocessing
    
    def _start_workers(self):
        """✅ 启动或重启worker进程"""
        self.remotes, self.work_remotes = zip(*[self.ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = [
            self.ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, self.env_fns)
        ]
        
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        
        # Wait for all workers to initialize
        print(f"Initializing {self.num_envs} worker processes...")
        for i, remote in enumerate(self.remotes):
            status, data = remote.recv()
            if status == 'error':
                raise RuntimeError(f"Worker {i} failed to initialize: {data}")
        print(f"✓ All {self.num_envs} workers ready!")
    
    def _restart_worker(self, env_idx):
        """✅ 重启单个worker来清理共享内存"""
        print(f"  🔄 Restarting worker {env_idx} to clear shared memory...")
        
        # Close old worker
        try:
            self.remotes[env_idx].send(('close', None))
            self.ps[env_idx].join(timeout=2.0)
            if self.ps[env_idx].is_alive():
                self.ps[env_idx].terminate()
                self.ps[env_idx].join()
        except:
            pass
        
        try:
            self.remotes[env_idx].close()
        except:
            pass
        
        # Create new worker
        parent_conn, child_conn = self.ctx.Pipe()
        p = self.ctx.Process(
            target=worker,
            args=(child_conn, parent_conn, CloudpickleWrapper(self.env_fns[env_idx]))
        )
        p.daemon = True
        p.start()
        child_conn.close()
        
        # Wait for initialization
        status, data = parent_conn.recv()
        if status == 'error':
            raise RuntimeError(f"Worker {env_idx} restart failed: {data}")
        
        # Update references
        self.remotes = list(self.remotes)
        self.ps = list(self.ps)
        self.remotes[env_idx] = parent_conn
        self.ps[env_idx] = p
        self.step_counts[env_idx] = 0  # Reset counter
        
        print(f"  ✓ Worker {env_idx} restarted successfully")

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self, timeout=None):
        """
        ✅ FIXED: 带超时的step_wait，防止死锁 + Windows共享内存修复
        
        Args:
            timeout: 超时时间（秒），None则使用默认值
        """
        if timeout is None:
            timeout = self.timeout
        
        results = []
        for i, remote in enumerate(self.remotes):
            try:
                # ✅ FIX: 使用wait()等待，带超时
                if not wait([remote], timeout=timeout):
                    raise TimeoutError(f"Worker {i} timeout after {timeout}s - possible deadlock")
                
                # 再次检查是否有数据
                if remote.poll():
                    status, data = remote.recv()
                    if status == 'error':
                        raise RuntimeError(f"Worker {i} step failed: {data}")
                    results.append(data)
                else:
                    raise RuntimeError(f"Worker {i} pipe broken (no data after wait)")
                    
            except EOFError:
                raise RuntimeError(f"Worker {i} died unexpectedly (EOFError)")
            except OSError as e:
                if e.errno == errno.EBADF:
                    raise RuntimeError(f"Worker {i} pipe closed (bad file descriptor)")
                raise
        
        self.waiting = False
        
        # Unpack results
        states, rewards, terminateds, truncateds, infos = zip(*results)
        
        # ✅ Windows 共享内存修复：追踪步数并定期重启workers
        for i in range(self.num_envs):
            self.step_counts[i] += 1
            if self.step_counts[i] >= self.worker_restart_interval:
                # 重启worker并重新reset环境
                self._restart_worker(i)
                # Reset刚重启的worker
                self.remotes[i].send(('reset', None))
                status, data = self.remotes[i].recv()
                if status == 'error':
                    raise RuntimeError(f"Worker {i} reset after restart failed: {data}")
                # Update state
                new_state, new_info = data
                states = list(states)
                infos = list(infos)
                states[i] = new_state
                infos[i] = new_info
                
        self._current_states = list(states)
        return list(states), list(rewards), list(terminateds), list(truncateds), list(infos)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self, timeout=None):
        """
        ✅ FIXED: 带超时的reset
        """
        if timeout is None:
            timeout = self.timeout
            
        for remote in self.remotes:
            remote.send(('reset', None))
        
        results = []
        for i, remote in enumerate(self.remotes):
            try:
                # ✅ FIX: 超时检测
                if not wait([remote], timeout=timeout):
                    raise TimeoutError(f"Worker {i} reset timeout after {timeout}s")
                
                if remote.poll():
                    status, data = remote.recv()
                    if status == 'error':
                        raise RuntimeError(f"Worker {i} reset failed: {data}")
                    results.append(data)
                else:
                    raise RuntimeError(f"Worker {i} pipe broken during reset")
            except EOFError:
                raise RuntimeError(f"Worker {i} died during reset (EOFError)")
        
        states, infos = zip(*results)
        self._current_states = list(states)
        return list(states), list(infos)
    
    def reset_single(self, env_idx):
        """Reset a single environment"""
        self.remotes[env_idx].send(('reset', None))
        status, data = self.remotes[env_idx].recv()
        if status == 'error':
            raise RuntimeError(f"Worker {env_idx} reset failed: {data}")
        state, info = data
        self._current_states[env_idx] = state
        return state, info


    def get_batch_states(self):
        """Get batched PyG Data from all current states"""
        valid_states = [s for s in self._current_states if s is not None]
        if not valid_states: 
            return None
        return Batch.from_data_list(valid_states)
    
    def get_attr(self, attr_name, indices=None):
        """Get attribute from environments (works through IPC)"""
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            self.remotes[i].send(('get_attr', attr_name))
        
        results = []
        for i in indices:
            status, data = self.remotes[i].recv()
            if status == 'error':
                raise RuntimeError(f"Worker {i} get_attr failed: {data}")
            results.append(data)
        return results
    
    def set_attr(self, attr_name, value, indices=None):
        """Set attribute in environments (works through IPC)"""
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            self.remotes[i].send(('set_attr', (attr_name, value)))
        
        # Wait for acknowledgment
        for i in indices:
            status, data = self.remotes[i].recv()
            if status == 'error':
                raise RuntimeError(f"Worker {i} set_attr failed: {data}")


    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def __del__(self):
        self.close()

# 保持接口兼容
def make_parallel_envs(initial_state, final_state, num_envs, max_vertices, max_steps, reward_weights):
    from feynman_env import FeynmanDiagramEnv
    import copy
    
    def make_env():
        return FeynmanDiagramEnv(
            initial_state=copy.deepcopy(initial_state),
            final_state=copy.deepcopy(final_state),
            max_vertices=max_vertices,
            max_steps=max_steps,
            reward_weights=copy.deepcopy(reward_weights) if reward_weights else None
        )
    
    # 列表推导式创建函数列表
    env_fns = [make_env for _ in range(num_envs)]
    return ParallelEnvs(env_fns)


if __name__ == "__main__":
    # 测试并行环境
    print("Testing ParallelEnvs with Multiprocessing...")
    
    try:
        envs = make_parallel_envs(
            initial_state=['mu'],
            final_state=['e', 'nu_e_bar', 'nu_mu'],
            num_envs=4,
            max_vertices=10,
            max_steps=50,
            reward_weights=None
        )
        
        # 测试reset
        states, infos = envs.reset()
        print(f"Reset complete: {len(states)} states")
        
        # 测试batch
        batch = envs.get_batch_states()
        print(f"Batched: {batch}")
        
        # 测试step
        actions = [
            {'action_type': 3, 'vertex_idx': 0, 'particle_type': 0, 'target_vertex': 0}
            for _ in range(4)
        ]
        next_states, rewards, terms, truncs, infos = envs.step(actions)
        print(f"Step complete: rewards = {rewards}")
        
        envs.close()
        print("All tests passed!")
        
    except ImportError as e:
        print(f"Skipping test (missing dependency): {e}")
