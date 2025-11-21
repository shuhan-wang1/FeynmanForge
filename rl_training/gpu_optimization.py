"""
Performance Optimization Tips for Feynman-GCPN Training

This file provides configuration options to maximize GPU utilization.
"""

import torch

# ========================================
# GPU Optimization Settings
# ========================================

def configure_gpu_optimization(device_id=0):
    """
    Configure PyTorch for optimal GPU performance
    Call this before training starts
    
    Args:
        device_id: GPU device ID to use (-1 for CPU)
    
    Returns:
        torch.device: The configured device
    """
    if device_id >= 0 and torch.cuda.is_available():
        # Enable cuDNN auto-tuner for convolution algorithms
        torch.backends.cudnn.benchmark = True
        
        # Enable TF32 on Ampere GPUs (3080 Ti) for faster training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocator strategy
        # 'max_split_size_mb' can help with fragmentation
        # torch.cuda.set_per_process_memory_fraction(0.95, 0)
        
        print("✅ GPU optimizations enabled:")
        print(f"   - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   - TF32 for matmul: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"   - TF32 for cuDNN: {torch.backends.cudnn.allow_tf32}")
        print(f"   - GPU: {torch.cuda.get_device_name(device_id)}")
        print(f"   - CUDA version: {torch.version.cuda}")
        return torch.device(f'cuda:{device_id}')
    else:
        print("⚠️  CUDA not available, using CPU")
        return torch.device('cpu')


# ========================================
# Training Hyperparameters for Better GPU Utilization
# ========================================

OPTIMIZED_CONFIG = {
    # Increase batch size for better GPU utilization
    'batch_size': 256,  # 从128增加到256
    
    # Larger rollout buffer = more data per update
    'rollout_steps': 8192,  # 从4096增加到8192
    
    # Reduce PPO epochs if memory is tight
    'epochs_per_update': 6,  # 从4增加到6，更多GPU计算
    
    # Model size - larger models utilize GPU better
    'hidden_dim': 384,  # 从256增加到384
    'num_mp_layers': 5,  # 从4增加到5
    
    # Learning rate
    'learning_rate': 2e-4,  # 略微降低平衡更大模型
    
    # Mixed precision training (if supported)
    'use_amp': True,  # Automatic Mixed Precision
    
    # Gradient accumulation (if batch size limited by memory)
    'gradient_accumulation_steps': 1,
    
    # DataLoader settings
    'num_workers': 0,  # Set to 0 for RL (no pre-loading needed)
    'pin_memory': True,  # Faster host-to-device transfer
}


# ========================================
# Diagnostic Tools
# ========================================

def check_gpu_utilization():
    """Print current GPU utilization stats"""
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU Utilization Check")
        print("=" * 60)
        
        # Memory usage
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
        
        print(f"Memory Allocated: {allocated:.2f} GB")
        print(f"Memory Reserved:  {reserved:.2f} GB")
        print(f"Peak Allocated:   {max_allocated:.2f} GB")
        
        # Try to get nvidia-smi info
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                gpu_util, mem_util = result.stdout.strip().split(',')
                print(f"GPU Utilization:  {gpu_util.strip()}%")
                print(f"Mem Utilization:  {mem_util.strip()}%")
        except:
            pass
        
        print("=" * 60 + "\n")


def profile_training_step(model, sample_data, num_iterations=100):
    """
    Profile a training step to identify bottlenecks
    
    Args:
        model: The Feynman-GCPN model
        sample_data: A sample PyG Data object
        num_iterations: Number of iterations to profile
    """
    import time
    
    model.eval()
    sample_data = sample_data.to(model.device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_data)
    
    # Profile
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(sample_data)
    
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_iterations * 1000  # ms
    
    print(f"\n{'='*60}")
    print(f"Inference Profile ({num_iterations} iterations)")
    print(f"{'='*60}")
    print(f"Average time per forward pass: {avg_time:.2f} ms")
    print(f"Throughput: {1000/avg_time:.1f} FPS")
    print(f"{'='*60}\n")


# ========================================
# Suggested Usage
# ========================================

"""
Add this to your training script (train.py):

```python
from gpu_optimization import configure_gpu_optimization, check_gpu_utilization, OPTIMIZED_CONFIG

# Before creating the model
configure_gpu_optimization()

# Use optimized config
trainer = PPOTrainer(
    env=env,
    model=model,
    batch_size=OPTIMIZED_CONFIG['batch_size'],
    device='cuda'
)

# During training, periodically check utilization
check_gpu_utilization()

# Train with larger rollout steps
trainer.train(
    total_timesteps=500000,
    rollout_steps=OPTIMIZED_CONFIG['rollout_steps']
)
```

Expected GPU utilization with these settings:
- Memory: 60-80% of available VRAM
- Compute: 70-95% GPU utilization
- Training speed: 2-3x faster than default settings
"""
