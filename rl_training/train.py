"""
Quick Start Script for Feynman-GCPN
Run this to start training immediately with sensible defaults
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feynman_env import FeynmanDiagramEnv
from models import FeynmanGCPN
from training import PPOTrainer
from physics_engine import PhysicsConstants
from visualization_bridge import create_reaction_config
from gpu_optimization import configure_gpu_optimization, OPTIMIZED_CONFIG, check_gpu_utilization
from parallel_env import make_parallel_envs

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train Feynman-GCPN')
    
    parser.add_argument('--reaction', type=str, default='e+e->mu+mu',
                      help='Reaction to train on (format: particle1+particle2->particle3+particle4)')
    parser.add_argument('--timesteps', type=int, default=100000,
                      help='Total training timesteps')
    parser.add_argument('--hidden-dim', type=int, default=128,
                      help='Hidden dimension size')
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--lambda-penalty', type=float, default=5.0,
                      help='Physics gate penalty strength')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                      help='Directory for TensorBoard logs')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (auto/cpu/cuda)')
    
    return parser.parse_args()


def parse_reaction(reaction_str):
    """
    Parse reaction string like 'e+e->mu+mu' into initial and final states
    """
    if '->' not in reaction_str:
        raise ValueError("Reaction must contain '->' separator")
    
    initial_str, final_str = reaction_str.split('->')
    initial_particles = [p.strip() for p in initial_str.split('+')]
    final_particles = [p.strip() for p in final_str.split('+')]
    
    return initial_particles, final_particles


def main():
    args = parse_args()
    
    print("=" * 80)
    print("🚀 Feynman-GCPN Quick Start")
    print("=" * 80)
    
    # Configure GPU optimizations FIRST
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    if device.type == 'cuda':
        configure_gpu_optimization(device.index or 0)
    else:
        device = torch.device('cpu')
    
    # Parse reaction
    try:
        initial_state, final_state = parse_reaction(args.reaction)
        print(f"Reaction: {' + '.join(initial_state)} → {' + '.join(final_state)}")
    except Exception as e:
        print(f"❌ Invalid reaction format: {e}")
        print("Example: --reaction 'e+e->mu+mu'")
        return
    
    # Validate particles
    all_particles = [p.id for p in PhysicsConstants.get_all_particles()]
    for p in initial_state + final_state:
        if p not in all_particles:
            print(f"❌ Unknown particle: {p}")
            print(f"Available particles: {', '.join(all_particles)}")
            return
    
    # Create environment
    print("\n📦 Setting up environment...")
    
    # 降低并行环境数，增加每个环境的复杂度
    num_parallel_envs = 1  # 单环境，专注于GPU计算
    print(f"   Creating environment...")
    
    env = FeynmanDiagramEnv(
        initial_state=initial_state,
        final_state=final_state,
        max_vertices=10,
        max_steps=50
    )
    print(f"   ✅ Environment created")
    
    # Determine model size based on device
    hidden_dim = 384 if device.type == 'cuda' else args.hidden_dim
    num_mp_layers = 5 if device.type == 'cuda' else 3
    
    # Create model
    print("🧠 Building neural network...")
    num_particle_types = len(PhysicsConstants.get_all_particles()) + len(PhysicsConstants.BOSONS)
    
    model = FeynmanGCPN(
        node_input_dim=9,
        edge_input_dim=21,
        hidden_dim=hidden_dim,
        num_mp_layers=num_mp_layers,
        num_action_types=4,
        num_particle_types=num_particle_types,
        max_vertices=10,
        lambda_penalty=args.lambda_penalty
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    print(f"   Device: {device}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   MP layers: {num_mp_layers}")
    
    # Create trainer
    print("⚙️  Configuring PPO trainer...")
    
    # Use optimized batch size for GPU
    batch_size = 512 if device.type == 'cuda' else 64  # 大幅增加batch size
    rollout_steps = 1024 if device.type == 'cuda' else 512  # 减少rollout，增加update比重
    
    trainer = PPOTrainer(
        env=env,
        model=model,
        device=device,
        learning_rate=2e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        batch_size=batch_size,
        epochs_per_update=10,  # 增加epoch提高GPU利用率
        num_envs=num_parallel_envs
    )
    
    # Create reaction config for visualization
    create_reaction_config(initial_state, final_state)
    
    # 设置trainer使用环境进行可视化
    trainer.vis_env = env
    
    print("\n✅ Setup complete!")
    print(f"   Checkpoints will be saved to: {args.checkpoint_dir}")
    print(f"   TensorBoard logs: {args.log_dir}")
    print(f"   Training monitor: Open training_viz.html in your browser")
    print(f"   Batch size: {batch_size} (large for GPU)")
    print(f"   Rollout steps: {rollout_steps} (reduced)")
    print(f"   PPO epochs: 10 (increased for more GPU compute)")
    if device.type == 'cuda':
        print(f"   🚀 GPU optimizations: ENABLED")
    print("\n" + "=" * 80)
    
    # Check initial GPU utilization
    if device.type == 'cuda':
        check_gpu_utilization()
    
    # Start training
    trainer.train(
        total_timesteps=args.timesteps,
        rollout_steps=rollout_steps,
        log_interval=10,
        save_interval=100,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    print("\n🎉 Training complete!")
    print(f"   Best reward: {trainer.best_reward:.2f}")
    print(f"   Best diagram saved to: diagrams/current_best.json")
    print("\n💡 Next steps:")
    print("   1. View TensorBoard: tensorboard --logdir=logs")
    print("   2. Open training_viz.html to see the final diagram")
    print("   3. Import diagrams/current_best.json into Feynman Forge")


if __name__ == '__main__':
    main()
