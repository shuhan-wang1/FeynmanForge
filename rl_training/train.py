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
    parser.add_argument('--pretrained', type=str, default=None,
                      help='Path to pretrained model checkpoint')
    parser.add_argument('--entropy-coef', type=float, default=0.02,
                      help='Entropy coefficient for exploration (reduce if using pretrained model)')
    
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
    
    # Validate particles (支持反粒子 _bar 后缀)
    all_particles = [p.id for p in PhysicsConstants.get_all_particles()]
    for p in initial_state + final_state:
        # 移除 _bar 后缀进行验证
        p_base = p.replace('_bar', '') if p.endswith('_bar') else p
        if p_base not in all_particles:
            print(f"❌ Unknown particle: {p_base}")
            print(f"Available particles: {', '.join(all_particles)}")
            print(f"💡 Use '_bar' suffix for antiparticles, e.g., 'e_bar' for positron")
            return
    
    # Create environment
    print("\n📦 Setting up environment...")
    
    # 使用并行环境充分利用 CPU 多核
    # OPTIMIZED: Increased from 256 to 512 for better CPU utilization
    num_parallel_envs = 512 if device.type == 'cuda' else 128
    print(f"   Creating {num_parallel_envs} parallel environments...")

    # 创建并行环境包装器
    env = make_parallel_envs(
        num_envs=num_parallel_envs,
        initial_state=initial_state,
        final_state=final_state,
        max_vertices=10,
        max_steps=50
    )
    print(f"   ✅ {num_parallel_envs} parallel environments created")

    # 创建一个单独的环境用于可视化
    vis_env = FeynmanDiagramEnv(
        initial_state=initial_state,
        final_state=final_state,
        max_vertices=10,
        max_steps=50
    )

    # If using pretrained model, detect its architecture first
    if args.pretrained:
        import torch as torch_load
        print(f"\n📦 Loading pretrained model architecture from {args.pretrained}...")
        checkpoint = torch_load.load(args.pretrained, map_location=device)

        # Detect architecture from pretrained weights
        # Check node_encoder shape to determine hidden_dim
        node_encoder_shape = checkpoint['model_state_dict']['encoder.node_encoder.weight'].shape
        hidden_dim = node_encoder_shape[0]  # First dimension is hidden_dim

        # Count MP layers by checking how many mp_layers.X exist
        num_mp_layers = 0
        for key in checkpoint['model_state_dict'].keys():
            if key.startswith('encoder.mp_layers.'):
                layer_idx = int(key.split('.')[2])
                num_mp_layers = max(num_mp_layers, layer_idx + 1)

        print(f"   Detected architecture: hidden_dim={hidden_dim}, num_mp_layers={num_mp_layers}")
    else:
        # Determine model size based on device
        # OPTIMIZED: Increased hidden_dim from 384 to 768 for better GPU utilization
        # Increased MP layers from 5 to 8 for more model capacity
        hidden_dim = 768 if device.type == 'cuda' else args.hidden_dim
        num_mp_layers = 8 if device.type == 'cuda' else 3

    # Create model
    print("🧠 Building neural network...")
    num_particle_types = len(PhysicsConstants.get_all_particles()) + len(PhysicsConstants.BOSONS)

    model = FeynmanGCPN(
        node_input_dim=9,
        edge_input_dim=21,
        hidden_dim=hidden_dim,
        num_mp_layers=num_mp_layers,
        num_action_types=5,  # Updated from 4 to 5 for ACTION_MERGE
        num_particle_types=num_particle_types,
        max_vertices=10,
        lambda_penalty=args.lambda_penalty
    ).to(device)

    # Load pretrained weights if specified
    if args.pretrained:
        print(f"   Loading pretrained weights...")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ Loaded pretrained model!")
        if 'accuracy' in checkpoint:
            print(f"   Pretrain accuracy: {checkpoint['accuracy']:.2%}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    print(f"   Device: {device}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   MP layers: {num_mp_layers}")
    
    # Create trainer
    print("⚙️  Configuring PPO trainer...")

    # Use optimized batch size for GPU and parallel envs
    # OPTIMIZED: Increased batch size from 512 to 2048 for better GPU utilization
    # batch_size 应该是 num_parallel_envs 的倍数
    batch_size = 2048 if device.type == 'cuda' else 128
    # OPTIMIZED: Increased rollout_steps from 512 to 1024 for more data per update
    rollout_steps = 1024 if device.type == 'cuda' else 256  # 每个环境的步数

    # Adjust entropy coefficient based on whether using pretrained model
    # Lower entropy when using pretrained model to exploit learned behavior
    if args.pretrained:
        entropy_coef = args.entropy_coef  # Default: 0.02 (low exploration)
        print(f"   Using reduced entropy coefficient: {entropy_coef} (pretrained model)")
    else:
        entropy_coef = 0.3  # High exploration for random initialization
        print(f"   Using standard entropy coefficient: {entropy_coef}")

    trainer = PPOTrainer(
        env=env,
        model=model,
        device=device,
        learning_rate=2e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=entropy_coef,  # Adjusted based on pretrained flag
        batch_size=batch_size,
        epochs_per_update=6,  # OPTIMIZED: Increased from 4 to 6 for better learning
        num_envs=num_parallel_envs,
        use_amp=True  # Enable mixed precision training
    )
    
    # Create reaction config for visualization
    create_reaction_config(initial_state, final_state)
    
    # 设置trainer使用单独的可视化环境
    trainer.vis_env = vis_env
    
    print("\n✅ Setup complete!")
    print(f"   Parallel environments: {num_parallel_envs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Rollout steps: {rollout_steps}")
    print(f"   PPO epochs per update: 4")
    print(f"   Checkpoints will be saved to: {args.checkpoint_dir}")
    print(f"   TensorBoard logs: {args.log_dir}")
    print(f"   Training monitor: Open training_viz.html in your browser")
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
        save_interval=10,
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
