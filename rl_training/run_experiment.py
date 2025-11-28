"""
Main Entry Point for Feynman-GCPN V8 Experiments
Conservation Law Discovery through Reinforcement Learning
"""

import argparse
import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, QuickTestConfig
from models import FeynmanGCPN
from feynman_env import FeynmanDiagramEnv
from training import PPOTrainer
from evaluator import ConservationLawEvaluator
from particle_utils import get_particle_list, validate_reaction, get_reaction_string
from physics_engine import PhysicsConstants


def parse_args():
    parser = argparse.ArgumentParser(
        description='Feynman-GCPN V8: Conservation Law Discovery Experiment'
    )

    parser.add_argument('--quick', action='store_true',
                      help='Quick test mode (10k steps)')
    parser.add_argument('--steps', type=int, default=None,
                      help='Total training steps (overrides config)')
    parser.add_argument('--learnable-dim', type=int, default=None,
                      help='Number of learnable embedding dimensions')
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device (auto/cpu/cuda)')
    parser.add_argument('--reaction', type=str, default=None,
                      help='Single reaction to train on (e.g., "mu->e+nu_e_bar+nu_mu")')

    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup computation device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return device


def create_training_envs(config: Config):
    """Create environments for all training reactions"""
    reactions = config.get_training_reactions()
    envs = []

    print(f"\nCreating {len(reactions)} training environments:")
    for reaction in reactions:
        # Validate reaction
        is_valid, errors = validate_reaction(reaction['initial'], reaction['final'])
        if not is_valid:
            print(f"  ✗ {reaction['name']}: {errors}")
            continue

        env = FeynmanDiagramEnv(
            initial_state=reaction['initial'],
            final_state=reaction['final'],
            max_vertices=config.max_vertices,
            max_steps=config.max_steps,
            reward_weights=config.known_laws_reward
        )

        reaction_str = get_reaction_string(reaction['initial'], reaction['final'])
        print(f"  ✓ {reaction['name']}: {reaction_str}")
        envs.append(env)

    return envs


def main():
    args = parse_args()

    print("=" * 80)
    print("🚀 Feynman-GCPN V8: Conservation Law Discovery")
    print("=" * 80)

    # Load configuration
    if args.quick:
        config = QuickTestConfig()
        print("\n⚡ Quick test mode enabled")
    else:
        config = Config()

    # Override config with command line args
    if args.steps is not None:
        config.total_steps = args.steps
    if args.learnable_dim is not None:
        config.learnable_dim = args.learnable_dim
        config.total_embedding_dim = config.fixed_dim + config.learnable_dim

    # Setup device
    device = setup_device(args.device)

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 Experiment Configuration:")
    print(f"  Total steps: {config.total_steps:,}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Embedding: Fixed={config.fixed_dim} (Q,L) + Learnable={config.learnable_dim}")
    print(f"  Physics penalty λ: {config.physics_penalty}")
    print(f"  Sparsity weight: {config.sparsity_weight}")

    # Create environments
    if args.reaction:
        # Single reaction mode
        print(f"\n🎯 Single reaction mode: {args.reaction}")
        initial_str, final_str = args.reaction.split('->')
        initial = [p.strip() for p in initial_str.split('+')]
        final = [p.strip() for p in final_str.split('+')]

        is_valid, errors = validate_reaction(initial, final)
        if not is_valid:
            print(f"❌ Invalid reaction: {errors}")
            return

        env = FeynmanDiagramEnv(
            initial_state=initial,
            final_state=final,
            max_vertices=config.max_vertices,
            max_steps=config.max_steps,
            reward_weights=config.known_laws_reward
        )
        envs = [env]
    else:
        # Multi-reaction training (default)
        envs = create_training_envs(config)

    if not envs:
        print("❌ No valid environments created")
        return

    # Create model
    print("\n🧠 Building Feynman-GCPN V8 model...")
    num_particle_types = len(get_particle_list())

    model = FeynmanGCPN(
        node_input_dim=9,
        edge_input_dim=21,
        hidden_dim=config.hidden_dim,
        num_mp_layers=config.num_mp_layers,
        num_action_types=4,
        num_particle_types=num_particle_types,
        max_vertices=config.max_vertices,
        lambda_penalty=config.physics_penalty,
        fixed_dim=config.fixed_dim,
        learnable_dim=config.learnable_dim,
        sparsity_weight=config.sparsity_weight
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    print(f"  Split Embedding: {config.fixed_dim} fixed + {config.learnable_dim} learnable = {config.total_embedding_dim}")

    # Create trainer
    print("\n⚙️  Initializing PPO trainer...")
    # Use first env for trainer (will cycle through all envs during training)
    trainer = PPOTrainer(
        env=envs[0],
        model=model,
        device=device,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_epsilon=config.clip_epsilon,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
        batch_size=config.batch_size,
        epochs_per_update=4
    )

    # Store all envs for multi-task training
    trainer.training_envs = envs

    print("\n✅ Setup complete!")
    print(f"  Training on {len(envs)} reactions")
    print(f"  Output directory: {output_dir}")

    # Start training
    print("\n" + "=" * 80)
    print("🚂 Starting Training...")
    print("=" * 80 + "\n")

    trainer.train(
        total_timesteps=config.total_steps,
        rollout_steps=512,
        log_interval=100,
        save_interval=config.checkpoint_interval,
        checkpoint_dir=str(output_dir / "checkpoints"),
        log_dir=str(output_dir / "logs")
    )

    print("\n🎉 Training complete!")

    # Evaluate discovery
    print("\n" + "=" * 80)
    print("📊 Evaluating Conservation Law Discovery...")
    print("=" * 80)

    evaluator = ConservationLawEvaluator(model, config)
    metrics = evaluator.evaluate_discovery()

    # Print summary
    evaluator.print_discovery_summary(metrics)

    # Save results
    evaluator.save_analysis(metrics, str(output_dir / "discovery_analysis.json"))

    # Save final model
    final_model_path = output_dir / "final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics
    }, final_model_path)
    print(f"\n💾 Model saved to: {final_model_path}")

    # Print final summary
    print("\n" + "=" * 80)
    print("📈 Experiment Summary:")
    print("=" * 80)
    print(f"  Total steps: {config.total_steps:,}")
    print(f"  Best correlation (learnable ↔ B): {metrics['best_correlation']:.3f}")
    print(f"  Hadronic success rate: {metrics['train_hadronic_success_rate']:.1%}")
    print(f"  Baryon violation rate: {metrics['train_baryon_violation_rate']:.1%}")
    print(f"  Results saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
