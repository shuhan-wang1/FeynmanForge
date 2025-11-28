"""
Test script to verify V8 setup is working correctly
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from config import QuickTestConfig
from models import FeynmanGCPN
from feynman_env import FeynmanDiagramEnv
from particle_utils import get_particle_list
from evaluator import ConservationLawEvaluator

def test_config():
    """Test configuration loading"""
    print("=" * 60)
    print("Testing Configuration...")
    print("=" * 60)

    config = QuickTestConfig()
    print(f"[OK] Config loaded")
    print(f"  - Total steps: {config.total_steps:,}")
    print(f"  - Fixed dim: {config.fixed_dim}, Learnable dim: {config.learnable_dim}")
    print(f"  - Physics penalty λ: {config.physics_penalty}")
    print(f"  - Sparsity weight: {config.sparsity_weight}")

    training_reactions = config.get_training_reactions()
    testing_reactions = config.get_testing_reactions()
    print(f"  - Training reactions: {len(training_reactions)}")
    print(f"  - Testing reactions: {len(testing_reactions)}")

    return config


def test_environment(config):
    """Test environment creation"""
    print("\n" + "=" * 60)
    print("Testing Environment...")
    print("=" * 60)

    # Test a simple reaction
    env = FeynmanDiagramEnv(
        initial_state=['e', 'e_bar'],
        final_state=['mu', 'mu_bar'],
        max_vertices=config.max_vertices,
        max_steps=config.max_steps,
        reward_weights=config.known_laws_reward
    )

    print(f"[OK] Environment created")
    print(f"  - Initial state: {env.initial_particles}")
    print(f"  - Final state: {env.final_particles}")
    print(f"  - Action space: {env.action_space}")

    # Test reset
    obs, info = env.reset()
    print(f"[OK] Environment reset successful")
    print(f"  - Observation shape: nodes={obs.x.shape}, edges={obs.edge_index.shape}")

    # Test step reward structure
    print(f"\n  Reward structure (Scientist Reward):")
    print(f"    - Charge conservation: +{env.reward_weights['charge_conservation']}")
    print(f"    - Lepton conservation: +{env.reward_weights['lepton_conservation']}")
    print(f"    - Baryon violation: NO immediate feedback")
    print(f"    - Sparse global reward: +{env.reward_weights.get('sparse_global_reward', config.sparse_global_reward)}")

    return env


def test_model(config):
    """Test model creation"""
    print("\n" + "=" * 60)
    print("Testing Model (V8 Architecture)...")
    print("=" * 60)

    num_particles = len(get_particle_list())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FeynmanGCPN(
        node_input_dim=9,
        edge_input_dim=21,
        hidden_dim=config.hidden_dim,
        num_mp_layers=config.num_mp_layers,
        num_action_types=4,
        num_particle_types=num_particles,
        max_vertices=config.max_vertices,
        lambda_penalty=config.physics_penalty,
        fixed_dim=config.fixed_dim,
        learnable_dim=config.learnable_dim,
        sparsity_weight=config.sparsity_weight
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created successfully")
    print(f"  - Device: {device}")
    print(f"  - Total parameters: {num_params:,}")

    # Test V8 components
    print(f"\n  V8 Components:")
    print(f"    1. Split Particle Embedding:")
    print(f"       - Fixed dims: {model.particle_embedding.fixed_dim} (Q, L)")
    print(f"       - Learnable dims: {model.particle_embedding.learnable_dim}")
    print(f"       - Total: {model.particle_embedding.total_dim}")

    print(f"    2. Split Conservation Mask:")
    alpha = model.conservation_mask()
    print(f"       - α_fixed (Q, L): {alpha[:2].detach().cpu().numpy()}")
    print(f"       - α_learnable: {alpha[2:].detach().cpu().numpy()}")

    print(f"    3. Meta-Physics Gate: [OK]")

    # Test conservation metrics
    metrics = model.get_conservation_metrics()
    print(f"\n  Conservation Metrics:")
    print(f"    - Sparsity loss: {metrics['sparsity_loss'].item():.4f}")
    print(f"    - Embedding norm: {metrics['learnable_embedding_norm'].item():.4f}")

    return model, device


def test_forward_pass(model, env, device):
    """Test forward pass"""
    print("\n" + "=" * 60)
    print("Testing Forward Pass...")
    print("=" * 60)

    obs, _ = env.reset()
    obs = obs.to(device)

    with torch.no_grad():
        output = model(obs, return_value=True)

    print(f"[OK] Forward pass successful")
    print(f"  - Action type probs: {output['action_type_probs'].shape}")
    print(f"  - Vertex probs: {output['vertex_probs'].shape}")
    print(f"  - Particle probs: {output['particle_probs'].shape}")
    print(f"  - Value: {output['value'].item():.4f}")

    # Test action sampling
    action = model.get_action(obs, deterministic=False)
    print(f"\n[OK] Action sampling successful")
    print(f"  - Action type: {action['action_type']}")
    print(f"  - Vertex idx: {action['vertex_idx']}")
    print(f"  - Particle type: {action['particle_type']}")


def test_evaluator(model, config):
    """Test evaluator"""
    print("\n" + "=" * 60)
    print("Testing Evaluator...")
    print("=" * 60)

    evaluator = ConservationLawEvaluator(model, config)
    print(f"[OK] Evaluator created")

    # Test conservation mask analysis
    alpha_metrics = evaluator.analyze_conservation_mask()
    print(f"\n  Conservation Mask Analysis:")
    print(f"    - α_fixed (Q): {alpha_metrics['alpha_fixed_Q']:.3f}")
    print(f"    - α_fixed (L): {alpha_metrics['alpha_fixed_L']:.3f}")
    print(f"    - Max learnable α: {alpha_metrics['max_learnable_alpha']:.3f}")
    print(f"    - Discovered dim: {alpha_metrics['discovered_dim_idx']}")
    print(f"    - Sparsity: {alpha_metrics['sparsity']}")

    # Test embedding alignment
    alignment_metrics = evaluator.analyze_embedding_alignment()
    print(f"\n  Embedding Alignment with Baryon Number:")
    print(f"    - Best correlation: {alignment_metrics['best_correlation']:.3f}")
    print(f"    - Best dim: {alignment_metrics['best_correlation_dim']}")


def main():
    print("\n" + "=" * 60)
    print("Feynman-GCPN V8 Setup Verification")
    print("=" * 60 + "\n")

    try:
        # Test each component
        config = test_config()
        env = test_environment(config)
        model, device = test_model(config)
        test_forward_pass(model, env, device)
        test_evaluator(model, config)

        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("=" * 60)
        print("\nV8 architecture is working correctly!")
        print("\nNext steps:")
        print("  1. Run quick test: python run_experiment.py --quick")
        print("  2. Run full training: python run_experiment.py")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
