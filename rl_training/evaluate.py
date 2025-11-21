"""
Evaluate a trained Feynman-GCPN model
Generate diagrams and export them for visualization
"""

import sys
import os
import argparse
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feynman_env import FeynmanDiagramEnv
from models import FeynmanGCPN
from physics_engine import PhysicsConstants
from visualization_bridge import DiagramEvaluator, DiagramExporter


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained Feynman-GCPN')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--reaction', type=str, default='e+e->mu+mu',
                      help='Reaction to generate (format: particle1+particle2->particle3+particle4)')
    parser.add_argument('--num-episodes', type=int, default=10,
                      help='Number of diagrams to generate')
    parser.add_argument('--output-dir', type=str, default='evaluation',
                      help='Directory to save generated diagrams')
    parser.add_argument('--deterministic', action='store_true',
                      help='Use deterministic policy (argmax)')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (auto/cpu/cuda)')
    
    return parser.parse_args()


def parse_reaction(reaction_str):
    """Parse reaction string like 'e+e->mu+mu'"""
    if '->' not in reaction_str:
        raise ValueError("Reaction must contain '->' separator")
    
    initial_str, final_str = reaction_str.split('->')
    initial_particles = [p.strip() for p in initial_str.split('+')]
    final_particles = [p.strip() for p in final_str.split('+')]
    
    return initial_particles, final_particles


def main():
    args = parse_args()
    
    print("=" * 80)
    print("🔍 Feynman-GCPN Model Evaluation")
    print("=" * 80)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Device: {device}")
    
    # Parse reaction
    try:
        initial_state, final_state = parse_reaction(args.reaction)
        print(f"Reaction: {' + '.join(initial_state)} → {' + '.join(final_state)}")
    except Exception as e:
        print(f"❌ Invalid reaction format: {e}")
        return
    
    # Create environment
    print("\n📦 Creating environment...")
    env = FeynmanDiagramEnv(
        initial_state=initial_state,
        final_state=final_state,
        max_vertices=10,
        max_steps=50
    )
    
    # Create model
    print("🧠 Building model...")
    num_particle_types = len(PhysicsConstants.get_all_particles()) + len(PhysicsConstants.BOSONS)
    
    model = FeynmanGCPN(
        node_input_dim=9,
        edge_input_dim=21,
        hidden_dim=128,
        num_mp_layers=3,
        num_action_types=4,
        num_particle_types=num_particle_types,
        max_vertices=10,
        lambda_penalty=5.0
    )
    
    # Load checkpoint
    print(f"📂 Loading checkpoint: {args.checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"   ✅ Loaded successfully")
        print(f"   Training step: {checkpoint.get('global_step', 'unknown')}")
        print(f"   Best reward: {checkpoint.get('best_reward', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return
    
    # Create evaluator
    evaluator = DiagramEvaluator(model, env, device)
    
    # Generate diagrams
    print(f"\n🎨 Generating {args.num_episodes} diagrams...")
    print(f"   Mode: {'Deterministic' if args.deterministic else 'Stochastic'}")
    
    stats = evaluator.evaluate_multiple(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir
    )
    
    print(f"\n📊 Results saved to: {args.output_dir}/")
    print(f"   - Best diagram: best_diagram.json")
    print(f"   - Statistics: statistics.json")
    print(f"   - Individual diagrams: diagram_*.json")
    
    print("\n💡 Next steps:")
    print(f"   1. Open training_viz.html and point it to {args.output_dir}/best_diagram.json")
    print(f"   2. Import {args.output_dir}/best_diagram.json into Feynman Forge")
    print(f"   3. View statistics in {args.output_dir}/statistics.json")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
