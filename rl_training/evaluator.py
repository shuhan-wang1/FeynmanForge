"""
Evaluator for Conservation Law Discovery Analysis (V8)

Analyzes the model's learned embeddings and conservation masks to determine
if it has successfully "discovered" Baryon number conservation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path

from models import FeynmanGCPN
from feynman_env import FeynmanDiagramEnv
from physics_engine import PhysicsConstants
from particle_utils import get_baryon_number_particles, parse_particle_string
from config import Config


class ConservationLawEvaluator:
    """
    Evaluates whether the model has discovered Baryon number conservation

    Key metrics:
    1. α_learnable values: Which learnable dimensions have high confidence?
    2. Embedding alignment: Do learnable embeddings align with Baryon number?
    3. Baryon violation rate: Does the model avoid violating B in generated diagrams?
    """

    def __init__(self, model: FeynmanGCPN, config: Config):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

    def evaluate_discovery(self) -> Dict:
        """
        Comprehensive evaluation of conservation law discovery

        Returns:
            Dictionary with discovery metrics
        """
        metrics = {}

        # 1. Analyze conservation mask α
        alpha_metrics = self.analyze_conservation_mask()
        metrics.update(alpha_metrics)

        # 2. Analyze embedding alignment with Baryon number
        alignment_metrics = self.analyze_embedding_alignment()
        metrics.update(alignment_metrics)

        # 3. Test on reactions
        reaction_metrics = self.test_on_reactions()
        metrics.update(reaction_metrics)

        return metrics

    def analyze_conservation_mask(self) -> Dict:
        """
        Analyze the learned conservation mask α

        Returns:
            - alpha_fixed: Values for Q, L (should be ~1.0)
            - alpha_learnable: Values for learnable dims (one should be high for B)
            - max_learnable_alpha: Highest learnable α value
            - discovered_dim_idx: Index of dimension with highest α (candidate for B)
        """
        with torch.no_grad():
            alpha = self.model.conservation_mask()

        fixed_dim = self.config.fixed_dim
        alpha_fixed = alpha[:fixed_dim].cpu().numpy()
        alpha_learnable = alpha[fixed_dim:].cpu().numpy()

        max_idx = np.argmax(alpha_learnable)
        max_val = alpha_learnable[max_idx]

        return {
            'alpha_fixed_Q': float(alpha_fixed[0]),
            'alpha_fixed_L': float(alpha_fixed[1]),
            'alpha_learnable': alpha_learnable.tolist(),
            'max_learnable_alpha': float(max_val),
            'discovered_dim_idx': int(max_idx),
            'sparsity': float(np.sum(alpha_learnable > 0.5))  # How many dims are "active"?
        }

    def analyze_embedding_alignment(self) -> Dict:
        """
        Analyze if learnable embeddings align with Baryon number

        Method:
        1. Get embeddings for all particles
        2. Extract the "discovered" dimension (highest α_learnable)
        3. Compute correlation between discovered_dim and true Baryon number

        High correlation indicates successful discovery!
        """
        with torch.no_grad():
            alpha = self.model.conservation_mask()
            fixed_dim = self.config.fixed_dim
            alpha_learnable = alpha[fixed_dim:].cpu().numpy()

            # Find discovered dimension
            discovered_dim_idx = np.argmax(alpha_learnable)
            discovered_dim_global = fixed_dim + discovered_dim_idx

        # Get all particle embeddings and Baryon numbers
        particle_ids = [p.id for p in PhysicsConstants.get_all_particles()] + \
                       list(PhysicsConstants.BOSONS.keys())

        embeddings_list = []
        baryon_numbers = []

        for pid in particle_ids:
            # Get embedding
            with torch.no_grad():
                idx = torch.tensor([particle_ids.index(pid)], device=self.device)
                emb = self.model.particle_embedding(idx)[0]
                embeddings_list.append(emb.cpu().numpy())

            # Get true Baryon number
            p = PhysicsConstants.get_particle_by_id(pid)
            b = PhysicsConstants.get_boson_by_id(pid)
            baryon = p.baryon if p else (b.baryon if b else 0.0)
            baryon_numbers.append(baryon)

        embeddings = np.array(embeddings_list)  # [num_particles, embedding_dim]
        baryon_numbers = np.array(baryon_numbers)

        # Correlation between discovered dimension and Baryon number
        discovered_values = embeddings[:, discovered_dim_global]
        correlation = np.corrcoef(discovered_values, baryon_numbers)[0, 1]

        # Also check correlation for ALL learnable dims
        correlations_all = []
        for i in range(self.config.learnable_dim):
            dim_idx = fixed_dim + i
            dim_values = embeddings[:, dim_idx]
            corr = np.corrcoef(dim_values, baryon_numbers)[0, 1]
            correlations_all.append(corr)

        return {
            'discovered_dim_baryon_correlation': float(correlation),
            'all_learnable_baryon_correlations': [float(c) for c in correlations_all],
            'best_correlation': float(max(correlations_all, key=abs)),
            'best_correlation_dim': int(np.argmax([abs(c) for c in correlations_all]))
        }

    def test_on_reactions(self) -> Dict:
        """
        Test the model on training and testing reactions

        Metrics:
        - Success rate on hadronic reactions (requires B conservation)
        - Success rate on leptonic reactions (no B constraint)
        - Baryon violation rate
        """
        training_reactions = self.config.get_training_reactions()
        testing_reactions = self.config.get_testing_reactions()

        train_hadronic_success = 0
        train_leptonic_success = 0
        train_baryon_violations = 0

        # Test on training set
        for reaction in training_reactions:
            env = FeynmanDiagramEnv(
                initial_state=reaction['initial'],
                final_state=reaction['final'],
                max_vertices=self.config.max_vertices,
                max_steps=self.config.max_steps
            )

            success, violated_baryon = self._rollout_and_check(env)

            # Categorize
            has_quarks = any(pid in get_baryon_number_particles()
                           for pid in reaction['initial'] + reaction['final'])

            if has_quarks:
                if success:
                    train_hadronic_success += 1
                if violated_baryon:
                    train_baryon_violations += 1
            else:
                if success:
                    train_leptonic_success += 1

        num_hadronic_train = sum(1 for r in training_reactions
                                 if any(pid in get_baryon_number_particles()
                                       for pid in r['initial'] + r['final']))
        num_leptonic_train = len(training_reactions) - num_hadronic_train

        # Test on testing set
        test_success = 0
        test_baryon_violations = 0

        for reaction in testing_reactions:
            env = FeynmanDiagramEnv(
                initial_state=reaction['initial'],
                final_state=reaction['final'],
                max_vertices=self.config.max_vertices,
                max_steps=self.config.max_steps
            )

            success, violated_baryon = self._rollout_and_check(env)
            if success:
                test_success += 1
            if violated_baryon:
                test_baryon_violations += 1

        return {
            'train_hadronic_success_rate': train_hadronic_success / max(num_hadronic_train, 1),
            'train_leptonic_success_rate': train_leptonic_success / max(num_leptonic_train, 1),
            'train_baryon_violation_rate': train_baryon_violations / len(training_reactions),
            'test_success_rate': test_success / len(testing_reactions),
            'test_baryon_violation_rate': test_baryon_violations / len(testing_reactions)
        }

    def _rollout_and_check(self, env: FeynmanDiagramEnv, max_steps: int = 50) -> Tuple[bool, bool]:
        """
        Rollout one episode and check if it's successful and conserves Baryon number

        Returns:
            (success, violated_baryon)
        """
        obs, _ = env.reset()
        done = False
        steps = 0

        with torch.no_grad():
            while not done and steps < max_steps:
                action = self.model.get_action(obs.to(self.device), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

        # Check if successful (all conservation laws including B)
        success = env._check_global_conservation() if done else False

        # Check if Baryon was violated (even if other laws were okay)
        baryon_violated = not self._check_baryon_conservation(env)

        return success, baryon_violated

    def _check_baryon_conservation(self, env: FeynmanDiagramEnv) -> bool:
        """Check if Baryon number is conserved in the final diagram"""
        b_initial = []
        b_final = []

        for edge in env.edges:
            if not edge['is_external']:
                continue

            p = PhysicsConstants.get_particle_by_id(edge['particle_id'])
            boson = PhysicsConstants.get_boson_by_id(edge['particle_id'])

            if p:
                baryon = -p.baryon if edge['is_anti'] else p.baryon
            elif boson:
                baryon = boson.baryon
            else:
                baryon = 0.0

            vertex_id = edge['source'] if edge['source'] is not None else edge['target']
            if vertex_id is None:
                continue

            vertex = env.vertices[vertex_id]

            if vertex['type'] == 'initial':
                b_initial.append(baryon)
            elif vertex['type'] == 'final':
                b_final.append(baryon)

        from physics_engine import ConservationLaws
        baryon_ok, _ = ConservationLaws.check_baryon_conservation(b_initial, b_final)
        return baryon_ok

    def save_analysis(self, metrics: Dict, output_path: str):
        """Save analysis results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Analysis saved to {output_path}")

    def print_discovery_summary(self, metrics: Dict):
        """Print human-readable summary of discovery results"""
        print("\n" + "=" * 80)
        print("CONSERVATION LAW DISCOVERY ANALYSIS")
        print("=" * 80)

        print("\n1. Conservation Mask α:")
        print(f"   α_fixed (Q, L): {metrics['alpha_fixed_Q']:.3f}, {metrics['alpha_fixed_L']:.3f}")
        print(f"   α_learnable: {[f'{x:.3f}' for x in metrics['alpha_learnable']]}")
        print(f"   Discovered dimension: {metrics['discovered_dim_idx']} (α = {metrics['max_learnable_alpha']:.3f})")
        print(f"   Sparsity: {metrics['sparsity']:.0f} active dimensions")

        print("\n2. Embedding Alignment with Baryon Number:")
        print(f"   Correlation (discovered dim ↔ B): {metrics['discovered_dim_baryon_correlation']:.3f}")
        print(f"   Best correlation: {metrics['best_correlation']:.3f} (dim {metrics['best_correlation_dim']})")

        print("\n3. Performance on Reactions:")
        print(f"   Hadronic reactions (train): {metrics['train_hadronic_success_rate']:.1%}")
        print(f"   Leptonic reactions (train): {metrics['train_leptonic_success_rate']:.1%}")
        print(f"   Baryon violation rate (train): {metrics['train_baryon_violation_rate']:.1%}")
        print(f"   Test success rate: {metrics['test_success_rate']:.1%}")
        print(f"   Baryon violation rate (test): {metrics['test_baryon_violation_rate']:.1%}")

        print("\n" + "=" * 80)

        #判断是否成功发现
        discovered = (
            metrics['max_learnable_alpha'] > 0.7 and
            abs(metrics['discovered_dim_baryon_correlation']) > 0.8 and
            metrics['train_baryon_violation_rate'] < 0.1
        )

        if discovered:
            print("✓ DISCOVERY SUCCESSFUL: The model has learned Baryon conservation!")
        else:
            print("✗ Discovery incomplete. Model has not fully learned Baryon conservation.")

        print("=" * 80 + "\n")
