"""
Configuration and Hyperparameters for Feynman-GCPN V8
Conservation Law Discovery Experiment
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Config:
    """Main configuration for V8 conservation law discovery"""

    # ==================== Training Setup ====================
    total_steps: int = 1000_000
    batch_size: int = 512  # 增大batch size: 32 -> 128 (提高GPU利用率)
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clip parameter
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus

    # ==================== Model Architecture ====================
    hidden_dim: int = 768
    num_mp_layers: int = 8  # Message passing layers

    # Split Embedding (PQNE)
    fixed_dim: int = 2  # Dimensions for Q, L (Known laws)
    learnable_dim: int = 6  # Dimensions for discovery (Unknown laws)
    total_embedding_dim: int = 8  # fixed_dim + learnable_dim

    # Physics Gate
    physics_penalty: float = 2.0  # λ in paper
    gate_temperature: float = 1.0

    # Conservation Law Discovery Mechanism (CLDM)
    sparsity_weight: float = 0.001  # Regularization on α_learnable

    # ==================== Environment ====================
    max_vertices: int = 10
    max_steps: int = 50

    # ==================== Hybrid Reward Structure (Scientist Reward) ====================
    # Known laws: Immediate feedback
    known_laws_reward: Dict[str, float] = None

    # Unknown laws: Sparse feedback
    sparse_global_reward: float = 50.0  # Reward for complete valid diagram

    # ==================== Analysis Configuration ====================
    eval_interval: int = 10_000  # Evaluate every N steps
    checkpoint_interval: int = 50_000  # Save checkpoint every N steps

    # Metrics to track discovery progress
    discovery_metrics: List[str] = None

    def __post_init__(self):
        """Initialize default reward weights and metrics"""
        if self.known_laws_reward is None:
            self.known_laws_reward = {
                # Known laws (Charge, Lepton): Immediate feedback
                'charge_conservation': 2.0,
                'lepton_conservation': 2.0,
                'charge_violation': -0.5,
                'lepton_violation': -0.5,

                # Unknown laws (Baryon): NO immediate feedback
                # Only checked at terminal state for sparse_global_reward

                # Other rewards
                'color_violation': -1.0,
                'interaction_violation': -0.5,
                'target_match': 20.0,
                'topology_valid': 20.0,
                'successful_connection': 2.0,
                'vertex_created': 0.5,
                'step_penalty': -0.2,
                'invalid_action': -1.0,
                'complexity_penalty': -0.1,

                # SPARSE GLOBAL REWARD (terminal only, includes B check)
                'sparse_global_reward': 100.0,
            }

        # Validate
        assert self.fixed_dim + self.learnable_dim == self.total_embedding_dim
        assert self.physics_penalty > 0
        assert 0 < self.learning_rate < 1

        if self.discovery_metrics is None:
            self.discovery_metrics = [
                'alpha_learnable_values',  # Track α_learnable over time
                'baryon_violation_rate',   # Does model violate B?
                'embedding_alignment',     # Is E_learnable aligned with B?
                'success_rate_hadronic',   # Success on quark reactions
                'success_rate_leptonic',   # Success on lepton reactions
            ]

    # ==================== Dataset Configurations ====================
    @staticmethod
    def get_training_reactions() -> List[Dict[str, List[str]]]:
        """
        Training Set: Decay reactions (1→N)

        Critical: Hadronic decays are ESSENTIAL for Baryon number discovery
        The model must see quark interactions to learn B conservation
        """
        return [
            # Leptonic Decays
            {
                'name': 'muon_decay',
                'initial': ['mu'],
                'final': ['e', 'nu_e_bar', 'nu_mu']
            },
            {
                'name': 'tau_decay',
                'initial': ['tau'],
                'final': ['mu', 'nu_mu_bar', 'nu_tau']
            },
            {
                'name': 'z_to_leptons',
                'initial': ['z'],
                'final': ['e', 'e_bar']
            },

            # Hadronic Decays (CRITICAL for B discovery)
            {
                'name': 'z_to_uu',
                'initial': ['z'],
                'final': ['u', 'u_bar']
            },
            {
                'name': 'z_to_dd',
                'initial': ['z'],
                'final': ['d', 'd_bar']
            },
            {
                'name': 'z_to_cc',
                'initial': ['z'],
                'final': ['c', 'c_bar']
            },
        ]

    @staticmethod
    def get_testing_reactions() -> List[Dict[str, List[str]]]:
        """
        ✅ BUG FIX 6: Testing Set with DIFFERENT reactions from training
        
        Tests generalization to NEW reactions not seen during training.
        Critical: Avoid overlap with training reactions!
        """
        return [
            # --- 1. 轻子衰变（训练集中没有的） ---
            {
                'name': 'pion_leptonic_decay',  # π^- → μ^- + ν_μ_bar (training中没有)
                'initial': ['mu_bar'],  # 用反muon模拟
                'final': ['e_bar', 'nu_mu', 'nu_e_bar']
            },
            
            #--- 2. 强子反应（训练集中没有的夸克组合） ---
            {
                'name': 'z_to_strange',  # Z → s + s_bar (training只有u, d, c)
                'initial': ['z'],
                'final': ['s', 's_bar']
            },
            {
                'name': 'z_to_bottom',  # Z → b + b_bar (training中没有)
                'initial': ['z'],
                'final': ['b', 'b_bar']
            },
            
            # --- 3. 关键：带重子数的衰变（训练集中NO top decay!） ---
            # 测试模型是否真正学会了B守恒
            {
                'name': 'top_decay',  # t → b + W^+ (Net B = 1/3)
                'initial': ['t'],
                'final': ['b', 'w_plus']
            },
            
            # --- 4. 弱相互作用夸克衰变（训练集中没有） ---
            # 测试flavor changing weak decay是否保持B守恒
            {
                'name': 'strange_decay',  # s → u + W^- (Net B = 1/3)
                'initial': ['s'],
                'final': ['u', 'w_minus']
            },
            
            # --- 5. W玻色子衰变（训练集中没有） ---
            {
                'name': 'w_plus_decay',  # W^+ → e^+ + ν_e
                'initial': ['w_plus'],
                'final': ['e_bar', 'nu_e']
            }
        ]


@dataclass
class QuickTestConfig(Config):
    """Quick test configuration for debugging"""
    total_steps: int = 10_000
    batch_size: int = 64  # 增大: 16 -> 64
    hidden_dim: int = 64
    num_mp_layers: int = 2
    eval_interval: int = 2000
    checkpoint_interval: int = 5000


# Default configuration
DEFAULT_CONFIG = Config()
