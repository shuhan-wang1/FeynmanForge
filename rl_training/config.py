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
    total_steps: int = 500_000
    batch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clip parameter
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus

    # ==================== Model Architecture ====================
    hidden_dim: int = 128
    num_mp_layers: int = 3  # Message passing layers

    # Split Embedding (PQNE)
    fixed_dim: int = 2  # Dimensions for Q, L (Known laws)
    learnable_dim: int = 6  # Dimensions for discovery (Unknown laws)
    total_embedding_dim: int = 8  # fixed_dim + learnable_dim

    # Physics Gate
    physics_penalty: float = 5.0  # λ in paper
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
                'charge_violation': -2.0,
                'lepton_violation': -2.0,

                # Unknown laws (Baryon): NO immediate feedback
                # Only checked at terminal state for sparse_global_reward

                # Other rewards
                'color_violation': -1.0,
                'interaction_violation': -0.5,
                'target_match': 20.0,
                'topology_valid': 10.0,
                'successful_connection': 1.0,
                'vertex_created': 0.5,
                'step_penalty': -0.01,
                'invalid_action': -1.0,
                'complexity_penalty': -0.1,

                # SPARSE GLOBAL REWARD (terminal only, includes B check)
                'sparse_global_reward': 50.0,
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
        Testing Set: Scattering reactions (2→N)

        Tests generalization to NEW topologies not seen during training
        """
        return [
            # Annihilation
            # --- 1. 纯轻子过程 (B=0, L守恒) ---
        # 基础对照组，AI 容易学会
        {
            'name': 'muon_decay',
            'initial': ['mu'],
            'final': ['e', 'nu_e_bar', 'nu_mu']
        },
        
        # --- 2. 强子产生过程 (Net B=0) ---
        # AI 从这里学会夸克总是成对产生 (u + u_bar)
        {
            'name': 'z_to_quarks',
            'initial': ['z'],
            'final': ['u', 'u_bar']
        },

        # --- 3. 关键：带重子数的衰变 (Net B=1/3) ---
        # 这是“发现”重子数守恒的核心！
        # 如果 AI 试图把 t 变成 leptons，它就会因为违反 B 守恒而失去最终大奖
        {
            'name': 'top_decay',
            'initial': ['t'],
            'final': ['b', 'w_plus']
        },

        # --- 4. 关键：夸克变味衰变 (Net B=1/3) ---
        # 模拟中子衰变 (d -> u + W^-)
        # AI 需要学会构建 d -> u -> W -> (e + nu) 的图结构
        {
            'name': 'neutron_beta_decay_quark_level',
            'initial': ['d'],
            'final': ['u', 'e', 'nu_e_bar']
        }
        ]


@dataclass
class QuickTestConfig(Config):
    """Quick test configuration for debugging"""
    total_steps: int = 10_000
    batch_size: int = 16
    hidden_dim: int = 64
    num_mp_layers: int = 2
    eval_interval: int = 2_000
    checkpoint_interval: int = 5_000


# Default configuration
DEFAULT_CONFIG = Config()
