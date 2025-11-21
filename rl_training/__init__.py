"""
Feynman-GCPN: Physics-Informed Reinforcement Learning for Feynman Diagram Generation

This package implements a complete RL training pipeline for generating valid Feynman diagrams
using a Graph Convolutional Policy Network with a differentiable Physics Gate.

Key components:
- physics_engine: Particle database and conservation law validators
- feynman_env: Gymnasium environment (MDP formulation)
- models: MPNN encoder + Physics-Gated Policy Head
- training: PPO trainer
- visualization_bridge: Export diagrams to JSON for JavaScript visualization

Example usage:
    >>> from feynman_env import FeynmanDiagramEnv
    >>> from models import FeynmanGCPN
    >>> from training import PPOTrainer
    >>> 
    >>> env = FeynmanDiagramEnv(initial_state=['e', 'e'], final_state=['mu', 'mu'])
    >>> model = FeynmanGCPN()
    >>> trainer = PPOTrainer(env, model)
    >>> trainer.train(total_timesteps=100000)
"""

__version__ = '1.0.0'
__author__ = 'Shuhan Wang'

from .physics_engine import (
    PhysicsConstants,
    ConservationLaws,
    ParticleEncoder,
    Particle,
    Boson
)

from .feynman_env import FeynmanDiagramEnv

from .models import (
    FeynmanGCPN,
    FeynmanMPNN,
    PhysicsGate,
    PhysicsGatedPolicyHead,
    ValueHead
)

from .training import PPOTrainer, RolloutBuffer

from .visualization_bridge import (
    DiagramExporter,
    DiagramEvaluator,
    LiveMonitor,
    create_reaction_config
)

__all__ = [
    # Physics engine
    'PhysicsConstants',
    'ConservationLaws',
    'ParticleEncoder',
    'Particle',
    'Boson',
    
    # Environment
    'FeynmanDiagramEnv',
    
    # Models
    'FeynmanGCPN',
    'FeynmanMPNN',
    'PhysicsGate',
    'PhysicsGatedPolicyHead',
    'ValueHead',
    
    # Training
    'PPOTrainer',
    'RolloutBuffer',
    
    # Visualization
    'DiagramExporter',
    'DiagramEvaluator',
    'LiveMonitor',
    'create_reaction_config'
]
