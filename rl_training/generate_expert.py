"""
generate_expert.py
Generates expert trajectories for Supervised Pre-training.
Teaches the model how to build e+e- -> mu+mu- via s-channel.
"""
import torch
import numpy as np
from feynman_env import FeynmanDiagramEnv
from physics_engine import PhysicsConstants
import json


def generate_expert_trajectory():
    """
    Generate ONE expert trajectory for e+e- -> mu+mu-

    Returns trajectory as list of (state, action) pairs
    """
    # Initialize environment
    env = FeynmanDiagramEnv(
        initial_state=['e', 'e_bar'],  # e-, e+
        final_state=['mu', 'mu_bar'],   # mu-, mu+
        max_vertices=10,
        max_steps=50
    )

    trajectory = []
    state, info = env.reset()

    print(f"Initial vertices: {len(env.vertices)}")
    for i, v in enumerate(env.vertices):
        print(f"  V{i}: type={v['type']}, edges={v['connected_edges']}")

    # Map particle names to indices
    photon_idx = env.particle_to_idx['photon']
    mu_idx = env.particle_to_idx['mu']

    # Initial state:
    # V0: initial (e-)
    # V1: initial (e+)
    # V2: final (mu-)
    # V3: final (mu+)

    # ========== STEP 1: MERGE e- and e+ ==========
    # This creates an interaction vertex where e+ and e- annihilate
    # producing a virtual photon
    print("\n=== STEP 1: MERGE(0, 1, photon) ===")
    action_1 = {
        'action_type': env.ACTION_MERGE,
        'vertex_idx': 0,           # e-
        'target_vertex': 1,        # e+
        'particle_type': photon_idx
    }

    trajectory.append((state.clone(), action_1.copy()))
    state, reward, terminated, truncated, info = env.step(action_1)

    print(f"  Reward: {reward:.2f}")
    print(f"  Vertices after: {len(env.vertices)}")
    for i, v in enumerate(env.vertices):
        print(f"    V{i}: type={v['type']}, edges={v['connected_edges']}")

    # New vertex should be V4 (the annihilation vertex)
    annihilation_vertex = 4

    # ========== STEP 2: BRANCH from annihilation vertex ==========
    # The photon from MERGE creates an open edge
    # BRANCH will create: photon -> photon + mu
    # (We'll fix the second photon to mu_bar using SET_TYPE)
    print("\n=== STEP 2: BRANCH(4, mu) ===")
    action_2 = {
        'action_type': env.ACTION_BRANCH,
        'vertex_idx': annihilation_vertex,
        'target_vertex': 0,        # Not used for BRANCH
        'particle_type': mu_idx
    }

    trajectory.append((state.clone(), action_2.copy()))
    state, reward, terminated, truncated, info = env.step(action_2)

    print(f"  Reward: {reward:.2f}")
    print(f"  Vertices after: {len(env.vertices)}")
    for i, v in enumerate(env.vertices):
        print(f"    V{i}: type={v['type']}, edges={v['connected_edges']}")

    # New vertex V5 created with:
    # - photon in
    # - photon out (open) - THIS NEEDS TO BE CHANGED TO MU
    # - mu out (open)
    pair_production_vertex = 5

    # Find the edge that needs to be changed (photon out, state=open)
    v5 = env.vertices[pair_production_vertex]
    photon_edge_to_change = None

    for edge_id in v5['connected_edges']:
        edge = env.edges[edge_id]
        if (edge['particle_id'] == 'photon' and
            edge['state'] == 'open' and
            edge['source'] == pair_production_vertex):
            photon_edge_to_change = edge_id
            break

    print(f"  Edge to change: {photon_edge_to_change}")

    # ========== STEP 3: SET_TYPE to change photon -> mu ==========
    print("\n=== STEP 3: SET_TYPE(edge={photon_edge_to_change}, mu) ===")
    if photon_edge_to_change is not None:
        action_3 = {
            'action_type': env.ACTION_SET_TYPE,
            'vertex_idx': photon_edge_to_change,  # Uses vertex_idx slot for edge_idx
            'target_vertex': 0,
            'particle_type': mu_idx
        }

        trajectory.append((state.clone(), action_3.copy()))
        state, reward, terminated, truncated, info = env.step(action_3)

        print(f"  Reward: {reward:.2f}")

    # Now V5 has: photon in, mu out (open), mu out (open)
    # We need to connect these to the final states

    # ========== STEP 4: CONNECT V5 to final mu- (V2) ==========
    print("\n=== STEP 4: CONNECT(5, 2) ===")
    action_4 = {
        'action_type': env.ACTION_CONNECT,
        'vertex_idx': pair_production_vertex,
        'target_vertex': 2,  # final mu-
        'particle_type': 0   # Not used for CONNECT
    }

    trajectory.append((state.clone(), action_4.copy()))
    state, reward, terminated, truncated, info = env.step(action_4)

    print(f"  Reward: {reward:.2f}")

    # ========== STEP 5: CONNECT V5 to final mu+ (V3) ==========
    print("\n=== STEP 5: CONNECT(5, 3) ===")
    action_5 = {
        'action_type': env.ACTION_CONNECT,
        'vertex_idx': pair_production_vertex,
        'target_vertex': 3,  # final mu+
        'particle_type': 0
    }

    trajectory.append((state.clone(), action_5.copy()))
    state, reward, terminated, truncated, info = env.step(action_5)

    print(f"  Reward: {reward:.2f}")

    # ========== STEP 6: TERMINATE ==========
    print("\n=== STEP 6: TERMINATE ===")
    action_6 = {
        'action_type': env.ACTION_TERMINATE,
        'vertex_idx': 0,
        'target_vertex': 0,
        'particle_type': 0
    }

    trajectory.append((state.clone(), action_6.copy()))
    state, reward, terminated, truncated, info = env.step(action_6)

    print(f"  Final reward: {reward:.2f}")
    print(f"  Terminated: {terminated}")
    print(f"  Total trajectory length: {len(trajectory)}")

    return trajectory


def generate_expert_dataset(num_episodes=100):
    """
    Generate multiple expert trajectories
    """
    print(f"Generating {num_episodes} expert trajectories...")

    all_data = []

    for ep in range(num_episodes):
        if ep % 10 == 0:
            print(f"\nEpisode {ep}/{num_episodes}")

        try:
            trajectory = generate_expert_trajectory()
            all_data.extend(trajectory)
        except Exception as e:
            print(f"  ⚠️  Episode {ep} failed: {e}")
            continue

    print(f"\n✅ Generated {len(all_data)} total (state, action) pairs")
    return all_data


if __name__ == "__main__":
    # Test single trajectory
    print("=" * 80)
    print("Testing Expert Trajectory Generation")
    print("=" * 80)

    trajectory = generate_expert_trajectory()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total steps: {len(trajectory)}")
    print("\nAction sequence:")
    for i, (state, action) in enumerate(trajectory):
        action_names = ['CONNECT', 'BRANCH', 'SET_TYPE', 'TERMINATE', 'MERGE']
        print(f"  Step {i+1}: {action_names[action['action_type']]}")
