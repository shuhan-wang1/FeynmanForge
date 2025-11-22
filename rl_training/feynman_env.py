"""
Feynman Diagram Gymnasium Environment
Implements the MDP formulation for diagram construction
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional, Any
import copy

from physics_engine import (
    PhysicsConstants, 
    ConservationLaws, 
    ParticleEncoder,
    AntiparticleHelper
)


class FeynmanDiagramEnv(gym.Env):
    """
    Gymnasium environment for constructing Feynman diagrams.
    
    State: Heterogeneous DAG represented as PyG Data object
    Action: Hierarchical discrete actions (Connect, Branch, SetType, Terminate)
    Reward: Physics-informed reward based on conservation laws
    """
    
    metadata = {'render_modes': ['human', 'json']}
    
    # Action types
    ACTION_CONNECT = 0
    ACTION_BRANCH = 1
    ACTION_SET_TYPE = 2
    ACTION_TERMINATE = 3
    ACTION_MERGE = 4  # NEW: Merge two vertices for annihilation topology
    
    def __init__(
        self, 
        initial_state: List[str],
        final_state: List[str],
        max_vertices: int = 10,
        max_steps: int = 50,
        canvas_width: int = 800,
        canvas_height: int = 600,
        reward_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        self.initial_particles = initial_state
        self.final_particles = final_state
        self.max_vertices = max_vertices
        self.max_steps = max_steps
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Reward weights - OPTIMIZED for better exploration
        self.reward_weights = reward_weights or {
            'charge_violation': -0.5,
            'lepton_violation': -0.5,
            'baryon_violation': -0.5,
            'color_violation': -1.0,
            'interaction_violation': -0.5,
            'target_match': 50.0,              # INCREASED from 20.0 - major success!
            'topology_valid': 20.0,            # INCREASED from 10.0
            'successful_connection': 5.0,      # INCREASED from 2.0 - encourage connections
            'vertex_created': 3.0,             # INCREASED from 1.0 - encourage building
            'conservation_bonus': 5.0,         # INCREASED from 2.0 - reward physics correctness
            'complexity_penalty': -0.1,
            'step_penalty': -0.02,             # CHANGED from 0.0 - tiny penalty to prefer efficiency
            'invalid_action': -0.2,            # REDUCED from -0.5 - less harsh on exploration
            'progress_reward': 2.0,            # NEW - reward incremental progress
            'exploration_bonus': 0.5,          # NEW - reward trying new things
            'early_termination_penalty': -50.0, # NEW - much harsher than exploring
        }
        
        self.num_particle_types = len(PhysicsConstants.get_all_particles()) + len(PhysicsConstants.BOSONS)
        
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(5),  # Updated from 4 to 5 for ACTION_MERGE
            'vertex_idx': spaces.Discrete(max_vertices),
            'particle_type': spaces.Discrete(self.num_particle_types),
            'target_vertex': spaces.Discrete(max_vertices)
        })
        
        self.observation_space = spaces.Dict({
            'num_nodes': spaces.Discrete(max_vertices * 2),
            'num_edges': spaces.Discrete(max_vertices * 4),
        })
        
        self.graph = None
        self.vertices = []
        self.edges = []
        self.step_count = 0
        self.terminated = False
        self.last_progress = 0.0  # Track progress for incremental rewards
        self.action_history = []  # NEW: Track all actions taken
        
        all_particles = [p.id for p in PhysicsConstants.get_all_particles()]
        all_bosons = list(PhysicsConstants.BOSONS.keys())
        self.particle_list = all_particles + all_bosons
        self.particle_to_idx = {p: i for i, p in enumerate(self.particle_list)}
        
    def reset(self, seed=None, options=None) -> Tuple[Data, Dict]:
        super().reset(seed=seed)

        self.step_count = 0
        self.terminated = False
        self.vertices = []
        self.edges = []
        self.last_progress = 0.0  # Reset progress tracking
        self.action_history = []  # Reset action history
        
        initial_x = 80
        y_spacing = self.canvas_height / (len(self.initial_particles) + 1)
        for i, particle_id in enumerate(self.initial_particles):
            vertex = {
                'id': len(self.vertices),
                'type': 'initial',
                'x': initial_x,
                'y': y_spacing * (i + 1),
                'connected_edges': []
            }
            self.vertices.append(vertex)
        
        final_x = self.canvas_width - 80
        for i, particle_id in enumerate(self.final_particles):
            vertex = {
                'id': len(self.vertices),
                'type': 'final',
                'x': final_x,
                'y': y_spacing * (i + 1),
                'connected_edges': []
            }
            self.vertices.append(vertex)
        
        def parse_particle_id(p_str):
            if p_str.endswith('_bar'):
                return p_str.replace('_bar', ''), True
            return p_str, False
        
        for i, particle_str in enumerate(self.initial_particles):
            p_id, is_anti = parse_particle_id(particle_str)
            edge = {
                'id': len(self.edges),
                'source': self.vertices[i]['id'],
                'target': None,
                'particle_id': p_id,
                'is_anti': is_anti,
                'color': None,
                'is_external': True,
                'state': 'open'
            }
            self.edges.append(edge)
            self.vertices[i]['connected_edges'].append(edge['id'])
        
        for i, particle_str in enumerate(self.final_particles):
            p_id, is_anti = parse_particle_id(particle_str)
            edge = {
                'id': len(self.edges),
                'source': None,
                'target': self.vertices[len(self.initial_particles) + i]['id'],
                'particle_id': p_id,
                'is_anti': is_anti,
                'color': None,
                'is_external': True,
                'state': 'open'
            }
            self.edges.append(edge)
            self.vertices[len(self.initial_particles) + i]['connected_edges'].append(edge['id'])
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: Dict) -> Tuple[Data, float, bool, bool, Dict]:
        if self.terminated:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.step_count += 1
        reward = self.reward_weights['step_penalty']
        action_type = action['action_type']

        # Record action in history for visualization
        action_names = ['CONNECT', 'BRANCH', 'SET_TYPE', 'TERMINATE', 'MERGE']
        action_record = {
            'step': self.step_count,
            'action_type': action_names[action_type],
            'action_type_idx': action_type,
            'vertex_idx': action.get('vertex_idx', None),
            'target_vertex': action.get('target_vertex', None),
            'particle_type': action.get('particle_type', None),
            'num_vertices_before': len(self.vertices),
            'num_edges_before': len(self.edges),
            'success': False,  # Will be updated below
            'reward': 0.0  # Will be updated at end
        }

        # DEBUG: Log actions for first 20 steps only (reduced spam)
        if self.step_count <= 20:
            print(f"[ENV Step {self.step_count}] Action: {action_names[action_type]}, vertex_idx={action.get('vertex_idx', 'N/A')}, num_vertices={len(self.vertices)}")

        if action_type == self.ACTION_TERMINATE:
            self.terminated = True
            action_record['success'] = True  # TERMINATE always succeeds
            num_internal_edges = sum(1 for e in self.edges if not e['is_external'])
            is_connected = self._is_graph_connected()

            # HARSH penalty for terminating too early (before making real progress)
            if self.step_count < 3:
                reward += self.reward_weights['early_termination_penalty']
            elif num_internal_edges < 1 or not is_connected:
                reward -= 40.0  # INCREASED from 20.0 - penalize lazy termination
            else:
                reward += self._compute_terminal_reward()

        elif action_type == self.ACTION_CONNECT:
            success = self._execute_connect(action['vertex_idx'], action['target_vertex'])
            action_record['success'] = success  # Record success
            if self.step_count <= 20:
                print(f"  → CONNECT success={success}, reward_before={reward:.2f}")
            if success:
                reward += self.reward_weights.get('successful_connection', 2.0)
                step_reward = self._compute_step_reward(action['vertex_idx'])
                reward += step_reward
                if step_reward >= 0:
                    reward += self.reward_weights.get('conservation_bonus', 0.5)
            else:
                reward += self.reward_weights.get('invalid_action', -0.5)

        elif action_type == self.ACTION_BRANCH:
            success = self._execute_branch(action['vertex_idx'], action['particle_type'])
            action_record['success'] = success  # Record success
            if self.step_count <= 20:
                print(f"  → BRANCH success={success}, num_vertices_after={len(self.vertices)}")
            if success:
                reward += self.reward_weights.get('vertex_created', 1.0)
                step_reward = self._compute_step_reward(action['vertex_idx'])
                reward += step_reward
                if step_reward >= 0:
                    reward += self.reward_weights.get('conservation_bonus', 0.5)
            else:
                reward += self.reward_weights.get('invalid_action', -0.5)
                
        elif action_type == self.ACTION_SET_TYPE:
            success = self._execute_set_type(action['vertex_idx'], action['particle_type'])
            action_record['success'] = success  # Record success
            if not success:
                reward += self.reward_weights.get('invalid_action', -0.5)

        elif action_type == self.ACTION_MERGE:
            success = self._execute_merge(action['vertex_idx'], action['target_vertex'], action['particle_type'])
            action_record['success'] = success  # Record success
            if success:
                reward += self.reward_weights.get('vertex_created', 1.0)
                step_reward = self._compute_step_reward(len(self.vertices) - 1)  # New vertex is last
                reward += step_reward
                if step_reward >= 0:
                    reward += self.reward_weights.get('conservation_bonus', 0.5)
            else:
                reward += self.reward_weights.get('invalid_action', -0.5)

        # Add progress-based reward for incremental improvement
        if not self.terminated:
            new_progress = self._compute_progress_score()
            progress_delta = new_progress - self.last_progress
            if progress_delta > 0:
                reward += progress_delta * self.reward_weights['progress_reward']
                reward += self.reward_weights['exploration_bonus']  # Bonus for making progress
            self.last_progress = new_progress

        truncated = self.step_count >= self.max_steps

        # Update action record with final values
        action_record['reward'] = reward
        action_record['num_vertices_after'] = len(self.vertices)
        action_record['num_edges_after'] = len(self.edges)
        self.action_history.append(action_record)

        # DEBUG: Log final reward and termination
        if self.step_count <= 20:
            print(f"  → Final reward={reward:.2f}, terminated={self.terminated}, truncated={truncated}")

        return self._get_observation(), reward, self.terminated, truncated, self._get_info()
    
    def _execute_connect(self, vertex_idx: int, target_idx: int) -> bool:
        """
        Connect two vertices.
        Supports:
        1. Interaction <-> Interaction (Creates new propagator)
        2. Initial/Final <-> Interaction (Extends external line, plugs into port)
        """
        if vertex_idx >= len(self.vertices) or target_idx >= len(self.vertices):
            return False
        if vertex_idx == target_idx:
            return False
        
        v1 = self.vertices[vertex_idx]
        v2 = self.vertices[target_idx]
        
        v1_open = self._get_open_halflines(vertex_idx)
        v2_open = self._get_open_halflines(target_idx)
        
        if not v1_open or not v2_open:
            return False
            
        edge1_idx = v1_open[0]
        edge2_idx = v2_open[0]
        edge1 = self.edges[edge1_idx]
        edge2 = self.edges[edge2_idx]
        
        v1_is_ext = v1['type'] in ['initial', 'final']
        v2_is_ext = v2['type'] in ['initial', 'final']
        
        # Case A: Both are Interaction Vertices -> Create new internal propagator
        if not v1_is_ext and not v2_is_ext:
            new_edge = {
                'id': len(self.edges),
                'source': vertex_idx,
                'target': target_idx,
                'particle_id': 'photon',  # Default, agent will set type
                'is_anti': False,
                'color': None,
                'is_external': False,
                'state': 'connected'
            }
            self.edges.append(new_edge)
            v1['connected_edges'].append(new_edge['id'])
            v2['connected_edges'].append(new_edge['id'])
            
            # Consume ports
            edge1['state'] = 'connected'
            edge2['state'] = 'connected'
            return True
            
        # Case B: Two external vertices -> Allow if both are same type (both initial or both final)
        # This enables pair annihilation or final state merging scenarios
        if v1_is_ext and v2_is_ext:
            # Only allow if both are initial OR both are final (not mixed)
            if v1['type'] != v2['type']:
                return False
            # If both are initial, they can converge - delegate to merge operation
            # For now, we'll use ACTION_MERGE for this, so return False here
            return False

        # Case C: One External, One Interaction -> Merge external line into port
            
        # Identify which is external and which is interaction
        if v1_is_ext:
            ext_v, int_v = v1, v2
            ext_edge, int_edge_idx = edge1, edge2_idx
            ext_v_idx = vertex_idx
        else:
            ext_v, int_v = v2, v1
            ext_edge, int_edge_idx = edge2, edge1_idx
            ext_v_idx = target_idx
            
        # Perform merge
        if ext_v['type'] == 'initial':
            # Initial line goes INTO the interaction vertex
            if ext_edge['target'] is not None: return False
            ext_edge['target'] = int_v['id']
        else:
            # Final line comes FROM the interaction vertex
            if ext_edge['source'] is not None: return False
            ext_edge['source'] = int_v['id']
            
        ext_edge['state'] = 'connected'
        
        # Add external edge to interaction vertex
        int_v['connected_edges'].append(ext_edge['id'])
        
        # Remove the consumed placeholder port from interaction vertex
        if int_edge_idx in int_v['connected_edges']:
            int_v['connected_edges'].remove(int_edge_idx)
            
        # Mark placeholder edge as consumed/dead
        self.edges[int_edge_idx]['state'] = 'consumed'
        
        return True
    
    def _execute_branch(self, vertex_idx: int, particle_type_idx: int) -> bool:
        """Create a new interaction vertex by branching (3-way vertex)"""
        if vertex_idx >= len(self.vertices): return False
        if len(self.vertices) >= self.max_vertices: return False
        
        open_lines = self._get_open_halflines(vertex_idx)
        if not open_lines: return False
        
        if particle_type_idx >= len(self.particle_list): return False
        emitted_pid = self.particle_list[particle_type_idx]
        
        # Get the incoming particle ID from the line we are branching
        edge_to_connect_idx = open_lines[0]
        edge_to_connect = self.edges[edge_to_connect_idx]
        incoming_pid = edge_to_connect['particle_id']
        incoming_is_anti = edge_to_connect['is_anti']
        
        new_vertex = {
            'id': len(self.vertices),
            'type': 'interaction',
            'x': self.vertices[vertex_idx]['x'] + 50,
            'y': self.vertices[vertex_idx]['y'],
            'connected_edges': []
        }
        self.vertices.append(new_vertex)
        
        # Create outgoing edge 1: The continuation of the incoming particle
        new_edge1 = {
            'id': len(self.edges),
            'source': new_vertex['id'],
            'target': None,
            'particle_id': incoming_pid,
            'is_anti': incoming_is_anti,
            'color': None,
            'is_external': False,
            'state': 'open'
        }
        self.edges.append(new_edge1)
        new_vertex['connected_edges'].append(new_edge1['id'])
        
        # Create outgoing edge 2: The emitted particle (e.g., photon)
        new_edge2 = {
            'id': len(self.edges),
            'source': new_vertex['id'],
            'target': None,
            'particle_id': emitted_pid,
            'is_anti': False,
            'color': None,
            'is_external': False,
            'state': 'open'
        }
        self.edges.append(new_edge2)
        new_vertex['connected_edges'].append(new_edge2['id'])
        
        # Connect input line
        if edge_to_connect['source'] is None:
            edge_to_connect['source'] = new_vertex['id']
        else:
            edge_to_connect['target'] = new_vertex['id']
        edge_to_connect['state'] = 'connected'
        new_vertex['connected_edges'].append(edge_to_connect['id'])
        
        return True
    
    def _execute_set_type(self, edge_idx: int, particle_type_idx: int) -> bool:
        if edge_idx >= len(self.edges): return False
        if particle_type_idx >= len(self.particle_list): return False

        # Only allow changing type of internal edges
        if self.edges[edge_idx]['is_external']: return False

        self.edges[edge_idx]['particle_id'] = self.particle_list[particle_type_idx]
        return True

    def _execute_merge(self, vertex_idx1: int, vertex_idx2: int, particle_type_idx: int) -> bool:
        """
        Merge two vertices into a new interaction vertex.
        This is the KEY operation for annihilation topology: e⁺ + e⁻ → [vertex] → γ

        Args:
            vertex_idx1: First vertex to merge
            vertex_idx2: Second vertex to merge
            particle_type_idx: Type of particle to emit from the new vertex

        Returns:
            True if merge was successful, False otherwise
        """
        # Validation checks
        if vertex_idx1 >= len(self.vertices) or vertex_idx2 >= len(self.vertices):
            return False
        if vertex_idx1 == vertex_idx2:
            return False
        if len(self.vertices) >= self.max_vertices:
            return False
        if particle_type_idx >= len(self.particle_list):
            return False

        v1 = self.vertices[vertex_idx1]
        v2 = self.vertices[vertex_idx2]

        # Get open half-lines from both vertices
        v1_open = self._get_open_halflines(vertex_idx1)
        v2_open = self._get_open_halflines(vertex_idx2)

        if not v1_open or not v2_open:
            return False

        # Get the edges to merge
        edge1_idx = v1_open[0]
        edge2_idx = v2_open[0]
        edge1 = self.edges[edge1_idx]
        edge2 = self.edges[edge2_idx]

        # Calculate position for new interaction vertex (midpoint)
        new_x = (v1['x'] + v2['x']) / 2
        new_y = (v1['y'] + v2['y']) / 2

        # Create new interaction vertex
        new_vertex = {
            'id': len(self.vertices),
            'type': 'interaction',
            'x': new_x,
            'y': new_y,
            'connected_edges': []
        }
        new_vertex_id = new_vertex['id']
        self.vertices.append(new_vertex)

        # Connect edge1 to new vertex
        # If edge1 has source=None, it's an incoming line, set source to new vertex
        # If edge1 has target=None, it's an outgoing line, set target to new vertex
        if edge1['target'] is None:
            edge1['target'] = new_vertex_id
        elif edge1['source'] is None:
            edge1['source'] = new_vertex_id
        else:
            # Edge is already connected on both ends, this shouldn't happen for open lines
            return False
        edge1['state'] = 'connected'
        new_vertex['connected_edges'].append(edge1_idx)

        # Connect edge2 to new vertex
        if edge2['target'] is None:
            edge2['target'] = new_vertex_id
        elif edge2['source'] is None:
            edge2['source'] = new_vertex_id
        else:
            # Edge is already connected on both ends
            return False
        edge2['state'] = 'connected'
        new_vertex['connected_edges'].append(edge2_idx)

        # Create outgoing propagator (the particle emitted from annihilation)
        emitted_pid = self.particle_list[particle_type_idx]
        new_edge = {
            'id': len(self.edges),
            'source': new_vertex_id,
            'target': None,
            'particle_id': emitted_pid,
            'is_anti': False,
            'color': None,
            'is_external': False,
            'state': 'open'
        }
        self.edges.append(new_edge)
        new_vertex['connected_edges'].append(new_edge['id'])

        return True
    
    def _get_open_halflines(self, vertex_idx: int) -> List[int]:
        open_lines = []
        for edge_id in self.vertices[vertex_idx]['connected_edges']:
            edge = self.edges[edge_id]
            if edge['state'] == 'open':
                open_lines.append(edge_id)
        return open_lines
    
    def _compute_step_reward(self, vertex_idx: int) -> float:
        """
        Compute physics-based reward for a vertex.

        CRITICAL FIX: Only validate COMPLETE vertices!
        A vertex is complete when all its halflines are connected (no open edges).
        Incomplete vertices should NOT be penalized - they're under construction!

        Example:
        - Step 1: BRANCH creates vertex with 1 incoming, 1 open outgoing → DON'T validate
        - Step 2: CONNECT closes the open line → NOW validate

        This prevents punishing the model for intermediate states during construction.
        """
        vertex = self.vertices[vertex_idx]
        # Filter out consumed edges
        connected_edges = [self.edges[eid] for eid in vertex['connected_edges'] if self.edges[eid]['state'] != 'consumed']

        # CRITICAL: Only validate COMPLETE vertices (no open halflines)
        open_halflines = [e for e in connected_edges if e['state'] == 'open']
        if len(open_halflines) > 0:
            # Vertex still under construction, don't validate yet
            # Return 0 (no reward, no penalty) until vertex is complete
            return 0.0

        # ADDITIONAL: Require minimum number of edges for a valid vertex
        # A physics vertex needs at least 2 edges (e.g., decay: 1 in, 1 out is minimum)
        # More typically 3+ (e.g., QED vertex: e- in, e- out, photon out)
        if len(connected_edges) < 2:
            # Not enough edges for a valid interaction vertex
            return 0.0

        # Vertex is COMPLETE - now validate physics!
        incoming = [e for e in connected_edges if e['target'] == vertex_idx]
        outgoing = [e for e in connected_edges if e['source'] == vertex_idx]

        reward = 0.0

        q_in = [self._get_charge(e) for e in incoming]
        q_out = [self._get_charge(e) for e in outgoing]
        l_in = [self._get_lepton(e) for e in incoming]
        l_out = [self._get_lepton(e) for e in outgoing]
        b_in = [self._get_baryon(e) for e in incoming]
        b_out = [self._get_baryon(e) for e in outgoing]
        colors_in = [e['color'] for e in incoming]
        colors_out = [e['color'] for e in outgoing]

        charge_ok, charge_mismatch = ConservationLaws.check_charge_conservation(q_in, q_out)
        lepton_ok, lepton_mismatch = ConservationLaws.check_lepton_conservation(l_in, l_out)
        baryon_ok, baryon_mismatch = ConservationLaws.check_baryon_conservation(b_in, b_out)
        color_ok, color_mismatch = ConservationLaws.check_color_conservation(colors_in, colors_out)

        if not charge_ok: reward += self.reward_weights['charge_violation'] * charge_mismatch
        if not lepton_ok: reward += self.reward_weights['lepton_violation'] * lepton_mismatch
        if not baryon_ok: reward += self.reward_weights['baryon_violation'] * baryon_mismatch
        if not color_ok: reward += self.reward_weights['color_violation'] * color_mismatch

        particle_ids = [e['particle_id'] for e in connected_edges]
        rules_ok, violations = ConservationLaws.check_interaction_rules(particle_ids)
        if not rules_ok: reward += self.reward_weights['interaction_violation'] * len(violations)

        if charge_ok and lepton_ok and baryon_ok and color_ok and rules_ok:
            reward += 2.0

        return reward
    
    def _compute_terminal_reward(self) -> float:
        reward = 0.0
        is_connected = self._is_graph_connected()
        no_dangling = self._no_dangling_internal_lines()

        if not (is_connected and no_dangling):
            reward -= 10.0
            return reward
        else:
            reward += self.reward_weights['topology_valid']

        initial_match = self._check_external_match(self.initial_particles, 'initial')
        final_match = self._check_external_match(self.final_particles, 'final')

        if initial_match and final_match:
            reward += self.reward_weights['target_match']
        else:
            reward -= 5.0

        # CRITICAL FIX: Check GLOBAL conservation laws across entire diagram
        # Previous bug: Only checked vertex-by-vertex, allowed extra particles (neutrinos)
        global_conservation_ok = self._check_global_conservation()
        if not global_conservation_ok:
            reward -= 100.0  # HUGE penalty for violating global conservation

        # REMOVED: Complexity penalty was discouraging model from building ANY topology!
        # num_interaction_vertices = sum(1 for v in self.vertices if v['type'] == 'interaction')
        # reward += self.reward_weights['complexity_penalty'] * num_interaction_vertices

        return reward

    def _compute_progress_score(self) -> float:
        """
        Compute a progress score (0-1) indicating how close the diagram is to completion.
        Used for incremental reward shaping.
        """
        score = 0.0

        # Component 1: Internal structure (max 30 points)
        num_internal = sum(1 for e in self.edges if not e['is_external'])
        score += min(num_internal * 5.0, 30.0)

        # Component 2: Connectivity (20 points)
        if len(self.vertices) > 0 and self._is_graph_connected():
            score += 20.0

        # Component 3: External particle connections (30 points)
        connected_external = sum(1 for e in self.edges
                                if e['is_external'] and e['state'] == 'connected')
        total_external = len(self.initial_particles) + len(self.final_particles)
        if total_external > 0:
            score += (connected_external / total_external) * 30.0

        # Component 4: Interaction vertices created (20 points)
        num_interaction = sum(1 for v in self.vertices if v['type'] == 'interaction')
        score += min(num_interaction * 10.0, 20.0)

        # Normalize to 0-1 range
        return score / 100.0

    def _check_global_conservation(self) -> bool:
        """
        CRITICAL: Check global conservation laws across the ENTIRE diagram.

        This prevents diagrams like mu+mu_bar->e+e_bar+neutrino which conserve
        charge locally at each vertex but violate global lepton number conservation.

        Returns:
            True if all global conservation laws are satisfied
        """
        # Collect all initial state particles (incoming to diagram)
        initial_edges = []
        for v in self.vertices:
            if v['type'] == 'initial':
                for edge_id in v['connected_edges']:
                    edge = self.edges[edge_id]
                    if edge['is_external']:
                        initial_edges.append(edge)

        # Collect all final state particles (outgoing from diagram)
        final_edges = []
        for v in self.vertices:
            if v['type'] == 'final':
                for edge_id in v['connected_edges']:
                    edge = self.edges[edge_id]
                    if edge['is_external']:
                        final_edges.append(edge)

        # Check global charge conservation
        total_charge_in = sum(self._get_charge(e) for e in initial_edges)
        total_charge_out = sum(self._get_charge(e) for e in final_edges)
        if abs(total_charge_in - total_charge_out) > 1e-6:
            return False

        # Check global lepton number conservation (CRITICAL for neutrino bug)
        total_lepton_in = sum(self._get_lepton(e) for e in initial_edges)
        total_lepton_out = sum(self._get_lepton(e) for e in final_edges)
        if abs(total_lepton_in - total_lepton_out) > 1e-6:
            return False

        # Check global baryon number conservation
        total_baryon_in = sum(self._get_baryon(e) for e in initial_edges)
        total_baryon_out = sum(self._get_baryon(e) for e in final_edges)
        if abs(total_baryon_in - total_baryon_out) > 1e-6:
            return False

        # Check that we have the EXACT particles we expect (no extras!)
        initial_count = len(initial_edges)
        final_count = len(final_edges)
        expected_initial = len(self.initial_particles)
        expected_final = len(self.final_particles)

        if initial_count != expected_initial or final_count != expected_final:
            return False

        return True

    def _check_external_match(self, target_particles: List[str], vertex_type: str) -> bool:
        external_vertices = [v for v in self.vertices if v['type'] == vertex_type]
        if len(external_vertices) != len(target_particles): return False
        
        def parse_particle(p_str):
            if p_str.endswith('_bar'): return p_str.replace('_bar', ''), True
            return p_str, False
        
        target_parsed = [parse_particle(p) for p in target_particles]
        
        found_particles = []
        for v in external_vertices:
            for edge_id in v['connected_edges']:
                edge = self.edges[edge_id]
                if edge['is_external']:
                    p_id = edge['particle_id']
                    if edge['is_anti']: p_id += '_bar'
                    found_particles.append(p_id)
        
        return sorted(found_particles) == sorted(target_particles)
    
    def _is_graph_connected(self) -> bool:
        if len(self.vertices) == 0: return True
        visited = set()
        queue = [0]
        visited.add(0)
        
        while queue:
            v_id = queue.pop(0)
            vertex = self.vertices[v_id]
            for edge_id in vertex['connected_edges']:
                edge = self.edges[edge_id]
                # Follow edge if not consumed
                if edge['state'] == 'consumed': continue
                
                neighbors = []
                if edge['source'] is not None: neighbors.append(edge['source'])
                if edge['target'] is not None: neighbors.append(edge['target'])
                
                for neighbor_id in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)
                        
        return len(visited) == len(self.vertices)
    
    def _no_dangling_internal_lines(self) -> bool:
        for edge in self.edges:
            if edge['state'] == 'consumed': continue
            if not edge['is_external']:
                if edge['source'] is None or edge['target'] is None:
                    return False
        return True
    
    def _get_observation(self) -> Data:
        """
        Build PyG Data object from current diagram state

        CRITICAL FIX: Include external edges as self-loops so graph is always connected.
        Without this, initial state has 0 edges and MPNN can't do message passing!
        """
        # Node features
        node_features = []
        for v in self.vertices:
            type_vec = [0, 0, 0]
            if v['type'] == 'initial': type_vec[0] = 1
            elif v['type'] == 'final': type_vec[1] = 1
            elif v['type'] == 'interaction': type_vec[2] = 1

            x_norm = v['x'] / self.canvas_width
            y_norm = v['y'] / self.canvas_height
            num_conn = len(v['connected_edges'])

            q_net = 0.0; l_net = 0.0; b_net = 0.0
            for edge_id in v['connected_edges']:
                edge = self.edges[edge_id]
                if edge['state'] == 'consumed': continue
                is_incoming = (edge['target'] == v['id'])
                sign = 1.0 if is_incoming else -1.0
                q_net += sign * self._get_charge(edge)
                l_net += sign * self._get_lepton(edge)
                b_net += sign * self._get_baryon(edge)

            node_features.append(type_vec + [x_norm, y_norm, num_conn, q_net, l_net, b_net])

        edge_index = []
        edge_features = []

        # Add connected edges (internal propagators)
        for edge in self.edges:
            if edge['state'] == 'consumed': continue
            if edge['source'] is not None and edge['target'] is not None:
                edge_index.append([edge['source'], edge['target']])
                edge_features.append(ParticleEncoder.encode_particle(
                    edge['particle_id'], edge['is_anti'], edge['color']
                ))

        # CRITICAL FIX: Add self-loops for external edges (initial/final particles)
        # This ensures graph is always connected so MPNN can propagate information
        for edge in self.edges:
            if edge['state'] == 'consumed': continue
            if edge['is_external']:
                # External edge with only source (initial particle going out)
                if edge['source'] is not None and edge['target'] is None:
                    vertex_id = edge['source']
                    edge_index.append([vertex_id, vertex_id])  # Self-loop
                    edge_features.append(ParticleEncoder.encode_particle(
                        edge['particle_id'], edge['is_anti'], edge['color']
                    ))
                # External edge with only target (final particle coming in)
                elif edge['source'] is None and edge['target'] is not None:
                    vertex_id = edge['target']
                    edge_index.append([vertex_id, vertex_id])  # Self-loop
                    edge_features.append(ParticleEncoder.encode_particle(
                        edge['particle_id'], edge['is_anti'], edge['color']
                    ))

        x = torch.tensor(np.array(node_features, dtype=np.float32), dtype=torch.float32)
        if len(edge_index) > 0:
            edge_index_tensor = torch.tensor(np.array(edge_index, dtype=np.int64), dtype=torch.long).t().contiguous()
            edge_attr = torch.from_numpy(np.stack(edge_features)).float()
        else:
            # Fallback: At minimum, add self-loops to all vertices with dummy features
            edge_index_list = [[i, i] for i in range(len(self.vertices))]
            edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            # Use small random features instead of zeros (helps MPNN initialization)
            edge_attr = torch.randn((len(self.vertices), 21), dtype=torch.float32) * 0.01

        return Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr)

    def _get_info(self) -> Dict:
        return {
            'num_vertices': len(self.vertices),
            'num_edges': len(self.edges),
            'step_count': self.step_count,
            'is_terminated': self.terminated
        }
        
    def _get_charge(self, edge: Dict) -> float:
        p = PhysicsConstants.get_particle_by_id(edge['particle_id'])
        b = PhysicsConstants.get_boson_by_id(edge['particle_id'])
        charge = p.charge if p else (b.charge if b else 0.0)
        return -charge if edge['is_anti'] else charge

    def _get_lepton(self, edge: Dict) -> float:
        p = PhysicsConstants.get_particle_by_id(edge['particle_id'])
        lepton = p.lepton if p else 0.0
        return -lepton if edge['is_anti'] else lepton

    def _get_baryon(self, edge: Dict) -> float:
        p = PhysicsConstants.get_particle_by_id(edge['particle_id'])
        baryon = p.baryon if p else 0.0
        return -baryon if edge['is_anti'] else baryon

    def render(self, mode='human'):
        if mode == 'human':
            print(f"\n=== Step {self.step_count} ===")
            for i, v in enumerate(self.vertices):
                print(f"  V{i}: {v['type']} at ({v['x']:.0f}, {v['y']:.0f})")
            for i, e in enumerate(self.edges):
                if e['state'] != 'consumed':
                    print(f"  E{i}: {e['particle_id']} {e['source']}->{e['target']}")

    def get_diagram_json(self) -> List[Dict]:
        shapes = []
        for edge in self.edges:
            if edge['state'] == 'consumed': continue
            
            # Get coordinates (handle None for external lines)
            if edge['source'] is not None:
                source_v = self.vertices[edge['source']]
                p1 = {'x': source_v['x'], 'y': source_v['y']}
            else:
                # Start of diagram (approximate)
                p1 = {'x': 0, 'y': 300} 

            if edge['target'] is not None:
                target_v = self.vertices[edge['target']]
                p2 = {'x': target_v['x'], 'y': target_v['y']}
            else:
                # End of diagram (approximate or dangling)
                p2 = {'x': p1['x'] + 100, 'y': p1['y'] + 50} 
            
            p = PhysicsConstants.get_particle_by_id(edge['particle_id'])
            shape_type = 'fermion' if p else edge['particle_id']
            if shape_type in ['w_plus', 'w_minus']: shape_type = 'boson_w'
            elif shape_type == 'z': shape_type = 'boson_z'
            
            # Reverse drawing direction for antiparticles
            if edge['is_anti'] and shape_type == 'fermion':
                p1, p2 = p2, p1
            
            shapes.append({
                'id': edge['id'],
                'type': shape_type,
                'p1': p1, 'p2': p2,
                'props': {
                    'particleId': edge['particle_id'],
                    'isAnti': edge['is_anti'],
                    'color': edge['color'],
                    'category': 'fermion' if p else 'boson',
                    'group': self._get_particle_group(edge['particle_id'])
                }
            })
        return shapes

    def _get_particle_group(self, particle_id: str) -> str:
        if PhysicsConstants.is_lepton(particle_id): return 'lepton'
        if particle_id in ['u', 'c', 't']: return 'quark_u'
        if particle_id in ['d', 's', 'b']: return 'quark_d'
        return particle_id