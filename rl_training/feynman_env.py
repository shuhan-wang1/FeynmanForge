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
        """
        Args:
            initial_state: List of initial particle IDs (e.g., ['e', 'e'])
            final_state: List of final particle IDs (e.g., ['e', 'e'])
            max_vertices: Maximum number of vertices allowed
            max_steps: Maximum steps per episode
            canvas_width: Width for spatial coordinates (for visualization)
            canvas_height: Height for spatial coordinates
            reward_weights: Custom weights for reward components
        """
        super().__init__()
        
        self.initial_particles = initial_state
        self.final_particles = final_state
        self.max_vertices = max_vertices
        self.max_steps = max_steps
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Reward weights
        self.reward_weights = reward_weights or {
            'charge_violation': -5.0,
            'lepton_violation': -5.0,
            'baryon_violation': -5.0,
            'color_violation': -10.0,
            'interaction_violation': -3.0,
            'target_match': 10.0,
            'topology_valid': 5.0,
            'complexity_penalty': -0.5,
            'step_penalty': -0.01
        }
        
        # Action space: (action_type, vertex_idx, particle_type_idx, target_vertex_idx)
        # We'll use a simplified discrete action space
        self.num_particle_types = len(PhysicsConstants.get_all_particles()) + len(PhysicsConstants.BOSONS)
        
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(4),  # Connect, Branch, SetType, Terminate
            'vertex_idx': spaces.Discrete(max_vertices),
            'particle_type': spaces.Discrete(self.num_particle_types),
            'target_vertex': spaces.Discrete(max_vertices)
        })
        
        # Observation space is handled by PyG Data object
        # We'll define bounds for compatibility
        self.observation_space = spaces.Dict({
            'num_nodes': spaces.Discrete(max_vertices * 2),
            'num_edges': spaces.Discrete(max_vertices * 4),
        })
        
        # Initialize state
        self.graph = None
        self.vertices = []  # List of vertex dicts
        self.edges = []  # List of edge dicts
        self.step_count = 0
        self.terminated = False
        
        # Particle ID to index mapping
        all_particles = [p.id for p in PhysicsConstants.get_all_particles()]
        all_bosons = list(PhysicsConstants.BOSONS.keys())
        self.particle_list = all_particles + all_bosons
        self.particle_to_idx = {p: i for i, p in enumerate(self.particle_list)}
        
    def reset(self, seed=None, options=None) -> Tuple[Data, Dict]:
        """Reset environment to initial state with external lines"""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.terminated = False
        
        # Create initial and final vertices
        self.vertices = []
        self.edges = []
        
        # Initial state vertices (left side)
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
        
        # Final state vertices (right side)
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
        
        # Create external lines (initially untyped)
        for i, particle_id in enumerate(self.initial_particles):
            edge = {
                'id': len(self.edges),
                'source': self.vertices[i]['id'],
                'target': None,  # Open half-line
                'particle_id': particle_id,
                'is_anti': False,
                'color': None,
                'is_external': True,
                'state': 'open'  # 'open', 'connected', 'internal'
            }
            self.edges.append(edge)
            self.vertices[i]['connected_edges'].append(edge['id'])
        
        for i, particle_id in enumerate(self.final_particles):
            edge = {
                'id': len(self.edges),
                'source': None,  # Open half-line
                'target': self.vertices[len(self.initial_particles) + i]['id'],
                'particle_id': particle_id,
                'is_anti': False,
                'color': None,
                'is_external': True,
                'state': 'open'
            }
            self.edges.append(edge)
            self.vertices[len(self.initial_particles) + i]['connected_edges'].append(edge['id'])
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: Dict) -> Tuple[Data, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Dictionary with action_type, vertex_idx, particle_type, target_vertex
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.terminated:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        self.step_count += 1
        reward = self.reward_weights['step_penalty']  # Small penalty per step
        
        action_type = action['action_type']
        
        if action_type == self.ACTION_TERMINATE:
            # Terminate episode
            self.terminated = True
            terminal_reward = self._compute_terminal_reward()
            reward += terminal_reward
            
        elif action_type == self.ACTION_CONNECT:
            # Connect two open half-lines
            success = self._execute_connect(action['vertex_idx'], action['target_vertex'])
            if success:
                step_reward = self._compute_step_reward(action['vertex_idx'])
                reward += step_reward
            else:
                reward -= 1.0  # Penalty for invalid action
                
        elif action_type == self.ACTION_BRANCH:
            # Create a new vertex with branching
            success = self._execute_branch(action['vertex_idx'], action['particle_type'])
            if success:
                step_reward = self._compute_step_reward(action['vertex_idx'])
                reward += step_reward
            else:
                reward -= 1.0
                
        elif action_type == self.ACTION_SET_TYPE:
            # Set particle type for an edge
            success = self._execute_set_type(action['vertex_idx'], action['particle_type'])
            if not success:
                reward -= 0.5
        
        # Check if episode should terminate
        truncated = self.step_count >= self.max_steps
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, self.terminated, truncated, info
    
    def _execute_connect(self, vertex_idx: int, target_idx: int) -> bool:
        """
        Connect two vertices with a propagator
        Returns True if successful
        """
        if vertex_idx >= len(self.vertices) or target_idx >= len(self.vertices):
            return False
        
        if vertex_idx == target_idx:
            return False
        
        # Find open half-lines from both vertices
        v1_open = self._get_open_halflines(vertex_idx)
        v2_open = self._get_open_halflines(target_idx)
        
        if len(v1_open) == 0 or len(v2_open) == 0:
            return False
        
        # Connect the first available half-lines
        edge1 = self.edges[v1_open[0]]
        edge2 = self.edges[v2_open[0]]
        
        # Create a new internal edge connecting them
        new_edge = {
            'id': len(self.edges),
            'source': vertex_idx,
            'target': target_idx,
            'particle_id': 'photon',  # Default to photon, will be set later
            'is_anti': False,
            'color': None,
            'is_external': False,
            'state': 'connected'
        }
        
        self.edges.append(new_edge)
        self.vertices[vertex_idx]['connected_edges'].append(new_edge['id'])
        self.vertices[target_idx]['connected_edges'].append(new_edge['id'])
        
        # Mark old edges as connected
        edge1['state'] = 'connected'
        edge2['state'] = 'connected'
        
        return True
    
    def _execute_branch(self, vertex_idx: int, particle_type_idx: int) -> bool:
        """
        Create a new interaction vertex by branching from an existing half-line
        Example: e⁻ → e⁻ + γ
        """
        if vertex_idx >= len(self.vertices):
            return False
        
        if len(self.vertices) >= self.max_vertices:
            return False  # Max vertices reached
        
        # Find an open half-line from this vertex
        open_lines = self._get_open_halflines(vertex_idx)
        if len(open_lines) == 0:
            return False
        
        # Get particle type
        if particle_type_idx >= len(self.particle_list):
            return False
        particle_id = self.particle_list[particle_type_idx]
        
        # Create new interaction vertex
        new_vertex = {
            'id': len(self.vertices),
            'type': 'interaction',
            'x': self.vertices[vertex_idx]['x'] + 50,  # Offset position
            'y': self.vertices[vertex_idx]['y'],
            'connected_edges': []
        }
        self.vertices.append(new_vertex)
        
        # Create outgoing edge (e.g., photon)
        new_edge = {
            'id': len(self.edges),
            'source': new_vertex['id'],
            'target': None,
            'particle_id': particle_id,
            'is_anti': False,
            'color': None,
            'is_external': False,
            'state': 'open'
        }
        self.edges.append(new_edge)
        new_vertex['connected_edges'].append(new_edge['id'])
        
        # Connect the open half-line to the new vertex
        edge_to_connect = self.edges[open_lines[0]]
        if edge_to_connect['source'] is None:
            edge_to_connect['source'] = new_vertex['id']
        else:
            edge_to_connect['target'] = new_vertex['id']
        edge_to_connect['state'] = 'connected'
        new_vertex['connected_edges'].append(edge_to_connect['id'])
        
        return True
    
    def _execute_set_type(self, edge_idx: int, particle_type_idx: int) -> bool:
        """Set particle type for an edge"""
        if edge_idx >= len(self.edges):
            return False
        
        if particle_type_idx >= len(self.particle_list):
            return False
        
        particle_id = self.particle_list[particle_type_idx]
        self.edges[edge_idx]['particle_id'] = particle_id
        
        return True
    
    def _get_open_halflines(self, vertex_idx: int) -> List[int]:
        """Get indices of open half-lines connected to a vertex"""
        open_lines = []
        for edge_id in self.vertices[vertex_idx]['connected_edges']:
            edge = self.edges[edge_id]
            if edge['state'] == 'open':
                open_lines.append(edge_id)
        return open_lines
    
    def _compute_step_reward(self, vertex_idx: int) -> float:
        """
        Compute immediate reward based on local conservation laws at a vertex
        This implements the physics-informed step reward
        """
        vertex = self.vertices[vertex_idx]
        connected_edges = [self.edges[eid] for eid in vertex['connected_edges']]
        
        # Separate incoming and outgoing edges
        incoming = []
        outgoing = []
        
        for edge in connected_edges:
            if edge['target'] == vertex_idx:
                incoming.append(edge)
            elif edge['source'] == vertex_idx:
                outgoing.append(edge)
        
        reward = 0.0
        
        # Extract quantum numbers
        q_in = [self._get_charge(e) for e in incoming]
        q_out = [self._get_charge(e) for e in outgoing]
        
        l_in = [self._get_lepton(e) for e in incoming]
        l_out = [self._get_lepton(e) for e in outgoing]
        
        b_in = [self._get_baryon(e) for e in incoming]
        b_out = [self._get_baryon(e) for e in outgoing]
        
        colors_in = [e['color'] for e in incoming]
        colors_out = [e['color'] for e in outgoing]
        
        # Check conservation laws
        charge_ok, charge_mismatch = ConservationLaws.check_charge_conservation(q_in, q_out)
        lepton_ok, lepton_mismatch = ConservationLaws.check_lepton_conservation(l_in, l_out)
        baryon_ok, baryon_mismatch = ConservationLaws.check_baryon_conservation(b_in, b_out)
        color_ok, color_mismatch = ConservationLaws.check_color_conservation(colors_in, colors_out)
        
        if not charge_ok:
            reward += self.reward_weights['charge_violation'] * charge_mismatch
        if not lepton_ok:
            reward += self.reward_weights['lepton_violation'] * lepton_mismatch
        if not baryon_ok:
            reward += self.reward_weights['baryon_violation'] * baryon_mismatch
        if not color_ok:
            reward += self.reward_weights['color_violation'] * color_mismatch
        
        # Check interaction rules
        particle_ids = [e['particle_id'] for e in connected_edges]
        rules_ok, violations = ConservationLaws.check_interaction_rules(particle_ids)
        if not rules_ok:
            reward += self.reward_weights['interaction_violation'] * len(violations)
        
        return reward
    
    def _compute_terminal_reward(self) -> float:
        """
        Compute final reward based on:
        1. Target match (external lines match I → F)
        2. Topological validity (connected, no dangling lines)
        3. Complexity penalty (fewer vertices is better)
        """
        reward = 0.0
        
        # 1. Check target match
        initial_match = self._check_external_match(self.initial_particles, 'initial')
        final_match = self._check_external_match(self.final_particles, 'final')
        
        if initial_match and final_match:
            reward += self.reward_weights['target_match']
        else:
            reward -= self.reward_weights['target_match']
        
        # 2. Check topology
        is_connected = self._is_graph_connected()
        no_dangling = self._no_dangling_internal_lines()
        
        if is_connected and no_dangling:
            reward += self.reward_weights['topology_valid']
        else:
            reward -= self.reward_weights['topology_valid']
        
        # 3. Complexity penalty (encourage minimal diagrams)
        num_interaction_vertices = sum(1 for v in self.vertices if v['type'] == 'interaction')
        reward += self.reward_weights['complexity_penalty'] * num_interaction_vertices
        
        return reward
    
    def _check_external_match(self, target_particles: List[str], vertex_type: str) -> bool:
        """Check if external lines match the target state"""
        external_vertices = [v for v in self.vertices if v['type'] == vertex_type]
        
        if len(external_vertices) != len(target_particles):
            return False
        
        # Get particle IDs from connected edges
        found_particles = []
        for v in external_vertices:
            for edge_id in v['connected_edges']:
                edge = self.edges[edge_id]
                if edge['is_external']:
                    found_particles.append(edge['particle_id'])
        
        # Check if they match (order doesn't matter)
        return sorted(found_particles) == sorted(target_particles)
    
    def _is_graph_connected(self) -> bool:
        """Check if the graph is fully connected using BFS"""
        if len(self.vertices) == 0:
            return True
        
        visited = set()
        queue = [0]
        visited.add(0)
        
        while queue:
            v_id = queue.pop(0)
            vertex = self.vertices[v_id]
            
            for edge_id in vertex['connected_edges']:
                edge = self.edges[edge_id]
                neighbors = [edge['source'], edge['target']]
                
                for neighbor_id in neighbors:
                    if neighbor_id is not None and neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)
        
        return len(visited) == len(self.vertices)
    
    def _no_dangling_internal_lines(self) -> bool:
        """Check that all internal lines are fully connected"""
        for edge in self.edges:
            if not edge['is_external']:
                if edge['source'] is None or edge['target'] is None:
                    return False
        return True
    
    def _get_observation(self) -> Data:
        """
        Convert current graph state to PyTorch Geometric Data object
        
        Node features: [type (3), x_pos, y_pos, num_connections]
        Edge features: [particle_encoding (21 features from ParticleEncoder)]
        """
        # Node features
        node_features = []
        for v in self.vertices:
            # One-hot encoding for vertex type
            type_vec = [0, 0, 0]
            if v['type'] == 'initial':
                type_vec[0] = 1
            elif v['type'] == 'final':
                type_vec[1] = 1
            elif v['type'] == 'interaction':
                type_vec[2] = 1
            
            # Normalized spatial coordinates
            x_norm = v['x'] / self.canvas_width
            y_norm = v['y'] / self.canvas_height
            
            # Number of connections
            num_conn = len(v['connected_edges'])
            
            # Calculate net quantum numbers for this vertex
            q_net = 0.0
            l_net = 0.0
            b_net = 0.0
            
            for edge_id in v['connected_edges']:
                edge = self.edges[edge_id]
                # Determine if edge is incoming or outgoing relative to this vertex
                is_incoming = (edge['target'] == v['id'])
                sign = 1.0 if is_incoming else -1.0
                
                # Get properties
                q_net += sign * self._get_charge(edge)
                l_net += sign * self._get_lepton(edge)
                b_net += sign * self._get_baryon(edge)
            
            node_feat = type_vec + [x_norm, y_norm, num_conn, q_net, l_net, b_net]
            node_features.append(node_feat)
        
        # Edge index and edge features
        edge_index = []
        edge_features = []
        
        for edge in self.edges:
            if edge['source'] is not None and edge['target'] is not None:
                edge_index.append([edge['source'], edge['target']])
                
                # Encode particle using ParticleEncoder
                particle_feat = ParticleEncoder.encode_particle(
                    edge['particle_id'],
                    is_anti=edge['is_anti'],
                    color=edge['color']
                )
                edge_features.append(particle_feat)
        
        # Convert to tensors (optimize: convert to numpy first to avoid warning)
        x = torch.tensor(np.array(node_features, dtype=np.float32), dtype=torch.float32)
        
        if len(edge_index) > 0:
            edge_index_tensor = torch.tensor(np.array(edge_index, dtype=np.int64), dtype=torch.long).t().contiguous()
            # edge_features is already a list of numpy arrays, stack them first
            edge_attr = torch.from_numpy(np.stack(edge_features)).float()
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 21), dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr)
        
        return data
    
    def _get_info(self) -> Dict:
        """Return additional information about the current state"""
        return {
            'num_vertices': len(self.vertices),
            'num_edges': len(self.edges),
            'step_count': self.step_count,
            'is_terminated': self.terminated
        }
    
    def _get_charge(self, edge: Dict) -> float:
        """Get charge of particle on edge"""
        p = PhysicsConstants.get_particle_by_id(edge['particle_id'])
        b = PhysicsConstants.get_boson_by_id(edge['particle_id'])
        
        charge = 0.0
        if p:
            charge = p.charge
        elif b:
            charge = b.charge
        
        if edge['is_anti']:
            charge = -charge
        
        return charge
    
    def _get_lepton(self, edge: Dict) -> float:
        """Get lepton number of particle on edge"""
        p = PhysicsConstants.get_particle_by_id(edge['particle_id'])
        
        if p:
            lepton = p.lepton
            if edge['is_anti']:
                lepton = -lepton
            return lepton
        
        return 0.0
    
    def _get_baryon(self, edge: Dict) -> float:
        """Get baryon number of particle on edge"""
        p = PhysicsConstants.get_particle_by_id(edge['particle_id'])
        
        if p:
            baryon = p.baryon
            if edge['is_anti']:
                baryon = -baryon
            return baryon
        
        return 0.0
    
    def render(self, mode='human'):
        """Render the current state (for debugging)"""
        if mode == 'human':
            print(f"\n=== Step {self.step_count} ===")
            print(f"Vertices: {len(self.vertices)}")
            print(f"Edges: {len(self.edges)}")
            for i, v in enumerate(self.vertices):
                print(f"  V{i}: {v['type']} at ({v['x']:.0f}, {v['y']:.0f})")
            for i, e in enumerate(self.edges):
                print(f"  E{i}: {e['particle_id']} from V{e['source']} to V{e['target']} ({e['state']})")
    
    def get_diagram_json(self) -> List[Dict]:
        """
        Export current diagram in the format expected by feynman-logic.js
        Returns a list of shapes compatible with canvas-manager.js
        """
        shapes = []
        
        for edge in self.edges:
            if edge['source'] is not None and edge['target'] is not None:
                source_v = self.vertices[edge['source']]
                target_v = self.vertices[edge['target']]
                
                # Determine shape type
                p = PhysicsConstants.get_particle_by_id(edge['particle_id'])
                b = PhysicsConstants.get_boson_by_id(edge['particle_id'])
                
                if p:
                    shape_type = 'fermion'
                elif edge['particle_id'] == 'photon':
                    shape_type = 'photon'
                elif edge['particle_id'] in ['w_plus', 'w_minus']:
                    shape_type = 'boson_w'
                elif edge['particle_id'] == 'z':
                    shape_type = 'boson_z'
                elif edge['particle_id'] == 'gluon':
                    shape_type = 'gluon'
                elif edge['particle_id'] == 'higgs':
                    shape_type = 'higgs'
                else:
                    shape_type = 'fermion'
                
                shape = {
                    'id': edge['id'],
                    'type': shape_type,
                    'p1': {'x': source_v['x'], 'y': source_v['y']},
                    'p2': {'x': target_v['x'], 'y': target_v['y']},
                    'props': {
                        'particleId': edge['particle_id'],
                        'isAnti': edge['is_anti'],
                        'color': edge['color'],
                        'category': 'fermion' if p else 'boson',
                        'group': self._get_particle_group(edge['particle_id'])
                    }
                }
                
                shapes.append(shape)
        
        return shapes
    
    def _get_particle_group(self, particle_id: str) -> str:
        """Get particle group (lepton, quark_u, quark_d, or boson type)"""
        if PhysicsConstants.is_lepton(particle_id):
            return 'lepton'
        elif particle_id in ['u', 'c', 't']:
            return 'quark_u'
        elif particle_id in ['d', 's', 'b']:
            return 'quark_d'
        else:
            return particle_id
