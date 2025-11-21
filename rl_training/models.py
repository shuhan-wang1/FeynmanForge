"""
Feynman-GCPN Neural Network Architecture
Implements MPNN encoder with Physics-Gated Policy Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import numpy as np

from physics_engine import PhysicsConstants, ConservationLaws


class MPNNLayer(MessagePassing):
    """
    Message Passing Neural Network Layer
    Aggregates information from neighboring particles into vertex representations
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr='add')  # Use 'add' aggregation
        
        # Message MLP: processes (neighbor_node, edge) -> message
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Update GRU: combines old node state with aggregated message
        self.gru = nn.GRUCell(hidden_dim, node_dim)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        """
        Construct messages from neighbors
        x_j: neighbor node features [num_edges, node_dim]
        edge_attr: edge features [num_edges, edge_dim]
        """
        # Concatenate neighbor features with edge features
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)
    
    def update(self, aggr_out, x):
        """
        Update node representations using GRU
        aggr_out: aggregated messages [num_nodes, hidden_dim]
        x: current node features [num_nodes, node_dim]
        """
        return self.gru(aggr_out, x)


class FeynmanMPNN(nn.Module):
    """
    Multi-layer MPNN encoder for Feynman diagram states
    """
    
    def __init__(
        self, 
        node_input_dim: int = 6,      # [type(3), x, y, num_conn]
        edge_input_dim: int = 21,     # Particle encoding from ParticleEncoder
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_input_dim, hidden_dim)
        
        # MPNN layers
        self.mp_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MPNN
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr
            
        Returns:
            node_embeddings: [num_nodes, hidden_dim]
            graph_embedding: [hidden_dim] (global pooling)
        """
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        
        # Message passing
        for mp_layer, norm in zip(self.mp_layers, self.layer_norms):
            x_new = mp_layer(x, data.edge_index, edge_attr)
            x = norm(x_new)
        
        # Global graph embedding (mean pooling)
        if hasattr(data, 'batch'):
            graph_emb = global_mean_pool(x, data.batch)
        else:
            graph_emb = x.mean(dim=0, keepdim=True)
        
        return x, graph_emb.squeeze(0)


class PhysicsGate(nn.Module):
    """
    Differentiable Physics Gate: Γ(a)
    Computes conservation mismatch and suppresses invalid actions
    
    This is the CRITICAL component that makes Feynman-GCPN physics-aware
    """
    
    def __init__(self, lambda_penalty: float = 5.0, temperature: float = 1.0):
        """
        Args:
            lambda_penalty: Scaling factor for conservation violations
            temperature: Temperature for soft gating (lower = harder constraints)
        """
        super().__init__()
        
        self.lambda_penalty = lambda_penalty
        self.temperature = temperature
        
        # Learnable weights for different conservation laws
        self.register_parameter(
            'conservation_weights',
            nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 2.0]))  # Q, L, B, Color
        )
    
    def compute_mismatch(
        self,
        action_type: int,
        vertex_state: Dict,
        candidate_particle: str,
        candidate_color: Optional[str] = None,
        is_anti: bool = False
    ) -> torch.Tensor:
        """
        Compute ΔQ, ΔL, ΔB, ΔColor for a candidate action
        
        Returns:
            mismatch_vector: [4] tensor with [ΔQ, ΔL, ΔB, ΔColor]
        """
        # Extract current quantum numbers at the vertex
        q_in = vertex_state.get('charge_in', 0.0)
        q_out = vertex_state.get('charge_out', 0.0)
        l_in = vertex_state.get('lepton_in', 0.0)
        l_out = vertex_state.get('lepton_out', 0.0)
        b_in = vertex_state.get('baryon_in', 0.0)
        b_out = vertex_state.get('baryon_out', 0.0)
        colors_in = vertex_state.get('colors_in', [])
        colors_out = vertex_state.get('colors_out', [])
        
        # Get candidate particle properties
        p = PhysicsConstants.get_particle_by_id(candidate_particle)
        boson = PhysicsConstants.get_boson_by_id(candidate_particle)
        
        if p:
            cand_q = -p.charge if is_anti else p.charge
            cand_l = -p.lepton if is_anti else p.lepton
            cand_b = -p.baryon if is_anti else p.baryon
        elif boson:
            cand_q = boson.charge
            cand_l = boson.lepton
            cand_b = boson.baryon
        else:
            cand_q = cand_l = cand_b = 0.0
        
        # Compute mismatches (adding to outgoing)
        delta_q = abs((q_in) - (q_out + cand_q))
        delta_l = abs((l_in) - (l_out + cand_l))
        delta_b = abs((b_in) - (b_out + cand_b))
        
        # Color mismatch (simplified: count net color charge)
        new_colors_out = colors_out + [candidate_color] if candidate_color else colors_out
        _, color_mismatch = ConservationLaws.check_color_conservation(colors_in, new_colors_out)
        delta_color = color_mismatch
        
        return torch.tensor([delta_q, delta_l, delta_b, delta_color], dtype=torch.float32)
    
    def forward(self, mismatch_vector: torch.Tensor) -> torch.Tensor:
        """
        Compute gate value: Γ(a) = exp(-λ Σ w_k (Δ_k)^2)
        
        Args:
            mismatch_vector: [batch_size, 4] or [4]
            
        Returns:
            gate_value: [batch_size] or scalar in (0, 1]
        """
        # Weighted squared mismatch
        weighted_mismatch = (mismatch_vector ** 2) * self.conservation_weights
        total_penalty = weighted_mismatch.sum(dim=-1)
        
        # Exponential gate with temperature
        gate_value = torch.exp(-self.lambda_penalty * total_penalty / self.temperature)
        
        return gate_value


class PhysicsGatedPolicyHead(nn.Module):
    """
    Policy head with integrated physics gate
    Outputs π(a|G) = π_θ(a|G) · Γ(a) (normalized)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_action_types: int = 4,
        num_particle_types: int = 20,
        max_vertices: int = 10,
        lambda_penalty: float = 5.0
    ):
        super().__init__()
        
        self.num_action_types = num_action_types
        self.num_particle_types = num_particle_types
        self.max_vertices = max_vertices
        
        # Neural network policy (before physics gating)
        self.action_type_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_action_types)
        )
        
        self.vertex_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_vertices)
        )
        
        self.particle_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_particle_types)
        )
        
        # Cache particle list for the gate
        self.particle_list = [p.id for p in PhysicsConstants.get_all_particles()] + \
                             list(PhysicsConstants.BOSONS.keys())
        
        # Physics gate
        self.physics_gate = PhysicsGate(lambda_penalty=lambda_penalty)
    
    def forward(
        self,
        graph_embedding: torch.Tensor,
        vertex_states: Optional[List[Dict]] = None,
        mask_invalid: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with physics gating
        
        Args:
            graph_embedding: [batch_size, embedding_dim] or [embedding_dim]
            vertex_states: List of vertex quantum number states (for gate computation)
            mask_invalid: Whether to apply physics gate
            
        Returns:
            Dictionary with action logits and probabilities
        """
        # Raw neural network outputs
        action_type_logits = self.action_type_head(graph_embedding)
        vertex_logits = self.vertex_head(graph_embedding)
        particle_logits = self.particle_head(graph_embedding)
        
        if mask_invalid and vertex_states is not None:
            # Apply physics gate to particle selection
            particle_logits = self.apply_physics_mask(
                particle_logits, 
                vertex_states, 
                self.particle_list
            )
        
        # Softmax to get probabilities
        action_type_probs = F.softmax(action_type_logits, dim=-1)
        vertex_probs = F.softmax(vertex_logits, dim=-1)
        particle_probs = F.softmax(particle_logits, dim=-1)
        
        return {
            'action_type_logits': action_type_logits,
            'action_type_probs': action_type_probs,
            'vertex_logits': vertex_logits,
            'vertex_probs': vertex_probs,
            'particle_logits': particle_logits,
            'particle_probs': particle_probs
        }
    
    def apply_physics_mask(
        self,
        particle_logits: torch.Tensor,
        vertex_states: List[Dict],
        particle_list: List[str]
    ) -> torch.Tensor:
        """
        Apply physics gate to mask out invalid particle choices
        
        Args:
            particle_logits: [num_particle_types] raw logits
            vertex_states: Current quantum numbers at each vertex
            particle_list: List of particle IDs
            
        Returns:
            masked_logits: [num_particle_types] with physics penalties applied
        """
        gate_values = []
        
        for i, particle_id in enumerate(particle_list):
            # Compute mismatch for adding this particle
            # (Assuming we're adding to vertex 0 for simplicity)
            if len(vertex_states) > 0:
                mismatch = self.physics_gate.compute_mismatch(
                    action_type=2,  # SET_TYPE
                    vertex_state=vertex_states[0],
                    candidate_particle=particle_id
                )
                gate_value = self.physics_gate(mismatch)
            else:
                gate_value = torch.tensor(1.0)
            
            gate_values.append(gate_value)
        
        gate_tensor = torch.stack(gate_values)
        
        # Apply gate as multiplicative mask (in log space: add log(gate))
        masked_logits = particle_logits + torch.log(gate_tensor + 1e-8)
        
        return masked_logits


class ValueHead(nn.Module):
    """
    Value function V(G) for critic in PPO
    """
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        
        self.value_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """
        Estimate value of the graph state
        
        Args:
            graph_embedding: [batch_size, embedding_dim] or [embedding_dim]
            
        Returns:
            value: [batch_size, 1] or [1]
        """
        return self.value_mlp(graph_embedding)


class FeynmanGCPN(nn.Module):
    """
    Complete Feynman-GCPN model combining:
    1. MPNN Encoder
    2. Physics-Gated Policy Head
    3. Value Head
    """
    
    def __init__(
        self,
        node_input_dim: int = 6,
        edge_input_dim: int = 21,
        hidden_dim: int = 128,
        num_mp_layers: int = 3,
        num_action_types: int = 4,
        num_particle_types: int = 20,
        max_vertices: int = 10,
        lambda_penalty: float = 5.0
    ):
        super().__init__()
        
        self.encoder = FeynmanMPNN(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mp_layers
        )
        
        self.policy_head = PhysicsGatedPolicyHead(
            embedding_dim=hidden_dim,
            num_action_types=num_action_types,
            num_particle_types=num_particle_types,
            max_vertices=max_vertices,
            lambda_penalty=lambda_penalty
        )
        
        self.value_head = ValueHead(embedding_dim=hidden_dim)
    
    def forward(
        self,
        data: Data,
        vertex_states: Optional[List[Dict]] = None,
        return_value: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass
        
        Args:
            data: PyG Data object
            vertex_states: Quantum number states for physics gating
            return_value: Whether to compute value estimate
            
        Returns:
            Dictionary with policy outputs and optionally value
        """
        # Encode graph
        node_embeddings, graph_embedding = self.encoder(data)
        
        # Policy
        policy_output = self.policy_head(graph_embedding, vertex_states)
        
        output = {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding,
            **policy_output
        }
        
        # Value
        if return_value:
            value = self.value_head(graph_embedding)
            output['value'] = value
        
        return output
    
    def get_action(
        self,
        data: Data,
        vertex_states: Optional[List[Dict]] = None,
        deterministic: bool = False
    ) -> Dict[str, int]:
        """
        Sample an action from the policy
        
        Args:
            data: Current state as PyG Data
            vertex_states: For physics gating
            deterministic: If True, take argmax; else sample
            
        Returns:
            action: Dictionary with action_type, vertex_idx, particle_type, target_vertex
        """
        with torch.no_grad():
            output = self.forward(data, vertex_states, return_value=False)
            
            if deterministic:
                action_type = output['action_type_probs'].argmax().item()
                vertex_idx = output['vertex_probs'].argmax().item()
                particle_type = output['particle_probs'].argmax().item()
                target_vertex = output['vertex_probs'].argmax().item()  # Simplified
            else:
                action_type = torch.multinomial(output['action_type_probs'], 1).item()
                vertex_idx = torch.multinomial(output['vertex_probs'], 1).item()
                particle_type = torch.multinomial(output['particle_probs'], 1).item()
                target_vertex = torch.multinomial(output['vertex_probs'], 1).item()
        
        return {
            'action_type': action_type,
            'vertex_idx': vertex_idx,
            'particle_type': particle_type,
            'target_vertex': target_vertex
        }
    
    def evaluate_actions(
        self,
        data: Data,
        actions: Dict[str, torch.Tensor],
        vertex_states: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update
        
        Args:
            data: State as PyG Data
            actions: Dictionary of action tensors
            vertex_states: For physics gating
            
        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy
        """
        output = self.forward(data, vertex_states, return_value=True)
        
        # Compute log probabilities
        action_type_log_prob = torch.log(output['action_type_probs'][actions['action_type']] + 1e-8)
        vertex_log_prob = torch.log(output['vertex_probs'][actions['vertex_idx']] + 1e-8)
        particle_log_prob = torch.log(output['particle_probs'][actions['particle_type']] + 1e-8)
        
        total_log_prob = action_type_log_prob + vertex_log_prob + particle_log_prob
        
        # Compute entropy
        action_type_entropy = -(output['action_type_probs'] * torch.log(output['action_type_probs'] + 1e-8)).sum()
        vertex_entropy = -(output['vertex_probs'] * torch.log(output['vertex_probs'] + 1e-8)).sum()
        particle_entropy = -(output['particle_probs'] * torch.log(output['particle_probs'] + 1e-8)).sum()
        
        total_entropy = action_type_entropy + vertex_entropy + particle_entropy
        
        return total_log_prob, output['value'], total_entropy
