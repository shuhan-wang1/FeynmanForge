"""
Feynman-GCPN Neural Network Architecture
Implements MPNN encoder with Physics-Gated Policy Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
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
        
        # Ensure dimensions are explicit and compatible
        self._node_dim = node_dim  # Renamed to avoid conflict with MessagePassing.node_dim
        self._edge_dim = edge_dim
        self._hidden_dim = hidden_dim
        
        # Message MLP: processes (neighbor_node, edge) -> message
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Update GRU: combines old node state with aggregated message
        # CRITICAL: Both input and hidden must be same dimension
        assert node_dim == hidden_dim, f"GRU requires node_dim={node_dim} == hidden_dim={hidden_dim}"
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
    
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
        node_input_dim: int = 7,      # [type(3), num_conn, q_net, l_net, b_net] - removed x, y
        edge_input_dim: int = 22,     # Particle encoding from ParticleEncoder (now includes is_reverse)
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # FIXED: Use deeper encoders for heterogeneous features
        # Node features: [type(3), x, y, num_conn, q_net, l_net, b_net] - mixed one-hot and continuous
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Edge features: 21 dimensions with particle properties
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
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
        
        # Global graph embedding
        # CRITICAL FIX: Use SUM pooling instead of MEAN pooling!
        # Mean pooling destroys information about graph size:
        #   - Before: mean([v₀, v₁, v₂, v₃]) 
        #   - After:  mean([v₀, v₁, v₂, v₃, v₄])
        #   → Only 20% change even though we added a vertex!
        # 
        # Sum pooling preserves size information:
        #   - Before: sum([v₀, v₁, v₂, v₃]) = 4 * avg
        #   - After:  sum([v₀, v₁, v₂, v₃, v₄]) = 5 * avg
        #   → Clear signal that graph grew!
        #
        # This is why your value loss was 519.5 (exploded) - the value function
        # couldn't distinguish between different graph sizes!
        if hasattr(data, 'batch'):
            graph_emb = global_add_pool(x, data.batch)
        else:
            graph_emb = x.sum(dim=0, keepdim=True)
        
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
        
        # Create tensor on the same device as conservation_weights
        return torch.tensor(
            [delta_q, delta_l, delta_b, delta_color], 
            dtype=torch.float32,
            device=self.conservation_weights.device
        )
    
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
        num_action_types: int = 5,  # Updated from 4 to 5 for ACTION_MERGE
        num_particle_types: int = 20,
        max_vertices: int = 10,
        lambda_penalty: float = 5.0
    ):
        super().__init__()
        
        self.num_action_types = num_action_types
        self.num_particle_types = num_particle_types
        self.max_vertices = max_vertices
        
        # FIXED: Add action type embeddings for hierarchical/conditional policy
        # This allows vertex selection to be conditioned on action type
        self.action_type_embed = nn.Embedding(num_action_types, embedding_dim)
        
        # Neural network policy (before physics gating)
        self.action_type_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_action_types)
        )
        
        # CRITICAL FIX: Node-wise scoring (Pointer Network)
        # Source vertex: "which vertex to modify" (sees each vertex's features!)
        # Input: [node_emb(H) + graph_emb(H) + action_emb(H)] = 3H
        # Output: 1 score per node
        self.source_vertex_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),  # node + graph + action
            nn.ReLU(),
            nn.Linear(128, 1)  # Output 1 score per node
        )
        
        # Target vertex: "which vertex to connect to" (sees each vertex's features!)
        self.target_vertex_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),  # node + graph + action
            nn.ReLU(),
            nn.Linear(128, 1)  # Output 1 score per node
        )
        
        # Particle head: conditioned on action type AND selected vertex
        self.particle_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),  # graph_emb + action_emb
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
        node_embeddings: torch.Tensor,  # CRITICAL NEW INPUT: [num_nodes, hidden_dim]
        vertex_states: Optional[List[Dict]] = None,
        mask_invalid: bool = False,
        num_vertices: Optional[int] = None,
        action_masks: Optional[Dict[str, torch.Tensor]] = None  # NEW: Validity masks
    ) -> Dict[str, torch.Tensor]:
        """
        CRITICAL FIX: Pointer Network Architecture + Action Masking
        
        The model can now "see" individual vertices and their features!
        Before: Blind - could only guess based on global graph summary
        After: Has eyes - looks at each vertex's specific features (open edges, type, etc.)
        
        Action Masking forces invalid actions to have 0% probability.

        Args:
            graph_embedding: [embedding_dim] - Global graph context
            node_embeddings: [num_nodes, embedding_dim] - PER-VERTEX features!
            vertex_states: List of vertex quantum number states
            mask_invalid: Whether to apply physics gate
            num_vertices: Actual number of vertices in the graph
            action_masks: Dict with boolean masks for valid actions (keys: action_type, source_vertex, target_vertex, particle_type)

        Returns:
            Dictionary with action logits and probabilities
        """
        # Step 1: Predict action type (global decision is fine)
        action_type_logits = self.action_type_head(graph_embedding)
        
        # EXPLORATION FIX: Apply bias against TERMINATE
        TERMINATE_ACTION_IDX = 3
        TERMINATE_BIAS = -5.0
        action_type_logits[..., TERMINATE_ACTION_IDX] = action_type_logits[..., TERMINATE_ACTION_IDX] + TERMINATE_BIAS
        
        # ACTION MASKING: Set invalid action types to -inf
        if action_masks is not None and 'action_type' in action_masks:
            mask = action_masks['action_type']
            action_type_logits = torch.where(
                mask,
                action_type_logits,
                torch.tensor(float('-inf'), device=action_type_logits.device)
            )
        
        # Step 2: Compute action embedding
        action_probs_for_emb = F.softmax(action_type_logits, dim=-1)
        all_action_embs = self.action_type_embed.weight
        action_emb = (action_probs_for_emb.unsqueeze(-1) * all_action_embs).sum(dim=0)
        
        # Step 3: POINTER NETWORK - Score each node individually
        # Instead of: score = MLP(global)
        # We do: score_i = MLP(node_i || global || action)
        
        num_nodes = node_embeddings.shape[0]
        
        # Broadcast global context to all nodes: [num_nodes, embedding_dim]
        graph_emb_expanded = graph_embedding.unsqueeze(0).expand(num_nodes, -1)
        action_emb_expanded = action_emb.unsqueeze(0).expand(num_nodes, -1)
        
        # Concatenate: [Node_Features || Global_Context || Action_Context]
        # Shape: [num_nodes, embedding_dim * 3]
        vertex_input = torch.cat([node_embeddings, graph_emb_expanded, action_emb_expanded], dim=-1)
        
        # Score EACH node individually - model can now "see" which vertex has open edges!
        # Output: [num_nodes, 1] -> squeeze to [num_nodes]
        source_vertex_logits = self.source_vertex_head(vertex_input).squeeze(-1)
        target_vertex_logits = self.target_vertex_head(vertex_input).squeeze(-1)
        
        # Pad to max_vertices for consistency (if graph has fewer vertices)
        if num_nodes < self.max_vertices:
            pad_size = self.max_vertices - num_nodes
            pad = torch.full((pad_size,), float('-inf'), device=source_vertex_logits.device)
            source_vertex_logits = torch.cat([source_vertex_logits, pad], dim=0)
            target_vertex_logits = torch.cat([target_vertex_logits, pad], dim=0)
        elif num_nodes > self.max_vertices:
            # Truncate if somehow exceeds max
            source_vertex_logits = source_vertex_logits[:self.max_vertices]
            target_vertex_logits = target_vertex_logits[:self.max_vertices]
        
        # ACTION MASKING: Apply vertex masks
        if action_masks is not None:
            if 'source_vertex' in action_masks:
                mask = action_masks['source_vertex']
                source_vertex_logits = torch.where(
                    mask,
                    source_vertex_logits,
                    torch.tensor(float('-inf'), device=source_vertex_logits.device)
                )
            if 'target_vertex' in action_masks:
                mask = action_masks['target_vertex']
                target_vertex_logits = torch.where(
                    mask,
                    target_vertex_logits,
                    torch.tensor(float('-inf'), device=target_vertex_logits.device)
                )
        
        # Step 4: Particle selection (global context is fine for this)
        conditioned_input = torch.cat([graph_embedding, action_emb], dim=-1)
        particle_logits = self.particle_head(conditioned_input)
        
        # ACTION MASKING: Apply particle mask
        if action_masks is not None and 'particle_type' in action_masks:
            mask = action_masks['particle_type']
            particle_logits = torch.where(
                mask,
                particle_logits,
                torch.tensor(float('-inf'), device=particle_logits.device)
            )

        # Physics gate currently disabled - would need proper target vertex indexing
        # Future improvement: Pass target_vertex_idx to apply_physics_mask
        # if mask_invalid and vertex_states is not None and target_vertex_idx is not None:
        #     particle_logits = self.apply_physics_mask(
        #         particle_logits,
        #         vertex_states[target_vertex_idx],
        #         self.particle_list
        #     )

        # Softmax to get probabilities
        # SAFETY: Handle edge case where all logits are -inf (all actions masked)
        # This can happen if all vertices have no open edges
        def safe_softmax(logits, dim=-1):
            """Softmax that handles all-masked case by returning uniform distribution"""
            # Check if all values are -inf
            if torch.all(logits == float('-inf')):
                # Return uniform distribution over all positions
                return torch.ones_like(logits) / logits.shape[dim]
            # Check for any inf/nan
            if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits) & (logits > 0)):
                # Replace nan/+inf with -inf, then handle
                logits = torch.where(torch.isnan(logits) | (logits == float('inf')), 
                                    torch.tensor(float('-inf'), device=logits.device), logits)
                if torch.all(logits == float('-inf')):
                    return torch.ones_like(logits) / logits.shape[dim]
            probs = F.softmax(logits, dim=dim)
            # Final safety check
            if torch.any(torch.isnan(probs)):
                return torch.ones_like(probs) / probs.shape[dim]
            return probs
        
        action_type_probs = safe_softmax(action_type_logits, dim=-1)
        source_vertex_probs = safe_softmax(source_vertex_logits, dim=-1)
        target_vertex_probs = safe_softmax(target_vertex_logits, dim=-1)
        particle_probs = safe_softmax(particle_logits, dim=-1)

        return {
            'action_type_logits': action_type_logits,
            'action_type_probs': action_type_probs,
            'source_vertex_logits': source_vertex_logits,
            'source_vertex_probs': source_vertex_probs,
            'target_vertex_logits': target_vertex_logits,
            'target_vertex_probs': target_vertex_probs,
            'particle_logits': particle_logits,
            'particle_probs': particle_probs,
            # Keep old keys for backward compatibility during transition
            'vertex_logits': source_vertex_logits,
            'vertex_probs': source_vertex_probs
        }
    
    def apply_physics_mask(
        self,
        particle_logits: torch.Tensor,
        vertex_states: List[Dict],
        particle_list: List[str]
    ) -> torch.Tensor:
        """
        DISABLED: Was suppressing valid actions due to hardcoded vertex[0] check.
        Returns raw logits without physics masking.
        
        The original implementation incorrectly assumed vertex_states[0] was the
        target vertex, but vertex[0] is typically the initial state particle which
        has different conservation requirements (source node with no incoming edges).
        This caused the physics gate to incorrectly suppress valid particle selections.
        """
        return particle_logits


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
        node_input_dim: int = 7,  # Updated: removed x, y (canvas bias)
        edge_input_dim: int = 22,  # Updated from 21 to 22 (added is_reverse flag)
        hidden_dim: int = 128,
        num_mp_layers: int = 3,
        num_action_types: int = 5,  # Updated from 4 to 5 for ACTION_MERGE
        num_particle_types: int = 18,  # 12 fermions + 6 bosons
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
        return_value: bool = True,
        action_masks: Optional[Dict[str, torch.Tensor]] = None  # NEW: Pass action masks
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass

        Args:
            data: PyG Data object
            vertex_states: Quantum number states for physics gating
            return_value: Whether to compute value estimate
            action_masks: Optional validity masks for actions

        Returns:
            Dictionary with policy outputs and optionally value
        """
        # Encode graph
        node_embeddings, graph_embedding = self.encoder(data)

        # CRITICAL FIX: Extract actual number of vertices from graph
        # For individual graphs: data.x.shape[0] is the number of nodes
        # For batched graphs: This gets called on individual graphs after to_data_list()
        # so data.x.shape[0] is still correct
        num_vertices = data.x.shape[0]

        # Policy (with vertex masking)
        # POINTER NETWORK FIX: Pass node embeddings so policy can see individual vertex features
        # ACTION MASKING FIX: Pass masks to force invalid actions to 0% probability
        policy_output = self.policy_head(
            graph_embedding,
            node_embeddings,  # Now policy can distinguish vertices by their features
            vertex_states,
            num_vertices=num_vertices,
            action_masks=action_masks  # NEW: Apply action masking
        )

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
        action_masks: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Dict[str, int]:
        """
        Sample an action from the policy
        
        Args:
            data: Current state as PyG Data
            vertex_states: For physics gating
            action_masks: Optional action masks to prevent invalid actions
            deterministic: If True, take argmax; else sample
            
        Returns:
            action: Dictionary with action_type, vertex_idx, particle_type, target_vertex
        """
        with torch.no_grad():
            output = self.forward(data, vertex_states, return_value=False, action_masks=action_masks)
            
            if deterministic:
                action_type = output['action_type_probs'].argmax().item()
                vertex_idx = output['source_vertex_probs'].argmax().item()
                particle_type = output['particle_probs'].argmax().item()

                # Use dedicated target vertex distribution
                # Mask out the selected source vertex to prevent MERGE(v, v)
                target_probs = output['target_vertex_probs'].clone()
                target_probs[vertex_idx] = 0
                if target_probs.sum() > 0:
                    target_probs = target_probs / target_probs.sum()
                    target_vertex = target_probs.argmax().item()
                else:
                    target_vertex = 0
            else:
                action_type = torch.multinomial(output['action_type_probs'], 1).item()
                vertex_idx = torch.multinomial(output['source_vertex_probs'], 1).item()
                particle_type = torch.multinomial(output['particle_probs'], 1).item()

                # Use dedicated target vertex distribution
                # Mask out the selected source vertex to prevent MERGE(v, v)
                target_probs = output['target_vertex_probs'].clone()
                target_probs[vertex_idx] = 0
                if target_probs.sum() > 0:
                    target_probs = target_probs / target_probs.sum()
                    target_vertex = torch.multinomial(target_probs, 1).item()
                else:
                    target_vertex = (vertex_idx + 1) % len(target_probs)
        
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

        # Use target_vertex_probs from the dedicated target head
        # Mask out vertex_idx to match the sampling procedure
        target_probs = output['target_vertex_probs'].clone()
        target_probs[actions['vertex_idx']] = 0
        if target_probs.sum() > 0:
            target_probs = target_probs / target_probs.sum()
        target_vertex_log_prob = torch.log(target_probs[actions['target_vertex']] + 1e-8)

        total_log_prob = action_type_log_prob + vertex_log_prob + particle_log_prob + target_vertex_log_prob

        # Compute entropy (including target_vertex distribution)
        action_type_entropy = -(output['action_type_probs'] * torch.log(output['action_type_probs'] + 1e-8)).sum()
        vertex_entropy = -(output['vertex_probs'] * torch.log(output['vertex_probs'] + 1e-8)).sum()
        particle_entropy = -(output['particle_probs'] * torch.log(output['particle_probs'] + 1e-8)).sum()
        target_vertex_entropy = -(target_probs * torch.log(target_probs + 1e-8)).sum()

        total_entropy = action_type_entropy + vertex_entropy + particle_entropy + target_vertex_entropy

        return total_log_prob, output['value'], total_entropy
