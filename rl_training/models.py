"""
Feynman-GCPN V8 Neural Network Architecture
Implements Split Embedding (PQNE) + Meta-Physics Gate for Conservation Law Discovery

Key innovations:
1. Split Particle Embedding: E(p) = [E_fixed(Q,L) ⊕ E_learnable(unknown)]
2. Split Conservation Mask: α = [α_fixed ≈ 1.0 ⊕ α_learnable]
3. Meta-Physics Gate: Γ(a) = exp(-λ Σ α_k (Δ_k)²)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import numpy as np

from physics_engine import PhysicsConstants, ConservationLaws


class SplitParticleEmbedding(nn.Module):
    """
    Split Particle Embedding (PQNE - Physics-Quantum Number Encoding)

    E(p) = [E_fixed(p) ⊕ E_learnable(p)]

    - E_fixed: Encodes KNOWN quantum numbers (Charge Q, Lepton L) - FROZEN
    - E_learnable: Randomly initialized, model must learn to encode UNKNOWN properties (Baryon B)

    This forces the model to "discover" conservation laws by learning meaningful
    representations in E_learnable that align with hidden conservation constraints.
    """

    def __init__(self, num_particles: int, fixed_dim: int = 2, learnable_dim: int = 6):
        """
        Args:
            num_particles: Total number of particle types
            fixed_dim: Dimensions for known laws (Q, L)
            learnable_dim: Dimensions for discovery (unknown laws like B)
        """
        super().__init__()

        self.num_particles = num_particles
        self.fixed_dim = fixed_dim
        self.learnable_dim = learnable_dim
        self.total_dim = fixed_dim + learnable_dim

        # Fixed embedding: Encode Q and L directly (FROZEN, no gradient)
        self.fixed_embedding = self._create_fixed_embedding()
        self.fixed_embedding.requires_grad = False  # CRITICAL: Freeze these

        # Learnable embedding: Random initialization for discovery
        self.learnable_embedding = nn.Parameter(
            torch.randn(num_particles, learnable_dim) * 0.01
        )

    def _create_fixed_embedding(self) -> torch.Tensor:
        """
        Create fixed embedding matrix encoding Q and L
        Shape: [num_particles, fixed_dim=2]

        Row i: [charge_i, lepton_i]
        """
        all_particles = [p.id for p in PhysicsConstants.get_all_particles()]
        all_bosons = list(PhysicsConstants.BOSONS.keys())
        particle_list = all_particles + all_bosons

        fixed_emb = torch.zeros(self.num_particles, self.fixed_dim)

        for i, pid in enumerate(particle_list):
            p = PhysicsConstants.get_particle_by_id(pid)
            b = PhysicsConstants.get_boson_by_id(pid)

            if p:
                charge = p.charge
                lepton = p.lepton
            elif b:
                charge = b.charge
                lepton = b.lepton
            else:
                charge = lepton = 0.0

            # Normalize to [-1, 1]
            fixed_emb[i, 0] = charge
            fixed_emb[i, 1] = lepton

        return fixed_emb

    def forward(self, particle_indices: torch.Tensor) -> torch.Tensor:
        """
        Get split embeddings for particle indices

        Args:
            particle_indices: [batch_size] or [N] indices

        Returns:
            embeddings: [batch_size, total_dim] = [E_fixed ⊕ E_learnable]
        """
        # Move fixed embedding to same device
        fixed_emb_device = self.fixed_embedding.to(particle_indices.device)

        # Lookup fixed and learnable parts
        fixed_part = fixed_emb_device[particle_indices]  # [N, fixed_dim]
        learnable_part = self.learnable_embedding[particle_indices]  # [N, learnable_dim]

        # Concatenate
        return torch.cat([fixed_part, learnable_part], dim=-1)

    def get_embedding_for_particle(self, particle_id: str) -> torch.Tensor:
        """Get embedding for a specific particle ID"""
        all_particles = [p.id for p in PhysicsConstants.get_all_particles()]
        all_bosons = list(PhysicsConstants.BOSONS.keys())
        particle_list = all_particles + all_bosons

        idx = particle_list.index(particle_id)
        return self.forward(torch.tensor([idx], device=self.learnable_embedding.device))[0]


class SplitConservationMask(nn.Module):
    """
    Split Conservation Mask (CLDM - Conservation Law Discovery Mechanism)

    α = [α_fixed ⊕ α_learnable]

    - α_fixed ≈ 1.0: We TELL the model "Charge and Lepton must be conserved"
    - α_learnable: The model LEARNS which dimensions should be conserved

    This allows the model to discover new conservation laws by learning
    which of its learnable embedding dimensions should be enforced.
    """

    def __init__(self, fixed_dim: int = 2, learnable_dim: int = 6):
        super().__init__()

        self.fixed_dim = fixed_dim
        self.learnable_dim = learnable_dim

        # Fixed mask: HIGH confidence for Q, L (frozen)
        self.alpha_fixed = nn.Parameter(
            torch.ones(fixed_dim) * 0.99,
            requires_grad=False  # FROZEN
        )

        # Learnable mask: Starts near zero, model learns to increase for important dims
        self.alpha_learnable_logits = nn.Parameter(
            torch.randn(learnable_dim) * 0.1 - 2.0  # Initialize low (~0.1 after sigmoid)
        )

    def forward(self) -> torch.Tensor:
        """
        Get full conservation confidence mask α ∈ [0, 1]^D

        Returns:
            alpha: [total_dim] with values in [0, 1]
        """
        # Convert learnable logits to [0, 1] via sigmoid
        alpha_learnable = torch.sigmoid(self.alpha_learnable_logits)

        # Concatenate fixed and learnable
        return torch.cat([self.alpha_fixed, alpha_learnable], dim=0)

    def get_sparsity_loss(self) -> torch.Tensor:
        """
        L1 regularization on α_learnable to encourage sparsity
        (model should only "discover" a few conservation laws, not all dimensions)
        """
        alpha_learnable = torch.sigmoid(self.alpha_learnable_logits)
        return alpha_learnable.abs().sum()


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
            graph_emb = global_mean_pool(x, data.batch)  # [num_graphs, hidden_dim]
        else:
            graph_emb = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        
        # ✅ BUG FIX 10: Always return 2D tensor [batch_size, hidden_dim]
        # Don't squeeze - let callers handle dimensions explicitly
        return x, graph_emb  # graph_emb is always [batch_size, hidden_dim]


class MetaPhysicsGate(nn.Module):
    """
    Meta-Physics Gate with Split Conservation Mask (V8)

    Γ(a) = exp(-λ Σ_k α_k (Δ_k(a))²)

    where:
    - Δ_k(a): Conservation mismatch for dimension k (computed in embedding space)
    - α_k: Confidence mask (fixed for Q, L; learned for unknown laws)
    - λ: Penalty strength

    This gate modulates the policy π'(a) = π_θ(a) · Γ(a) to enforce conservation
    """

    def __init__(
        self,
        split_mask: SplitConservationMask,
        lambda_penalty: float = 5.0,
        temperature: float = 1.0,
        delta_clip: float = 10.0,  # ✅ 添加delta裁剪阈值
        max_penalty: float = 50.0,  # ✅ 最大penalty值
        min_gate_value: float = 1e-6  # ✅ 最小gate值
    ):
        """
        Args:
            split_mask: The split conservation mask (α)
            lambda_penalty: Scaling factor λ for conservation violations
            temperature: Temperature for soft gating
            delta_clip: Maximum delta value (prevent gradient vanishing)
            max_penalty: Maximum total penalty (prevent exp underflow)
            min_gate_value: Minimum gate value (prevent log(0))
        """
        super().__init__()

        self.split_mask = split_mask
        self.lambda_penalty = lambda_penalty
        self.temperature = temperature
        self.delta_clip = delta_clip
        self.max_penalty = max_penalty
        self.min_gate_value = min_gate_value
    
    def compute_embedding_mismatch(
        self,
        incoming_embeddings: torch.Tensor,
        outgoing_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute conservation mismatch in EMBEDDING SPACE (V8 key innovation)

        Instead of checking Q, L, B separately, we compute:
        Δ_k = |Σ E_in[k] - Σ E_out[k]|

        For the fixed dimensions (k=0,1), this is equivalent to Q, L conservation.
        For learnable dimensions (k=2..7), the model must learn what to conserve.

        Args:
            incoming_embeddings: [N_in, embedding_dim] particle embeddings flowing IN
            outgoing_embeddings: [N_out, embedding_dim] particle embeddings flowing OUT

        Returns:
            mismatch: [embedding_dim] conservation mismatch per dimension
        """
        # Sum over particles
        sum_in = incoming_embeddings.sum(dim=0)  # [embedding_dim]
        sum_out = outgoing_embeddings.sum(dim=0)  # [embedding_dim]

        # Mismatch per dimension
        delta = torch.abs(sum_in - sum_out)  # [embedding_dim]

        return delta

    def forward(
        self,
        incoming_embeddings: torch.Tensor,
        outgoing_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Meta-Physics Gate value: Γ = exp(-λ Σ_k α_k Δ_k²)
        
        ✅ FIXED: Added numerical stability protections

        Args:
            incoming_embeddings: [N_in, D] embeddings of incoming particles
            outgoing_embeddings: [N_out, D] embeddings of outgoing particles

        Returns:
            gate_value: Scalar in (0, 1], higher = more conserved
        """
        # Get conservation mask α
        alpha = self.split_mask()  # [embedding_dim]

        # Compute mismatch per dimension
        delta = self.compute_embedding_mismatch(incoming_embeddings, outgoing_embeddings)
        
        # ✅ FIX 1: 裁剪delta防止极端值导致梯度消失
        delta = torch.clamp(delta, max=self.delta_clip)

        # Weighted penalty: α_k * Δ_k²
        weighted_penalty = alpha * (delta ** 2)
        total_penalty = weighted_penalty.sum()
        
        # ✅ FIX 2: 裁剪total_penalty防止exp(-大值)下溢
        # exp(-50) ≈ 2e-22 (接近float32极限)
        total_penalty = torch.clamp(total_penalty, max=self.max_penalty)

        # Exponential gate
        gate_value = torch.exp(-self.lambda_penalty * total_penalty / self.temperature)
        
        # ✅ FIX 3: 确保gate_value有下界，防止后续log(gate_value)产生-inf
        gate_value = torch.clamp(gate_value, min=self.min_gate_value)

        return gate_value


class PhysicsGatedPolicyHead(nn.Module):
    """
    Policy head with integrated Meta-Physics Gate
    Outputs π(a|G) = π_θ(a|G) · Γ(a) (normalized)

    V8 Innovation: Uses particle embeddings and conservation mask to gate particle selection
    """

    def __init__(
        self,
        embedding_dim: int,
        particle_embedding: 'SplitParticleEmbedding',
        meta_physics_gate: 'MetaPhysicsGate',
        num_action_types: int = 4,
        num_particle_types: int = 20,
        max_vertices: int = 10
    ):
        super().__init__()

        self.num_action_types = num_action_types
        self.num_particle_types = num_particle_types
        self.max_vertices = max_vertices

        # V8: Store references to shared components
        self.particle_embedding = particle_embedding
        self.meta_physics_gate = meta_physics_gate

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

        # Cache particle list
        self.particle_list = [p.id for p in PhysicsConstants.get_all_particles()] + \
                             list(PhysicsConstants.BOSONS.keys())

    def forward(
        self,
        graph_embedding: torch.Tensor,
        vertex_states: Optional[List[Dict]] = None,
        apply_physics_gate: bool = True,
        step_count: int = 0  # ✅ EARLY TERMINATION PENALTY: Add step_count parameter
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Meta-Physics Gate

        Args:
            graph_embedding: [embedding_dim] graph representation
            vertex_states: Current vertex states for gate computation
            apply_physics_gate: Whether to apply Meta-Physics Gate (set False during rollout)
            step_count: Current step count for early termination penalty

        Returns:
            Dictionary with action logits, probabilities, and gate values
        """
        # Raw neural network outputs
        action_type_logits = self.action_type_head(graph_embedding)
        vertex_logits = self.vertex_head(graph_embedding)
        particle_logits = self.particle_head(graph_embedding)

        # V8: Apply Meta-Physics Gate to particle logits during training
        # This modulates the policy based on conservation laws in embedding space
        gate_values = None
        if apply_physics_gate and vertex_states is not None:
            # Compute gate values for each particle candidate
            gate_values = self._compute_physics_gate_values(vertex_states)

            # Modulate particle logits: log π' = log π + log Γ
            # (equivalent to π' = π · Γ, but numerically stable)
            particle_logits = particle_logits + torch.log(gate_values + 1e-8)

        # ✅ EARLY TERMINATION PENALTY: Block Termination Action if step_count < 2
        # ACTION_TERMINATE = 3 (the last action type)
        if step_count < 2:
            # Set termination logit to -inf to make probability ~0
            action_type_logits[..., 3] = float('-inf')

        # Softmax to get probabilities
        action_type_probs = F.softmax(action_type_logits, dim=-1)
        vertex_probs = F.softmax(vertex_logits, dim=-1)
        particle_probs = F.softmax(particle_logits, dim=-1)

        output = {
            'action_type_logits': action_type_logits,
            'action_type_probs': action_type_probs,
            'vertex_logits': vertex_logits,
            'vertex_probs': vertex_probs,
            'particle_logits': particle_logits,
            'particle_probs': particle_probs
        }

        if gate_values is not None:
            output['gate_values'] = gate_values

        return output

    def _compute_physics_gate_values(self, vertex_states: List[Dict]) -> torch.Tensor:
        """
        Compute Meta-Physics Gate values for all particle candidates

        For each particle type, compute Γ(adding this particle to vertex)
        This forces the model to learn conservation through gate gradients

        Returns:
            gate_values: [num_particle_types] gate values for each particle
        """
        # Get current vertex state (assume we're modifying the first interaction vertex)
        # In a more sophisticated version, this would be vertex-specific
        vertex_state = vertex_states[0] if vertex_states else {'incoming': [], 'outgoing': []}

        incoming_edges = vertex_state.get('incoming', [])
        outgoing_edges = vertex_state.get('outgoing', [])

        # ✅ BUG FIX 11: Get particle IDs AND is_anti flags
        incoming_particle_ids = [e.get('particle_id', 'photon') for e in incoming_edges]
        incoming_is_anti = [e.get('is_anti', False) for e in incoming_edges]
        
        outgoing_particle_ids = [e.get('particle_id', 'photon') for e in outgoing_edges]
        outgoing_is_anti = [e.get('is_anti', False) for e in outgoing_edges]

        # Convert to indices and get embeddings
        if incoming_particle_ids:
            incoming_indices = torch.tensor([self.particle_list.index(pid) for pid in incoming_particle_ids],
                                           device=self.particle_embedding.learnable_embedding.device)
            incoming_embs = self.particle_embedding(incoming_indices)
            
            # ✅ BUG FIX 11: Flip antiparticle embeddings (ALL dimensions)
            for i, is_anti in enumerate(incoming_is_anti):
                if is_anti:
                    incoming_embs[i] *= -1
        else:
            incoming_embs = torch.zeros(1, self.particle_embedding.total_dim,
                                       device=self.particle_embedding.learnable_embedding.device)

        if outgoing_particle_ids:
            outgoing_indices = torch.tensor([self.particle_list.index(pid) for pid in outgoing_particle_ids],
                                           device=self.particle_embedding.learnable_embedding.device)
            outgoing_embs = self.particle_embedding(outgoing_indices)
            
            # ✅ BUG FIX 11: Flip antiparticle embeddings (ALL dimensions)
            for i, is_anti in enumerate(outgoing_is_anti):
                if is_anti:
                    outgoing_embs[i] *= -1
        else:
            outgoing_embs = torch.zeros(1, self.particle_embedding.total_dim,
                                       device=self.particle_embedding.learnable_embedding.device)

        # Compute gate value for each candidate particle
        gate_values = torch.zeros(self.num_particle_types,
                                 device=self.particle_embedding.learnable_embedding.device)

        for i in range(self.num_particle_types):
            # Get embedding for candidate particle
            candidate_idx = torch.tensor([i], device=self.particle_embedding.learnable_embedding.device)
            candidate_emb = self.particle_embedding(candidate_idx)

            # Hypothetical outgoing if we add this particle
            hypothetical_outgoing = torch.cat([outgoing_embs, candidate_emb], dim=0)

            # Compute gate value: Γ = exp(-λ Σ α_k (Δ_k)²)
            gate_value = self.meta_physics_gate(incoming_embs, hypothetical_outgoing)
            gate_values[i] = gate_value

        return gate_values
    
    def _compute_physics_gate_values_batched(self, vertex_states_list: List[List[Dict]]) -> torch.Tensor:
        """
        🚀 VECTORIZED: Compute gate values for all envs × all particles in one GPU call
        
        OLD: 2000 Python loops (100 envs × 20 particles)
        NEW: 1 matrix operation
        
        Returns:
            gate_values: [num_envs, num_particle_types]
        """
        num_envs = len(vertex_states_list)
        device = self.particle_embedding.learnable_embedding.device
        
        # 1. Pre-compute all particle embeddings: [num_particle_types, embed_dim]
        all_particle_indices = torch.arange(self.num_particle_types, device=device)
        all_particle_embs = self.particle_embedding(all_particle_indices)
        
        # 2. Batch collect incoming/outgoing embedding sums for each env
        incoming_sums = []
        outgoing_sums = []
        
        for vertex_states in vertex_states_list:
            # ✅ FIX Bug A: Don't hardcode [0] - use the FIRST vertex that has particles
            # This allows the Physics Gate to check conservation at the appropriate vertex
            vertex_state = None
            if vertex_states:
                # Find first non-empty vertex (has incoming or outgoing particles)
                for vs in vertex_states:
                    if vs.get('incoming') or vs.get('outgoing'):
                        vertex_state = vs
                        break
                # Fallback if all vertices are empty
                if vertex_state is None:
                    vertex_state = vertex_states[0] if vertex_states else {'incoming': [], 'outgoing': []}
            else:
                vertex_state = {'incoming': [], 'outgoing': []}
            
            # Incoming sum
            in_ids = [e.get('particle_id', 'photon') for e in vertex_state.get('incoming', [])]
            in_is_anti = [e.get('is_anti', False) for e in vertex_state.get('incoming', [])]  # ✅ BUG FIX 2a
            
            if in_ids:
                in_idx = torch.tensor([self.particle_list.index(p) for p in in_ids], device=device)
                in_embs = self.particle_embedding(in_idx)  # [N_in, embed_dim]
                
                # ✅ BUG FIX 9: Flip ALL dimensions for antiparticles
                # Antiparticles have OPPOSITE quantum numbers: Q → -Q, L → -L, B → -B
                # Since learnable dims may encode B (which we want model to discover),
                # they must also be flipped for antiparticles!
                for i, is_anti in enumerate(in_is_anti):
                    if is_anti:
                        in_embs[i] *= -1  # Flip ALL dimensions, not just fixed!
                
                in_sum = in_embs.sum(dim=0)
            else:
                in_sum = torch.zeros(self.particle_embedding.total_dim, device=device)
            incoming_sums.append(in_sum)
            
            # Outgoing sum
            out_ids = [e.get('particle_id', 'photon') for e in vertex_state.get('outgoing', [])]
            out_is_anti = [e.get('is_anti', False) for e in vertex_state.get('outgoing', [])]  # ✅ BUG FIX 2a
            
            if out_ids:
                out_idx = torch.tensor([self.particle_list.index(p) for p in out_ids], device=device)
                out_embs = self.particle_embedding(out_idx)  # [N_out, embed_dim]
                
                # ✅ BUG FIX 9: Flip ALL dimensions for antiparticles
                for i, is_anti in enumerate(out_is_anti):
                    if is_anti:
                        out_embs[i] *= -1  # Flip ALL dimensions, not just fixed!
                
                out_sum = out_embs.sum(dim=0)
            else:
                out_sum = torch.zeros(self.particle_embedding.total_dim, device=device)
            outgoing_sums.append(out_sum)
        
        # Stack: [num_envs, embed_dim]
        incoming_batch = torch.stack(incoming_sums)
        outgoing_batch = torch.stack(outgoing_sums)
        
        # 3. Vectorized computation of delta for all env×particle combinations
        # delta[i,j,k] = incoming[i,k] - (outgoing[i,k] + particle[j,k])
        # Using broadcasting: [num_envs, 1, D] - [num_envs, 1, D] - [1, num_particles, D]
        delta = incoming_batch.unsqueeze(1) - outgoing_batch.unsqueeze(1) - all_particle_embs.unsqueeze(0)
        # delta: [num_envs, num_particle_types, embed_dim]
        
        # 4. Compute gate: exp(-λ * Σ_k α_k * δ_k²)
        alpha = self.meta_physics_gate.split_mask()  # [embed_dim]
        weighted_penalty = alpha * (delta ** 2)  # [num_envs, num_particles, embed_dim]
        total_penalty = weighted_penalty.sum(dim=-1)  # [num_envs, num_particles]
        
        # ✅ FIX Bug B: Add missing temperature division for proper constraint scaling
        gate_values = torch.exp(-self.meta_physics_gate.lambda_penalty * total_penalty / self.meta_physics_gate.temperature)
        
        return gate_values  # [num_envs, num_particle_types]
    
    def forward_batch(
        self,
        graph_embeddings: torch.Tensor,
        vertex_states_list: Optional[List[List[Dict]]] = None,
        apply_physics_gate: bool = True,
        step_counts: Optional[torch.Tensor] = None  # ✅ EARLY TERMINATION PENALTY: [num_envs]
    ) -> Dict[str, torch.Tensor]:
        """
        🚀 OPTIMIZED: Batched forward pass for multiple environments
        
        This is the KEY OPTIMIZATION - processes all environments in a single GPU call
        instead of looping through them sequentially.
        
        Args:
            graph_embeddings: [num_envs, embedding_dim] batched graph representations
            vertex_states_list: List of vertex_states for each environment (for Physics Gate)
            apply_physics_gate: Whether to apply Meta-Physics Gate 
            step_counts: [num_envs] tensor of current step counts for early termination penalty
            
        Returns:
            Dictionary with batched outputs:
                - action_type_logits: [num_envs, num_action_types]
                - action_type_probs: [num_envs, num_action_types]
                - vertex_logits: [num_envs, max_vertices]
                - vertex_probs: [num_envs, max_vertices]
                - particle_logits: [num_envs, num_particle_types]
                - particle_probs: [num_envs, num_particle_types]
                - gate_values: [num_envs, num_particle_types] (if apply_physics_gate)
        """
        num_envs = graph_embeddings.shape[0]
        
        # Batch process neural network outputs (FAST on GPU!)
        action_type_logits = self.action_type_head(graph_embeddings)  # [num_envs, num_action_types]
        vertex_logits = self.vertex_head(graph_embeddings)  # [num_envs, max_vertices]
        particle_logits = self.particle_head(graph_embeddings)  # [num_envs, num_particle_types]
        
        # 🚀 V8 VECTORIZED: Single GPU call instead of 2000 Python loops!
        gate_values_batch = None
        if apply_physics_gate and vertex_states_list is not None:
            # ONE matrix operation replaces 100 envs × 20 particles loops
            gate_values_batch = self._compute_physics_gate_values_batched(vertex_states_list)
            
            # Modulate particle logits: log π' = log π + log Γ
            particle_logits = particle_logits + torch.log(gate_values_batch + 1e-8)
        
        # ✅ EARLY TERMINATION PENALTY: Block Termination Action for envs with step_count < 2
        # ACTION_TERMINATE = 3 (the last action type)
        if step_counts is not None:
            # Create mask: True for envs where step_count < 2
            early_mask = step_counts < 2  # [num_envs]
            # Set termination logit to -inf for these envs
            action_type_logits[early_mask, 3] = float('-inf')
        
        # Softmax to get probabilities (batched)
        action_type_probs = F.softmax(action_type_logits, dim=-1)
        vertex_probs = F.softmax(vertex_logits, dim=-1)
        particle_probs = F.softmax(particle_logits, dim=-1)
        
        output = {
            'action_type_logits': action_type_logits,
            'action_type_probs': action_type_probs,
            'vertex_logits': vertex_logits,
            'vertex_probs': vertex_probs,
            'particle_logits': particle_logits,
            'particle_probs': particle_probs
        }
        
        if gate_values_batch is not None:
            output['gate_values'] = gate_values_batch
        
        return output



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
    Complete Feynman-GCPN V8 model combining:
    1. Split Particle Embedding (PQNE)
    2. Split Conservation Mask (CLDM)
    3. MPNN Encoder
    4. Meta-Physics Gate
    5. Policy Head
    6. Value Head
    """

    def __init__(
        self,
        node_input_dim: int = 9,  # Updated for new node features
        edge_input_dim: int = 21,
        hidden_dim: int = 128,
        num_mp_layers: int = 3,
        num_action_types: int = 4,
        num_particle_types: int = 20,
        max_vertices: int = 10,
        lambda_penalty: float = 5.0,
        fixed_dim: int = 2,
        learnable_dim: int = 6,
        sparsity_weight: float = 0.001
    ):
        super().__init__()

        self.num_particle_types = num_particle_types
        self.sparsity_weight = sparsity_weight

        # V8 Components: Split Embedding and Mask
        self.particle_embedding = SplitParticleEmbedding(
            num_particles=num_particle_types,
            fixed_dim=fixed_dim,
            learnable_dim=learnable_dim
        )

        self.conservation_mask = SplitConservationMask(
            fixed_dim=fixed_dim,
            learnable_dim=learnable_dim
        )

        # MPNN Encoder
        self.encoder = FeynmanMPNN(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mp_layers
        )

        # Meta-Physics Gate
        self.meta_physics_gate = MetaPhysicsGate(
            split_mask=self.conservation_mask,
            lambda_penalty=lambda_penalty
        )

        # Policy and Value Heads
        # V8 CRITICAL FIX: Pass particle_embedding and meta_physics_gate to enable gate gradients
        self.policy_head = PhysicsGatedPolicyHead(
            embedding_dim=hidden_dim,
            particle_embedding=self.particle_embedding,
            meta_physics_gate=self.meta_physics_gate,
            num_action_types=num_action_types,
            num_particle_types=num_particle_types,
            max_vertices=max_vertices
        )

        self.value_head = ValueHead(embedding_dim=hidden_dim)
    
    def forward(
        self,
        data: Data,
        vertex_states: Optional[List[Dict]] = None,
        return_value: bool = True,
        step_count: int = 0  # ✅ EARLY TERMINATION PENALTY: Add step_count parameter
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass
        
        Args:
            data: PyG Data object
            vertex_states: Quantum number states for physics gating
            return_value: Whether to compute value estimate
            step_count: Current step count for early termination penalty
            
        Returns:
            Dictionary with policy outputs and optionally value
        """
        # Encode graph
        node_embeddings, graph_embedding = self.encoder(data)
        
        # ✅ BUG FIX 10b: Handle 2D graph_embedding from encoder
        # encoder now always returns [batch_size, hidden_dim]
        # For single sample forward, squeeze to [hidden_dim]
        if graph_embedding.shape[0] == 1:
            graph_embedding_1d = graph_embedding.squeeze(0)
        else:
            graph_embedding_1d = graph_embedding
        
        # Policy (single sample version expects 1D)
        # ✅ EARLY TERMINATION PENALTY: Pass step_count to policy_head
        policy_output = self.policy_head(graph_embedding_1d, vertex_states, step_count=step_count)
        
        output = {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding,  # Keep original 2D for consistency
            **policy_output
        }
        
        # Value
        if return_value:
            value = self.value_head(graph_embedding)  # value_head can handle both 1D and 2D
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

    def get_conservation_metrics(self) -> Dict[str, torch.Tensor]:
        """
        Get metrics for analyzing conservation law discovery

        Returns:
            Dictionary with:
            - alpha_fixed: Fixed conservation confidences for Q, L
            - alpha_learnable: Learned conservation confidences
            - sparsity_loss: L1 penalty on alpha_learnable
            - learnable_embedding_norm: Norm of learnable embeddings
        """
        alpha = self.conservation_mask()
        fixed_dim = self.conservation_mask.fixed_dim

        metrics = {
            'alpha_fixed': alpha[:fixed_dim],
            'alpha_learnable': alpha[fixed_dim:],
            'sparsity_loss': self.conservation_mask.get_sparsity_loss(),
            'learnable_embedding_norm': self.particle_embedding.learnable_embedding.norm()
        }

        return metrics
