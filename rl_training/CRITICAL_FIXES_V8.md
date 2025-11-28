# Critical Architecture Fixes for V8

This document details the two fatal architecture defects that were identified and fixed in the Feynman-GCPN V8 implementation.

## Background: The V8 Discovery Mechanism

The V8 architecture is designed to **discover** conservation laws (specifically Baryon number conservation) through three key innovations:

1. **Split Particle Embedding (PQNE)**: E(p) = [E_fixed(Q,L) ⊕ E_learnable(unknown)]
   - Fixed dimensions encode known laws (Charge, Lepton)
   - Learnable dimensions should discover Baryon conservation

2. **Split Conservation Mask (CLDM)**: α = [α_fixed ≈ 1.0 ⊕ α_learnable]
   - α_fixed tells model Q,L must conserve
   - α_learnable should learn which dimensions matter (discover B)

3. **Meta-Physics Gate**: Γ(a) = exp(-λ Σ_k α_k Δ_k²)
   - Modulates policy based on conservation in embedding space
   - Gradients flow through gate to α_learnable, enabling discovery

4. **Scientist Reward**: Immediate feedback for Q,L; sparse terminal reward for B
   - Model must learn B from sparse signal without explicit supervision

## Critical Defect #1: Meta-Physics Gate Not Connected

### The Problem

Although `MetaPhysicsGate` was carefully defined, it was **never called in the forward pass**. This meant:

- No gradients flowed to `α_learnable`
- Model could not discover conservation laws
- The entire V8 discovery mechanism was broken

### Location

`models.py` - `PhysicsGatedPolicyHead.forward()`

The gate was initialized in `FeynmanGCPN.__init__()` but never used:

```python
# Gate defined but never called!
self.meta_physics_gate = MetaPhysicsGate(...)

# Forward pass just computed policy without gate
particle_logits = self.particle_head(graph_embedding)
# Missing: Apply gate to modulate logits
```

### The Fix

**Modified `PhysicsGatedPolicyHead.__init__`** to accept shared references:

```python
def __init__(
    self,
    embedding_dim: int,
    particle_embedding,      # V8 FIX: Shared particle embedding
    meta_physics_gate,       # V8 FIX: Shared physics gate
    num_action_types: int,
    num_particle_types: int,
    max_vertices: int
):
    self.particle_embedding = particle_embedding
    self.meta_physics_gate = meta_physics_gate
```

**Added `_compute_physics_gate_values()` method**:

```python
def _compute_physics_gate_values(self, vertex_states: List[Dict]) -> torch.Tensor:
    """
    Compute Meta-Physics Gate values for all particle candidates

    For each particle type i:
    1. Get current incoming/outgoing particle embeddings
    2. Get embedding for candidate particle i
    3. Compute hypothetical outgoing = current_outgoing + candidate
    4. Compute gate value: Γ_i = exp(-λ Σ_k α_k (Δ_k)²)

    Returns:
        gate_values: [num_particle_types] gate values
    """
    # ... implementation computes gate for each candidate particle ...
```

**Modified `forward()` to apply gate**:

```python
def forward(self, graph_embedding, vertex_states, apply_physics_gate=True):
    # Base policy from neural network
    particle_logits = self.particle_head(graph_embedding)

    # V8 CRITICAL FIX: Apply Meta-Physics Gate to modulate particle selection
    if apply_physics_gate and vertex_states is not None:
        gate_values = self._compute_physics_gate_values(vertex_states)
        # Modulate logits: higher gate = higher probability
        particle_logits = particle_logits + torch.log(gate_values + 1e-8)

    particle_probs = F.softmax(particle_logits, dim=-1)
```

**Updated `FeynmanGCPN.__init__`** to pass shared references:

```python
self.policy_head = PhysicsGatedPolicyHead(
    embedding_dim=hidden_dim,
    particle_embedding=self.particle_embedding,  # V8 FIX
    meta_physics_gate=self.meta_physics_gate,    # V8 FIX
    num_action_types=num_action_types,
    num_particle_types=num_particle_types,
    max_vertices=max_vertices
)
```

### Why This Fix is Critical

Now gradients flow through the gate:

1. Policy selects particles → uses gate values
2. Gate values depend on α_learnable
3. Backprop updates α_learnable to maximize reward
4. Model learns which embedding dimensions matter for conservation
5. **Discovery of Baryon conservation becomes possible**

---

## Critical Defect #2: Multi-Task Training Broken

### The Problem

The trainer was supposed to cycle through multiple reactions (leptonic AND hadronic), but it **always used the first environment** (`envs[0]`). This meant:

- Model only trained on one reaction (e.g., `e+e_bar->mu+mu_bar`)
- **Never saw quarks** (hadronic reactions like `z->u+u_bar`)
- Could not possibly learn Baryon conservation without seeing quarks
- Multi-task dataset configuration was completely ignored

### Location

`training.py` - `PPOTrainer` class

```python
# In run_experiment.py:
trainer = PPOTrainer(env=envs[0], ...)  # Only passes first env
trainer.training_envs = envs  # Sets attribute but never used!

# In training.py:
def _collect_rollout_single(self, num_steps, deterministic):
    state, info = self.env.reset()  # Always uses self.env (envs[0])
    # ...
    next_state, reward, done = self.env.step(action)  # Always envs[0]
```

### The Fix

**Added multi-task support to `PPOTrainer.__init__`**:

```python
def __init__(self, env, model, ...):
    self.env = env  # Keep for backward compatibility

    # V8 CRITICAL FIX: Multi-task training support
    self.training_envs = None  # Will be set by run_experiment.py
    self.current_env_idx = 0
```

**Added environment cycling methods**:

```python
def _get_current_env(self):
    """
    V8 CRITICAL FIX: Get current training environment

    Cycles through training_envs if available (multi-task training)
    Otherwise uses the single self.env
    """
    if self.training_envs is not None and len(self.training_envs) > 0:
        return self.training_envs[self.current_env_idx]
    return self.env

def _cycle_training_env(self):
    """
    V8 CRITICAL FIX: Switch to next training environment

    This ensures the model sees all reactions including hadronic ones with quarks
    """
    if self.training_envs is not None and len(self.training_envs) > 1:
        self.current_env_idx = (self.current_env_idx + 1) % len(self.training_envs)
```

**Updated `_collect_rollout_single()` to use current env**:

```python
def _collect_rollout_single(self, num_steps, deterministic=False):
    episode_rewards = []
    episode_lengths = []

    # V8 FIX: Get current training environment (may cycle through reactions)
    current_env = self._get_current_env()
    state, info = current_env.reset()
    episode_reward = 0
    episode_length = 0

    for step in range(num_steps):
        # ... collect experience using current_env ...
        next_state, reward, done = current_env.step(action)

        if done:
            episode_rewards.append(episode_reward)

            # V8 CRITICAL FIX: Cycle to next training environment after episode
            # This ensures the model sees all reactions (leptonic AND hadronic)
            self._cycle_training_env()
            current_env = self._get_current_env()

            # Reset with new environment
            state, info = current_env.reset()
            episode_reward = 0
```

### Why This Fix is Critical

Now the model sees diverse reactions:

1. **Episode 1**: Train on `e+e_bar->mu+mu_bar` (leptons)
2. **Episode 2**: Train on `z->u+u_bar` (quarks with Baryon number!)
3. **Episode 3**: Train on `z->d+d_bar` (more quarks)
4. **Episode 4**: Cycle back to `e+e_bar->mu+mu_bar`
5. And so on...

Without seeing quarks, the model cannot possibly learn Baryon conservation because:
- Leptons all have B=0
- Baryon number only varies in hadronic reactions
- The learnable embedding dimensions have no signal to align with B

---

## Testing the Fixes

Run the verification script:

```bash
python test_v8_setup.py
```

Expected output:
```
============================================================
[SUCCESS] ALL TESTS PASSED!
============================================================

V8 architecture is working correctly!
```

Key indicators that fixes are working:

1. **Meta-Physics Gate Connected**:
   - Forward pass includes gate values
   - Conservation metrics show non-zero sparsity loss
   - Gradients flow to α_learnable

2. **Multi-Task Training Active**:
   - `_get_current_env()` cycles through all reactions
   - Model sees both leptonic and hadronic reactions
   - Training logs show varied reaction types

---

## Next Steps

1. **Run Quick Test**:
   ```bash
   python run_experiment.py --quick
   ```
   - 10k steps with 6 training reactions
   - Verifies end-to-end training works

2. **Run Full Training**:
   ```bash
   python run_experiment.py
   ```
   - 500k steps
   - Monitor α_learnable values
   - Check embedding alignment with Baryon number

3. **Analyze Discovery**:
   - Track `discovery_metrics` in TensorBoard
   - Check if α_learnable converges to select one dimension
   - Verify selected dimension correlates with true Baryon numbers

---

## Summary

Both defects were **architectural failures** that completely broke the V8 discovery mechanism:

1. **Defect 1**: Gate never called → no gradients → no discovery
2. **Defect 2**: Single-task training → no quark exposure → no Baryon signal

Both are now fixed. The model can now:
- See diverse reactions including quarks
- Receive gradients through the Meta-Physics Gate
- Learn to discover Baryon conservation from sparse rewards

**Status**: ✅ All tests passing. Ready for training.
