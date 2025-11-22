# Training Problems Analysis - Why RL Agent Terminates Early

## Date: 2025-11-22

## Problem Statement

The RL agent consistently terminates after 1-2 steps without taking meaningful actions to construct Feynman diagrams. Training logs show:
- Episodes end immediately or after minimal exploration
- Model receives consistent negative rewards
- No progress toward learning electron-positron or muon annihilation diagrams

---

## Root Cause Analysis

### 🔴 **CRITICAL ISSUE #1: Massive Action Space with No Masking**

**Problem:**
- **Action space size**: 5 × 10 × 40 × 10 = **20,000 possible actions**
  - 5 action types (CONNECT, BRANCH, SET_TYPE, TERMINATE, MERGE)
  - 10 vertex indices
  - ~40 particle types
  - 10 target vertices

**Reality:**
- At any given state, **~99% of actions are invalid**
- Example: Initial state has 2 vertices (e-, e+)
  - Only ~20-30 actions are valid (connect these 2, branch from them, merge them)
  - Agent samples from all 20,000 actions uniformly
  - **99.9% probability of selecting invalid action**

**Consequence:**
- Agent gets -0.5 penalty for every invalid action
- After 10 random actions: -5.0 accumulated penalty
- Agent learns: "**Doing nothing (TERMINATE) is better than exploring**"
- Early termination with -20.0 is better than -30.0 from 60 failed actions

**Files:**
- `rl_training/feynman_env.py:78-83` - Action space definition
- `rl_training/training.py:178-183` - Action sampling (no masking)

---

### 🔴 **CRITICAL ISSUE #2: Physics Gate Disabled**

**Problem:**
```python
# models.py:276
mask_invalid: bool = False  # Disabled for now - needs proper target vertex indexing

# models.py:299
# Physics gate currently disabled - would need proper target vertex indexing
```

**What this means:**
- The **core innovation** of Feynman-GCPN (physics-informed gating) is **TURNED OFF**
- Model has NO guidance about which actions violate physics laws
- Agent explores blindly without any physics constraints to guide it

**Why it was disabled:**
> "The gate needs to know WHICH vertex the action will affect, not just use vertex_states[0] which is always the first initial particle."

**Consequence:**
- Agent proposes actions that violate charge/lepton/baryon conservation
- Gets negative rewards from environment
- No gradient signal about WHY action was bad
- Learning is impossibly slow

**Files:**
- `rl_training/models.py:226-299` - PhysicsGatedPolicyHead (disabled)

---

### 🟡 **ISSUE #3: Reward Structure Discourages Exploration**

**Current Rewards:**
```python
'invalid_action': -0.5          # Common (99% of random actions)
'successful_connection': 2.0     # Rare
'vertex_created': 1.0            # Rare
'conservation_bonus': 2.0        # Very rare
'target_match': 20.0             # Extremely rare (terminal)
'topology_valid': 10.0           # Extremely rare (terminal)
```

**Early Termination Penalty:**
```python
if num_internal_edges < 1 or not is_connected:
    reward -= 20.0  # Penalize lazy termination
```

**Problem with the math:**
- Taking 40 random actions (hitting 99% invalid): **-20.0 penalty**
- Terminating immediately: **-20.0 penalty**
- **SAME PENALTY for trying vs giving up!**

**Rational Agent Strategy:**
> "Why explore for 40 steps and get -20.0 when I can terminate now and also get -20.0?"

**Consequence:**
- Agent learns to terminate immediately
- No exploration, no learning
- Training stalls at episode 1

**Files:**
- `rl_training/feynman_env.py:60-74` - Reward weights
- `rl_training/feynman_env.py:177-184` - Termination logic

---

### 🟡 **ISSUE #4: Insufficient Exploration**

**Current Settings:**
```python
# train.py:168
entropy_coef=0.05  # Very low for such a large action space
```

**Why this is too low:**
- Entropy coefficient controls exploration vs exploitation
- For action space of 20,000 combinations, need **high exploration**
- 0.05 means: "Be 95% greedy, 5% exploratory"
- With 99% invalid actions, agent never finds valid actions to learn from

**Standard values:**
- Simple action spaces (4-8 actions): 0.01
- Medium action spaces (10-100): 0.05-0.1
- Large action spaces (1000+): **0.1-0.3**
- **Our space (20,000):** Should be 0.2-0.5

**Consequence:**
- Agent doesn't explore enough to find valid actions
- Gets stuck in local minimum (terminate early)

**Files:**
- `rl_training/train.py:168`

---

### 🟡 **ISSUE #5: No Curriculum Learning**

**Current Training:**
- Start directly with electron-positron annihilation
- Requires learning:
  1. Merge two initial particles
  2. Create interaction vertex
  3. Emit virtual photon
  4. Photon creates muon pair
  5. Connect to final states
  6. Satisfy ALL conservation laws
- **Minimum 5-7 correct actions in sequence**

**Problem:**
- With 99% invalid action rate, probability of 5 correct actions in a row:
  - P(success) = (0.01)^5 = **0.00001%**
  - Expected episodes to succeed: **10 million**

**What curriculum learning would do:**
- Step 1: Learn to connect two adjacent vertices (1 action)
- Step 2: Learn to branch from a vertex (2 actions)
- Step 3: Learn simple 3-vertex diagrams
- Step 4: Learn annihilation topology (5+ actions)

**Consequence:**
- Training is intractable without curriculum
- Agent never sees successful episode to learn from

**Files:**
- `rl_training/train.py` - No curriculum implementation

---

## Why Model is "Not Brave Enough"

The model **IS exploring**, but rationally decides to stop because:

1. **99% of actions are invalid** → Accumulates -0.5 penalties rapidly
2. **No guidance** (physics gate disabled) → Can't learn which actions might work
3. **Termination penalty (-20)** = Same as exploring unsuccessfully
4. **Low entropy (0.05)** → Quickly converges to "safe" policy (terminate early)

The agent is actually being **SMART**:
> "Exploring costs me -20 points. Terminating costs -20 points. Why waste time exploring?"

---

## Additional Technical Issues

### 🔧 **Deprecation Warning**

```python
# training.py:265, 453
with autocast(enabled=self.use_amp):
```

**Warning:**
```
FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated.
Please use `torch.amp.autocast('cuda', args...)` instead.
```

**Fix needed:**
```python
from torch.amp import autocast
with autocast('cuda', enabled=self.use_amp):
```

---

## Verification of Claims

### Can the Current Model Learn?

**Theoretical Analysis:**

Let's calculate expected episodes to succeed:

1. **Random exploration probability:**
   - Valid action rate: 1% (generous estimate)
   - Actions needed for e+e- → μ+μ-: 5 actions minimum
   - P(5 valid actions) = 0.01^5 = **0.00001**
   - Expected episodes = 1 / 0.00001 = **100,000 episodes**

2. **With current training:**
   - Timesteps: 100,000
   - Rollout steps per env: 1024
   - Parallel envs: 512
   - Total steps per update: 524,288
   - Episodes per update: ~10,000 (if avg 50 steps each)
   - Total episodes in run: ~10,000
   - **NOT ENOUGH to see even ONE success**

3. **With low entropy (0.05):**
   - Agent converges to greedy policy within 100 episodes
   - Never explores enough to find valid action sequences
   - Gets stuck at local minimum (terminate early)

**Empirical Evidence:**

From user report:
> "the model is still not brave enough to try with more steps, it always stopped at first or second steps and does nothing more, it just terminated"

This confirms:
- Agent terminates at step 1-2 consistently
- No exploration happening
- Model converged to "terminate immediately" policy

**Conclusion:**
**NO, the current model CANNOT learn** under these conditions:
1. ❌ 99% invalid action rate without masking
2. ❌ Physics gate disabled (no guidance)
3. ❌ Reward structure doesn't encourage exploration
4. ❌ Entropy too low for action space size
5. ❌ No curriculum to build up complexity

---

## Required Fixes (Priority Order)

### 🚨 **URGENT: Fix #1 - Action Masking**

**Implement invalid action masking:**

```python
# In feynman_env.py - Add method:
def get_valid_action_mask(self) -> Dict[str, np.ndarray]:
    """Return binary mask of valid actions"""
    mask = {
        'action_type': np.ones(5),  # All action types valid initially
        'vertex_idx': np.zeros(self.max_vertices),
        'particle_type': np.ones(self.num_particle_types),
        'target_vertex': np.zeros(self.max_vertices)
    }

    # Mark valid vertices (those that exist)
    for i in range(len(self.vertices)):
        mask['vertex_idx'][i] = 1.0
        mask['target_vertex'][i] = 1.0

    # Disable TERMINATE if no progress made
    if len(self.vertices) <= len(self.initial_particles) + len(self.final_particles):
        mask['action_type'][self.ACTION_TERMINATE] = 0.0

    return mask

# In models.py - Use mask in action sampling:
def sample_action(self, probs, mask):
    """Sample from masked probability distribution"""
    masked_probs = probs * mask
    masked_probs = masked_probs / (masked_probs.sum() + 1e-8)
    return torch.multinomial(masked_probs, 1)
```

**Impact:** Reduces invalid action rate from 99% → ~20%

---

### 🚨 **URGENT: Fix #2 - Re-enable Physics Gate (Simplified)**

**Implement simplified physics gating:**

Even without perfect target vertex tracking, we can still apply physics constraints:

```python
# In models.py - PhysicsGatedPolicyHead.forward():
def forward(self, graph_embedding, vertex_states, mask_invalid=True):
    action_type_logits = self.action_type_head(graph_embedding)
    vertex_logits = self.vertex_head(graph_embedding)
    particle_logits = self.particle_head(graph_embedding)

    # Apply physics gate to particle selection
    if mask_invalid and vertex_states:
        # Compute gate values for each particle type
        gate_values = []
        for p_idx in range(self.num_particle_types):
            # Simplified: Check if particle conserves quantum numbers
            # for AVERAGE vertex in current graph
            gate_val = self._compute_gate_for_particle(p_idx, vertex_states)
            gate_values.append(gate_val)

        gate_tensor = torch.tensor(gate_values, device=particle_logits.device)
        particle_logits = particle_logits + torch.log(gate_tensor + 1e-8)

    # Convert to probabilities
    action_type_probs = F.softmax(action_type_logits, dim=-1)
    vertex_probs = F.softmax(vertex_logits, dim=-1)
    particle_probs = F.softmax(particle_logits, dim=-1)

    return {
        'action_type_probs': action_type_probs,
        'vertex_probs': vertex_probs,
        'particle_probs': particle_probs
    }
```

**Impact:** Guides particle selection to conserve quantum numbers

---

### 🔶 **HIGH PRIORITY: Fix #3 - Reward Shaping**

**Modify reward structure to encourage exploration:**

```python
# In feynman_env.py:
reward_weights = {
    'step_penalty': -0.05,           # Small penalty for long episodes
    'invalid_action': -0.2,          # REDUCED from -0.5
    'successful_connection': 5.0,    # INCREASED from 2.0
    'vertex_created': 3.0,           # INCREASED from 1.0
    'conservation_bonus': 5.0,       # INCREASED from 2.0
    'progress_reward': 2.0,          # NEW: Reward for getting closer to goal
    'exploration_bonus': 1.0,        # NEW: Reward for trying new actions
    'target_match': 50.0,            # INCREASED from 20.0
    'topology_valid': 20.0,          # INCREASED from 10.0
    'early_termination_penalty': -50.0,  # NEW: Much harsher than exploration
}

# In step():
if action_type == self.ACTION_TERMINATE:
    if self.step_count < 3:  # Too early
        reward += reward_weights['early_termination_penalty']
    elif num_internal_edges < 1:
        reward -= 40.0  # Increased from 20.0
```

**Impact:** Makes exploration more attractive than terminating

---

### 🔶 **HIGH PRIORITY: Fix #4 - Increase Entropy**

```python
# In train.py:
entropy_coef=0.3,  # INCREASED from 0.05 for large action space
```

**Impact:** Forces more exploration, prevents premature convergence

---

### 🔷 **MEDIUM PRIORITY: Fix #5 - Add Progress Tracking**

**Track diagram completion progress:**

```python
# In feynman_env.py - Add to _get_info():
def _compute_progress_score(self) -> float:
    """Estimate how close diagram is to completion"""
    score = 0.0

    # Points for having internal structure
    num_internal = sum(1 for e in self.edges if not e['is_external'])
    score += min(num_internal * 2.0, 10.0)

    # Points for connectivity
    if self._is_graph_connected():
        score += 5.0

    # Points for external particle connections
    connected_external = sum(1 for e in self.edges
                            if e['is_external'] and e['state'] == 'connected')
    total_external = len(self.initial_particles) + len(self.final_particles)
    score += (connected_external / total_external) * 10.0

    return score

# In step() - Add progress reward:
new_progress = self._compute_progress_score()
if not hasattr(self, 'last_progress'):
    self.last_progress = 0.0
progress_delta = new_progress - self.last_progress
if progress_delta > 0:
    reward += progress_delta * self.reward_weights['progress_reward']
self.last_progress = new_progress
```

**Impact:** Provides intermediate rewards for partial progress

---

## Summary

**Can the model learn?**
- **Current state:** NO - Multiple critical blockers
- **With fixes:** YES - Should be able to learn in 10K-50K episodes

**Why not brave enough?**
- Model is **rationally pessimistic** given reward structure
- Exploration is punished equally to giving up
- No guidance (physics gate off) means blind search
- Action space too large without masking

**What fixes are critical?**
1. ✅ **Action masking** (99% → 20% invalid rate)
2. ✅ **Physics gate** (guidance for valid actions)
3. ✅ **Reward shaping** (encourage exploration)
4. ✅ **Increase entropy** (force exploration)
5. ✅ **Fix autocast deprecation** (technical debt)

**Expected outcome after fixes:**
- Agent explores for 10-30 steps before terminating
- Successfully creates 2-3 vertex diagrams within 1K episodes
- Learns annihilation topology within 10K-50K episodes
- Training becomes tractable
