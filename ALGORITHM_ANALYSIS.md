# FeynmanForge RL Algorithm Analysis
**Date:** 2025-11-22
**Analysis:** Comprehensive review of topological learning capabilities

---

## Executive Summary

After thorough examination of your training code, I've identified **critical limitations** in the algorithm's ability to learn topological structures for particle reactions, particularly for **annihilation processes**. The algorithm currently **CANNOT** properly represent particle-antiparticle annihilation due to fundamental design constraints in the action space and graph representation.

**Key Finding:** Your algorithm has access to branching operations but they **DO NOT** work on initial and final state particles in a way that enables annihilation topology.

---

## 1. Current Algorithm Capabilities

### 1.1 Graph Representation
The algorithm represents Feynman diagrams as **Directed Acyclic Graphs (DAGs)**:

**Vertices:**
- `initial`: Source vertices for incoming particles (fixed at x=80)
- `final`: Sink vertices for outgoing particles (fixed at x=720)
- `interaction`: Internal vertices where particles interact

**Edges (Propagators):**
```python
{
    'source': vertex_id | None,   # None for initial particles
    'target': vertex_id | None,   # None for final particles
    'particle_id': str,
    'is_anti': bool,
    'is_external': bool,
    'state': 'open' | 'connected' | 'consumed'
}
```

### 1.2 Action Space (4 Operations)

#### **ACTION_CONNECT (Type 0)** - `feynman_env.py:215-302`
**Purpose:** Connect two vertices

**Behavior:**
- **Case A:** Two interaction vertices → Creates internal propagator
- **Case B:** External vertex ↔ Interaction vertex → Merges external line into interaction
- **Case C:** Two external vertices → **FORBIDDEN** (returns False, line 267)

**Critical Issue:** Initial and final particles cannot be directly connected to each other!

#### **ACTION_BRANCH (Type 1)** - `feynman_env.py:304-366`
**Purpose:** Create 3-way interaction vertex (emission/radiation)

**Algorithm:**
```python
def _execute_branch(self, vertex_idx, particle_type_idx):
    # Get open half-lines from the vertex
    open_lines = self._get_open_halflines(vertex_idx)
    if not open_lines: return False  # <-- CRITICAL CHECK

    # Create new interaction vertex
    new_vertex = {'type': 'interaction', ...}

    # Split incoming line into:
    # 1. Continuation of original particle
    # 2. Emitted particle (e.g., photon)
```

**What it can do:**
- `e⁻ → e⁻ + γ` (Bremsstrahlung)
- `q → q + g` (Gluon radiation)
- Any process where a particle emits another

**What it CANNOT do:**
- `e⁺ + e⁻ → vertex` (Annihilation - requires merging two lines, not splitting)
- Initial particles are pre-created as **external vertices** with **no open half-lines** to branch from

#### **ACTION_SET_TYPE (Type 2)** - `feynman_env.py:368-376`
**Purpose:** Change particle type on internal edges

**Restriction:** `if self.edges[edge_idx]['is_external']: return False` (line 373)
- Cannot modify initial/final particle types
- Only works on internal propagators

#### **ACTION_TERMINATE (Type 3)** - `feynman_env.py:176-183`
**Purpose:** End diagram construction and evaluate

---

## 2. Initial State Representation Analysis

### 2.1 How Initial Particles Are Created - `feynman_env.py:100-166`

```python
def reset(self, seed=None, options=None):
    # Step 1: Create initial state VERTICES
    for i, particle_id in enumerate(self.initial_particles):
        vertex = {
            'id': i,
            'type': 'initial',  # <-- Tagged as EXTERNAL
            'x': 80,            # Fixed left boundary
            'y': spacing * (i + 1)
        }
        self.vertices.append(vertex)

    # Step 2: Create initial state EDGES (external lines)
    for i, particle_str in enumerate(self.initial_particles):
        p_id, is_anti = parse_particle_id(particle_str)
        edge = {
            'source': self.vertices[i]['id'],  # <-- Connected to initial vertex
            'target': None,                     # <-- Open-ended (half-line)
            'particle_id': p_id,
            'is_anti': is_anti,
            'is_external': True,  # <-- Marked as EXTERNAL
            'state': 'open'       # <-- Available for connection
        }
```

**Key Observation:** Initial particles are created as **boundary conditions**, not as entities that can be branched or merged internally.

### 2.2 The Open Half-Line Problem

The `_get_open_halflines()` function - `feynman_env.py:378-384`:
```python
def _get_open_halflines(self, vertex_idx):
    open_lines = []
    for edge_id in self.vertices[vertex_idx]['connected_edges']:
        edge = self.edges[edge_id]
        if edge['state'] == 'open':  # <-- Only returns 'open' edges
            open_lines.append(edge_id)
    return open_lines
```

**For initial vertices:**
- The initial vertex has ONE edge connected to it
- That edge has `source=vertex_id, target=None, state='open'`
- So `_get_open_halflines(initial_vertex)` returns **this edge**

**But here's the issue:**
When you call `ACTION_BRANCH` on an initial vertex, it tries to:
1. Get the open half-line (the initial particle edge)
2. Create a new interaction vertex
3. Connect the initial edge to this new vertex
4. Create two NEW outgoing edges from the new vertex

**This creates a topology like:**
```
[Initial Vertex] --e⁻--> [New Interaction] --e⁻--> (open)
                                            └--γ--> (open)
```

**This is NOT annihilation! This is emission/splitting!**

---

## 3. Annihilation Topology Requirements

### 3.1 What Annihilation Should Look Like

For `e⁺ + e⁻ → μ⁺ + μ⁻` (via photon), the topology should be:

```
[Initial e⁻] ----→ [Vertex A] ----γ---→ [Vertex B] ----→ [Final μ⁻]

[Initial e⁺] ----→ [Vertex A]           [Vertex B] ----→ [Final μ⁺]
```

**Key property:** TWO initial particles converge at the SAME vertex (Vertex A).

### 3.2 Why Current Algorithm Cannot Do This

**Problem 1: No "Merge" Action**
- `ACTION_BRANCH` splits one line into two (1 → 2)
- `ACTION_CONNECT` connects endpoints but doesn't merge flows (1 + 1 → propagator, not vertex)
- **Missing:** An action that merges two initial particles into a single interaction vertex (2 → 1)

**Problem 2: External Vertex Connection Restriction**
In `_execute_connect()` line 267:
```python
if v1_is_ext and v2_is_ext:
    return False  # Cannot connect two external vertices
```

This **explicitly prevents** two initial particles from meeting!

**Problem 3: Graph Construction Order**
The algorithm builds diagrams by:
1. Starting with fixed initial/final vertices
2. Creating intermediate interaction vertices
3. Connecting them with propagators

But for annihilation, you need:
1. Create an interaction vertex
2. **Pull two initial particles into it** (not supported)
3. Create outgoing propagators

---

## 4. Specific Analysis of Branching Behavior

### 4.1 Test Case: Can Initial Particles Be Branched?

**Scenario:** `e⁺ + e⁻ → ???` (Initial state)

**Initial graph structure:**
```python
vertices = [
    {'id': 0, 'type': 'initial', 'x': 80, 'y': 200, 'connected_edges': [0]},
    {'id': 1, 'type': 'initial', 'x': 80, 'y': 300, 'connected_edges': [1]}
]
edges = [
    {'id': 0, 'source': 0, 'target': None, 'particle_id': 'e', 'is_anti': True, 'state': 'open'},
    {'id': 1, 'source': 1, 'target': None, 'particle_id': 'e', 'is_anti': False, 'state': 'open'}
]
```

**Agent action:** `ACTION_BRANCH` on vertex 0

**What happens (feynman_env.py:304-366):**
```python
vertex_idx = 0  # e⁺ initial vertex
open_lines = [0]  # The e⁺ half-line

# Get incoming particle properties
edge_to_connect = edges[0]  # e⁺ line
incoming_pid = 'e'
incoming_is_anti = True

# Create new interaction vertex
new_vertex = {
    'id': 2,
    'type': 'interaction',
    'x': 130,  # vertex_x + 50
    'y': 200
}

# Create continuation edge (e⁺ continues)
new_edge1 = {
    'id': 2,
    'source': 2,  # New interaction vertex
    'target': None,
    'particle_id': 'e',
    'is_anti': True,
    'state': 'open'
}

# Create emitted particle edge (e.g., photon)
new_edge2 = {
    'id': 3,
    'source': 2,
    'target': None,
    'particle_id': 'photon',  # Determined by particle_type_idx
    'is_anti': False,
    'state': 'open'
}

# Connect initial line to new vertex
edges[0]['target'] = 2  # e⁺ now goes INTO the interaction vertex
edges[0]['state'] = 'connected'
```

**Result topology:**
```
[Initial e⁺] --e⁺--> [Interaction(2)] --e⁺--> (open)
                                      └--γ---> (open)
```

**This is STILL NOT annihilation!** It's just the e⁺ emitting a photon before it does anything else.

### 4.2 Can Two Initial Particles Meet?

**Attempt 1:** Use `ACTION_CONNECT` to connect vertex 0 and vertex 1
```python
_execute_connect(vertex_idx=0, target_idx=1)

# Line 267 check:
v1_is_ext = vertices[0]['type'] == 'initial'  # True
v2_is_ext = vertices[1]['type'] == 'initial'  # True

if v1_is_ext and v2_is_ext:
    return False  # ❌ FORBIDDEN
```

**Attempt 2:** First branch each initial particle, then connect the branches
```python
# After branching both:
vertices = [
    {'id': 0, 'type': 'initial'},       # e⁺ source
    {'id': 1, 'type': 'initial'},       # e⁻ source
    {'id': 2, 'type': 'interaction'},   # e⁺ branch point
    {'id': 3, 'type': 'interaction'}    # e⁻ branch point
]

# Now connect vertex 2 and 3
_execute_connect(vertex_idx=2, target_idx=3)

# Both are interaction vertices, so this creates:
new_edge = {
    'source': 2,
    'target': 3,
    'particle_id': 'photon',
    'is_external': False
}
```

**Result:**
```
[Initial e⁺] --→ [V2] --photon--> [V3] --→ ...
                  └--e⁺--> (open)   └--e⁻--> (open)
```

**Still not annihilation!** The e⁺ and e⁻ don't actually meet at the same vertex.

---

## 5. Does the Algorithm Have Access to Annihilation?

### **Short Answer: NO** ❌

### **Long Answer:**

The algorithm **theoretically** has the components to represent annihilation graphs, but **practically** cannot construct them due to:

1. **Structural Constraint:** Initial particles are created as fixed boundary vertices, not as mobile entities
2. **Action Limitation:** No action can merge two initial particles into a single interaction vertex
3. **Connection Restriction:** External vertices cannot be directly connected (line 267)
4. **Branching Direction:** `ACTION_BRANCH` only creates diverging topologies (1→2), not converging ones (2→1)

**What the algorithm CAN learn:**
- Scattering with intermediate states: `A + B → [vertex] → C + D`
- Particle decay: `A → B + C`
- Radiation: `A → A + γ`
- Sequential interactions with propagators

**What the algorithm CANNOT learn:**
- Direct annihilation: `e⁺ + e⁻ → [common vertex] → γ`
- Pair production from initial particles
- Any topology where initial particles must share a common vertex

---

## 6. Evidence from Generated Diagrams

Looking at `/diagrams/current_best.json`:

```json
{
  "shapes": [
    {"id": "initial_0", "particleId": "e", "p1": {"x": 50}, "p2": {"x": 150}},
    {"id": "initial_1", "particleId": "e", "p1": {"x": 50}, "p2": {"x": 150}},
    {"id": "final_0", "particleId": "mu", "p1": {"x": 650}, "p2": {"x": 750}},
    {"id": "final_1", "particleId": "mu", "p1": {"x": 650}, "p2": {"x": 750}}
  ]
}
```

**Observation:** Only external lines are present. No internal vertices or propagators.

**Why?** The agent likely learned that:
1. Attempting to create interactions gets penalized (conservation violations, invalid actions)
2. Terminating immediately with just external lines avoids negative rewards
3. There's no path to creating a valid annihilation diagram, so it doesn't try

---

## 7. Physics Gate Analysis

The Physics Gate (`models.py:125-224`) is designed to suppress conservation-violating actions:

```python
def forward(self, mismatch_vector):
    # mismatch_vector = [ΔQ, ΔL, ΔB, ΔColor]
    weighted_penalty = Σ w_k * (Δ_k)²
    gate_value = exp(-λ * weighted_penalty / T)
    return gate_value  # In (0, 1]
```

**However, note lines 294-331:**
```python
# Disabled physics masking - was using incorrect vertex assumption
# if mask_invalid and vertex_states is not None:
#     particle_logits = self.apply_physics_mask(...)

# The original implementation incorrectly assumed vertex_states[0] was the
# target vertex, but vertex[0] is typically the initial state particle which
# has different conservation requirements (source node with no incoming edges).
```

**Key Finding:** The physics gate is **DISABLED** in the current code!

This means:
- The model is not receiving physics-informed guidance during action selection
- Conservation laws are only checked during reward computation (after the action)
- The model must learn physics through trial-and-error rather than hard constraints

---

## 8. Training Reward Structure Analysis

From `feynman_env.py:59-73`:

```python
reward_weights = {
    'charge_violation': -0.5,
    'lepton_violation': -0.5,
    'baryon_violation': -0.5,
    'color_violation': -1.0,
    'interaction_violation': -0.5,
    'target_match': 20.0,           # ✓ Strong positive
    'topology_valid': 10.0,          # ✓ Strong positive
    'successful_connection': 2.0,    # ✓ Positive
    'vertex_created': 1.0,           # ✓ Positive
    'conservation_bonus': 2.0,       # ✓ Positive
    'complexity_penalty': -0.1,      # Encourages simple diagrams
    'step_penalty': 0.0,             # No cost for exploration
    'invalid_action': -0.5,          # Mild penalty
}
```

**Terminal Reward (`feynman_env.py:424-446`):**
```python
def _compute_terminal_reward(self):
    reward = 0.0
    is_connected = self._is_graph_connected()
    no_dangling = self._no_dangling_internal_lines()

    if not (is_connected and no_dangling):
        reward -= 10.0  # Heavy penalty for invalid topology
        return reward
    else:
        reward += self.reward_weights['topology_valid']  # +10

    initial_match = self._check_external_match(self.initial_particles, 'initial')
    final_match = self._check_external_match(self.final_particles, 'final')

    if initial_match and final_match:
        reward += self.reward_weights['target_match']  # +20
    else:
        reward -= 5.0  # -5 for mismatch
```

**Analysis:**
- The reward structure is **correct** and encourages valid diagrams
- But if the action space cannot construct annihilation, no amount of reward shaping will help
- The agent is essentially trying to maximize rewards within an **insufficient action space**

---

## 9. Critical Issues Summary

### Issue 1: **No Annihilation Action** 🚨
**Location:** `feynman_env.py:34-37`
```python
ACTION_CONNECT = 0
ACTION_BRANCH = 1
ACTION_SET_TYPE = 2
ACTION_TERMINATE = 3
```

**Missing:** `ACTION_MERGE` - Merge two half-lines into a single interaction vertex

**Required behavior:**
```python
def _execute_merge(self, vertex_idx1, vertex_idx2, emitted_particle_type):
    """
    Merge two vertices (typically initial particles) into a single interaction vertex.
    Example: e⁺ + e⁻ → [new vertex] → emitted_particle
    """
    # Get open lines from both vertices
    line1 = self._get_open_halflines(vertex_idx1)[0]
    line2 = self._get_open_halflines(vertex_idx2)[0]

    # Create new interaction vertex
    new_vertex = {'type': 'interaction', ...}

    # Connect both lines to new vertex
    self.edges[line1]['target'] = new_vertex['id']
    self.edges[line2]['target'] = new_vertex['id']

    # Create outgoing propagator (e.g., photon for annihilation)
    new_edge = {
        'source': new_vertex['id'],
        'target': None,
        'particle_id': emitted_particle_type
    }
```

### Issue 2: **External Vertex Connection Blocked** 🚨
**Location:** `feynman_env.py:267`
```python
if v1_is_ext and v2_is_ext:
    return False  # Cannot connect two external vertices
```

**Impact:** Initial particles cannot interact directly

**Fix:** Allow external vertices to connect if they converge (both have `target=None` or both have `source=None`)

### Issue 3: **Physics Gate Disabled** ⚠️
**Location:** `models.py:294-331`

The physics-informed action masking is commented out, reducing training efficiency

### Issue 4: **Insufficient Graph Flexibility** 🚨
Initial/final particles are created as **fixed boundary vertices** rather than as **flexible entities** that can be manipulated

**Alternative design:**
- Initial particles could be created as "free-floating" edges that can be attached to newly created vertices
- Vertices could be created first, then particles assigned to them

---

## 10. Recommendations

### 10.1 Immediate Fixes (High Priority)

#### **1. Add `ACTION_MERGE` to action space**

```python
# In feynman_env.py
ACTION_MERGE = 4  # New action type

def _execute_merge(self, vertex_idx1, vertex_idx2, particle_type_idx):
    """Merge two vertices into a new interaction vertex"""
    if vertex_idx1 >= len(self.vertices) or vertex_idx2 >= len(self.vertices):
        return False
    if vertex_idx1 == vertex_idx2:
        return False

    v1_open = self._get_open_halflines(vertex_idx1)
    v2_open = self._get_open_halflines(vertex_idx2)

    if not v1_open or not v2_open:
        return False

    # Create new interaction vertex at midpoint
    new_x = (self.vertices[vertex_idx1]['x'] + self.vertices[vertex_idx2]['x']) / 2
    new_y = (self.vertices[vertex_idx1]['y'] + self.vertices[vertex_idx2]['y']) / 2

    new_vertex = {
        'id': len(self.vertices),
        'type': 'interaction',
        'x': new_x,
        'y': new_y,
        'connected_edges': []
    }
    self.vertices.append(new_vertex)

    # Connect both half-lines to new vertex
    edge1 = self.edges[v1_open[0]]
    edge2 = self.edges[v2_open[0]]

    # Determine connection direction based on edge type
    if edge1['target'] is None:
        edge1['target'] = new_vertex['id']
    else:
        edge1['source'] = new_vertex['id']
    edge1['state'] = 'connected'

    if edge2['target'] is None:
        edge2['target'] = new_vertex['id']
    else:
        edge2['source'] = new_vertex['id']
    edge2['state'] = 'connected'

    new_vertex['connected_edges'].extend([edge1['id'], edge2['id']])

    # Create outgoing propagator
    emitted_pid = self.particle_list[particle_type_idx]
    new_edge = {
        'id': len(self.edges),
        'source': new_vertex['id'],
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
```

#### **2. Remove external vertex connection restriction**

```python
# In feynman_env.py:267, change to:
if v1_is_ext and v2_is_ext:
    # Allow connection if both are initial OR both are final
    if v1['type'] == v2['type']:
        # Create interaction vertex between them
        return self._merge_external_vertices(vertex_idx, target_idx)
    else:
        return False  # Can't connect initial to final directly
```

#### **3. Re-enable physics gate with correct vertex indexing**

```python
# In models.py, fix the vertex state extraction
def forward(self, graph_embedding, vertex_states, mask_invalid=True, target_vertex_idx=None):
    # Use target_vertex_idx instead of hardcoded vertex_states[0]
    if mask_invalid and vertex_states and target_vertex_idx is not None:
        particle_logits = self.apply_physics_mask(
            particle_logits,
            vertex_states[target_vertex_idx],  # Use actual target vertex
            self.particle_list
        )
```

### 10.2 Architectural Improvements (Medium Priority)

#### **1. Flexible initial particle representation**
Instead of creating fixed vertices, create initial particles as "free-floating" edges that must be attached:

```python
def reset(self):
    # Don't create vertices for initial particles
    # Just create floating edges
    for particle_str in self.initial_particles:
        p_id, is_anti = parse_particle_id(particle_str)
        edge = {
            'id': len(self.edges),
            'source': None,  # Must be attached to a vertex
            'target': None,
            'particle_id': p_id,
            'is_anti': is_anti,
            'is_external': True,
            'state': 'open'
        }
        self.edges.append(edge)
```

#### **2. Add "Create Vertex" action**
Allow agent to create interaction vertices anywhere, then attach particles to them:

```python
ACTION_CREATE_VERTEX = 5

def _execute_create_vertex(self, x, y):
    """Create a new interaction vertex at specified position"""
    new_vertex = {
        'id': len(self.vertices),
        'type': 'interaction',
        'x': x,
        'y': y,
        'connected_edges': []
    }
    self.vertices.append(new_vertex)

    # Create placeholder open ports (like Feynman diagram vertices have multiple ports)
    for _ in range(3):  # Most vertices have 3 connections
        port_edge = {
            'id': len(self.edges),
            'source': new_vertex['id'],
            'target': None,
            'particle_id': 'photon',  # Default, will be set later
            'is_external': False,
            'state': 'open'
        }
        self.edges.append(port_edge)
        new_vertex['connected_edges'].append(port_edge['id'])

    return True
```

### 10.3 Training Improvements (Low Priority)

#### **1. Curriculum learning**
Start with simpler topologies and gradually increase complexity:

```python
# Stage 1: Just decay (A → B + C)
# Stage 2: Simple scattering (A + B → C + D with one vertex)
# Stage 3: Annihilation (e⁺ + e⁻ → γ → μ⁺ + μ⁻)
# Stage 4: Complex multi-vertex diagrams
```

#### **2. Reward shaping for annihilation**
Add specific rewards for:
- Creating vertices that have multiple incoming lines (+5)
- Successfully merging initial particles (+10)
- Creating topologically correct annihilation diagrams (+30)

#### **3. Demonstration learning**
Provide expert demonstrations of correct annihilation diagrams and use imitation learning

---

## 11. Test Case: e⁺ + e⁻ → μ⁺ + μ⁻

### Current Algorithm Attempt:
```
[Initial e⁻] ----→ (open)
[Initial e⁺] ----→ (open)
[Final μ⁻] ←---- (open)
[Final μ⁺] ←---- (open)
```
**Agent terminates immediately:** No valid action sequence exists ❌

### With `ACTION_MERGE`:
```
Step 1: MERGE vertex 0 (e⁻) and vertex 1 (e⁺) → Creates interaction V2
[Initial e⁻] ----→ [V2] ----→ (open photon)
[Initial e⁺] ----→ [V2]

Step 2: SET_TYPE on photon edge → γ
[Initial e⁻] ----→ [V2] ----γ---→ (open)
[Initial e⁺] ----→ [V2]

Step 3: BRANCH the photon line → Creates V3
[V2] ----γ---→ [V3] ----μ⁻---→ (open)
                      └--μ⁺---→ (open)

Step 4: CONNECT μ⁻ to final vertex 2
Step 5: CONNECT μ⁺ to final vertex 3
Step 6: TERMINATE

Final topology:
[Initial e⁻] ----→ [V2] ----γ---→ [V3] ----μ⁻---→ [Final μ⁻]
[Initial e⁺] ----→ [V2]                └--μ⁺---→ [Final μ⁺]
```
**Valid annihilation diagram!** ✓

---

## 12. Conclusion

### **Does your algorithm have the ability to learn topological structure?**
**Partial YES:** It can learn:
- Emission/radiation topologies (1 → many)
- Sequential scattering with propagators
- Internal vertex arrangements

**But CRITICAL NO for annihilation:** It cannot learn:
- Converging topologies (many → 1)
- Particle-antiparticle annihilation
- Pair production from initial particles

### **Why is it not branching initial and final state particles?**
1. **Branching initial particles DOES work** but creates the wrong topology (emission, not annihilation)
2. **The algorithm lacks ACTION_MERGE** needed for annihilation
3. **External vertex connection is blocked** (line 267)
4. **No action can make two initial particles meet at the same vertex**

### **Does it have access to represent annihilation?**
**NO.** The current action space is **fundamentally insufficient** for annihilation topologies.

### **Recommended Next Steps:**
1. ✅ Implement `ACTION_MERGE` (highest priority)
2. ✅ Remove external vertex connection restriction
3. ✅ Re-enable physics gate with correct indexing
4. ✅ Test with simple annihilation reactions
5. ✅ Add curriculum learning for gradual complexity

---

## 13. Code Locations for Fixes

| Issue | File | Lines | Action |
|-------|------|-------|--------|
| Add ACTION_MERGE | `feynman_env.py` | 34-37 | Add new action type |
| Implement merge logic | `feynman_env.py` | After 366 | Add `_execute_merge()` |
| Remove connection block | `feynman_env.py` | 267 | Allow external vertex connections |
| Fix physics gate | `models.py` | 294-331 | Re-enable with correct vertex index |
| Update action space | `feynman_env.py` | 77-82 | Extend to 5 action types |
| Update training | `training.py` | 179, 273 | Handle new action type |

---

**Analysis complete.** The algorithm needs architectural changes to support annihilation topologies.
