# Feynman-GCPN: Reinforcement Learning for Feynman Diagram Generation

A complete implementation of **Feynman-GCPN** (Graph Convolutional Policy Network), a novel Reinforcement Learning framework that learns to construct valid Feynman diagrams by navigating the topological space of particle interactions while respecting Standard Model conservation laws.

## 🚀 Overview

This project implements a **physics-informed RL agent** that:

1. **Constructs Feynman diagrams step-by-step** as sequential graph operations
2. **Enforces conservation laws** (charge, lepton number, baryon number, color charge) through a differentiable **Physics Gate**
3. **Learns valid particle interactions** purely from environment rewards, without hard-coded rules
4. **Visualizes results** in real-time using the existing JavaScript Feynman Forge frontend

## 📁 Project Structure

```
rl_training/
├── physics_engine.py          # Particle database & conservation law validators
├── feynman_env.py             # Gymnasium environment (MDP formulation)
├── models.py                  # MPNN encoder + Physics-Gated Policy Head
├── training.py                # PPO training loop
├── visualization_bridge.py    # Export diagrams to JSON for JS visualization
├── requirements.txt           # Python dependencies
└── README.md                  # This file

diagrams/                      # Output directory for generated diagrams
checkpoints/                   # Saved model weights
logs/                          # TensorBoard logs
training_viz.html              # Real-time training monitor
```

## 🔬 Methodology

### State Representation
The Feynman diagram is represented as a **heterogeneous Directed Acyclic Graph (DAG)**:
- **Vertices**: Interaction points (initial/final/interaction)
- **Edges**: Particle propagators with quantum numbers (charge, lepton, baryon, color, spin)

Encoded as PyTorch Geometric `Data` objects for neural network processing.

### Action Space
Hierarchical discrete actions:
- `Connect(u, v)`: Connect two vertices with a propagator
- `Branch(u)`: Create a new vertex from an open line (e.g., e⁻ → e⁻ + γ)
- `SetType(e, p)`: Assign particle identity to an edge
- `Terminate()`: End diagram construction

### Physics-Gated Policy Head ⭐

The **critical innovation**: A differentiable physics gate that masks invalid actions.

For each candidate action $a$, compute the **conservation mismatch vector**:

$$\Delta(a) = \begin{bmatrix}
|Q_{\text{in}} - Q_{\text{out}}| \\
|L_{\text{in}} - L_{\text{out}}| \\
|B_{\text{in}} - B_{\text{out}}| \\
\text{ColorMismatch}(a)
\end{bmatrix}$$

Then apply the **Physics Gate**:

$$\Gamma(a) = \exp\left( -\lambda \sum_{k} w_k \cdot (\Delta_k)^2 \right)$$

The final policy is:

$$\pi(a|G_t) = \frac{\pi_\theta(a|G_t) \cdot \Gamma(a)}{\sum_{a'} \pi_\theta(a'|G_t) \cdot \Gamma(a')}$$

This ensures the agent **cannot** select actions that blatantly violate conservation laws, dramatically accelerating learning.

### Reward Function

**Step Reward** (Local conservation):
$$r_{\text{step}} = -\alpha \sum_{v} (|\Delta Q_v| + |\Delta L_v| + |\Delta B_v| + \delta_{\text{color}})$$

**Terminal Reward** (Global topology):
$$r_{\text{final}} = r_{\text{target}} + r_{\text{topo}} + r_{\text{complexity}}$$

- $r_{\text{target}}$: +10 if external lines match $I \to F$, else -10
- $r_{\text{topo}}$: +5 if fully connected with no dangling lines
- $r_{\text{complexity}}$: $-\beta |V_T|$ (penalize higher-order diagrams)

## 🛠️ Installation

### 1. Install Python Dependencies

```powershell
cd rl_training
pip install -r requirements.txt
```

**Note**: PyTorch Geometric requires PyTorch to be installed first. If you encounter issues:

```powershell
# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyG
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 2. Verify Installation

```powershell
python -c "import torch; import torch_geometric; print('✅ All dependencies installed')"
```

## 🚂 Training

### Quick Start

Run the default training script (e⁻ + e⁺ → μ⁻ + μ⁺):

```powershell
cd rl_training
python training.py
```

This will:
- Train the agent for 100,000 timesteps
- Save checkpoints to `checkpoints/`
- Export TensorBoard logs to `logs/`
- Save the best diagram to `diagrams/current_best.json`

### Monitor Training in Real-Time

Open `training_viz.html` in your browser:

```powershell
# Open the visualization
start ../training_viz.html
```

The page will auto-refresh every 2 seconds and display:
- Current episode number
- Latest reward
- Diagram complexity (number of propagators)
- **Live rendering of the best diagram found so far**

### Custom Reactions

Edit `training.py` to define your own reaction:

```python
# Example: Electron-positron annihilation to photon to muon pair
initial_state = ['e', 'e']  # e⁻, e⁺ (antiparticle flag set automatically)
final_state = ['mu', 'mu']  # μ⁻, μ⁺

env = FeynmanDiagramEnv(
    initial_state=initial_state,
    final_state=final_state,
    max_vertices=10,
    max_steps=50
)
```

### Hyperparameter Tuning

Adjust PPO hyperparameters in `training.py`:

```python
trainer = PPOTrainer(
    env=env,
    model=model,
    learning_rate=3e-4,      # Learning rate
    gamma=0.99,              # Discount factor
    gae_lambda=0.95,         # GAE lambda
    clip_epsilon=0.2,        # PPO clip range
    value_coef=0.5,          # Value loss coefficient
    entropy_coef=0.01        # Entropy bonus
)
```

Physics gate penalty (controls how strictly conservation is enforced):

```python
model = FeynmanGCPN(
    lambda_penalty=5.0  # Higher = stricter enforcement
)
```

## 📊 TensorBoard Monitoring

Launch TensorBoard to visualize training metrics:

```powershell
tensorboard --logdir=logs
```

Then open `http://localhost:6006` to see:
- Mean episode reward
- Policy loss
- Value loss
- Entropy (exploration measure)

## 🎨 Visualization with Feynman Forge

The generated diagrams are automatically exported in a format compatible with your JavaScript frontend.

### Option 1: Manual Import

1. After training, the best diagram is saved to `diagrams/current_best.json`
2. Open `feymann.html` in your browser
3. Click **"Canvas Manager"** → **"Import"**
4. Select `diagrams/current_best.json`

### Option 2: Live Monitoring

Use `training_viz.html` (already set up) to watch diagrams update in real-time during training.

### JSON Format

The exported diagrams follow this structure (compatible with `canvas-manager.js`):

```json
{
  "timestamp": "2025-11-21T12:00:00",
  "metadata": {
    "episode": 150,
    "reward": 12.5,
    "initial_state": ["e", "e"],
    "final_state": ["mu", "mu"]
  },
  "shapes": [
    {
      "id": 0,
      "type": "fermion",
      "p1": {"x": 80, "y": 200},
      "p2": {"x": 400, "y": 300},
      "props": {
        "particleId": "e",
        "isAnti": false,
        "color": null,
        "category": "fermion",
        "group": "lepton"
      }
    }
  ]
}
```

## 🧪 Evaluation

Generate and evaluate multiple diagrams from a trained model:

```python
from visualization_bridge import DiagramEvaluator
from models import FeynmanGCPN
from feynman_env import FeynmanDiagramEnv

# Load trained model
model = FeynmanGCPN(...)
checkpoint = torch.load('checkpoints/model_final.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Create evaluator
env = FeynmanDiagramEnv(initial_state=['e', 'e'], final_state=['mu', 'mu'])
evaluator = DiagramEvaluator(model, env)

# Generate 10 diagrams
stats = evaluator.evaluate_multiple(num_episodes=10, output_dir='evaluation')

# Results saved to evaluation/
# - diagram_000.json, diagram_001.json, ...
# - best_diagram.json
# - statistics.json
```

## 📖 Architecture Details

### 1. MPNN Encoder

A standard Message Passing Neural Network that aggregates particle information:

```python
for layer in range(num_layers):
    # Message: neighbor node + edge features → message
    message = MLP(neighbor_features, edge_features)
    
    # Aggregate: sum messages from all neighbors
    aggregated = sum(messages)
    
    # Update: GRU(old_state, aggregated_message)
    new_state = GRU(node_state, aggregated)
```

### 2. Physics-Gated Policy Head

```python
# Raw neural network output
logits = MLP(graph_embedding)

# Compute physics gate for each action
for action in action_space:
    mismatch = compute_conservation_mismatch(action)
    gate_value = exp(-lambda * sum(w_k * mismatch_k^2))
    
# Mask logits
masked_logits = logits + log(gate_value)

# Softmax
policy = softmax(masked_logits)
```

### 3. Value Head

Standard MLP with global mean pooling:

```python
graph_embedding = mean(node_embeddings)
value = MLP(graph_embedding)
```

## 🔧 Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'torch_geometric'`:

```powershell
pip install torch-geometric
```

### CUDA Out of Memory

Reduce batch size or hidden dimensions:

```python
model = FeynmanGCPN(hidden_dim=64)  # Default is 128
trainer = PPOTrainer(batch_size=32)  # Default is 64
```

### Slow Training

Enable GPU acceleration (if available):

```python
trainer = PPOTrainer(device='cuda')
```

Check GPU usage:

```powershell
nvidia-smi
```

## 📚 Physics Database

The `physics_engine.py` mirrors your JavaScript constants exactly:

- **Leptons**: e⁻, μ⁻, τ⁻, νₑ, ν_μ, ν_τ
- **Quarks**: u, d, s, c, b, t (with color charges)
- **Bosons**: γ, g, W⁺, W⁻, Z⁰, H
- **Conservation Laws**: Q, L, B, Color, Spin, Parity
- **Interaction Rules**: Photon-charge coupling, Gluon-quark coupling, Higgs-mass coupling, CKM matrix

All quantum numbers match the Standard Model exactly.

## 🎯 Example Reactions

**1. Electron-positron annihilation (QED)**
```python
initial_state = ['e', 'e']  # e⁻, e⁺
final_state = ['mu', 'mu']  # μ⁻, μ⁺ via γ
```

**2. Beta decay (Weak interaction)**
```python
initial_state = ['d']  # down quark
final_state = ['u', 'e', 'nu_e']  # u + e⁻ + ν̄ₑ via W⁻
```

**3. Quark scattering (QCD)**
```python
initial_state = ['u', 'u']  # up quarks
final_state = ['u', 'u']  # via gluon exchange
```

## 🤝 Integration with Feynman Forge

The RL system is **fully compatible** with your existing frontend:

1. **Same particle database** (`PHYSICS` object)
2. **Same validation logic** (`ConservationLaws` matches `validateDiagram()`)
3. **Same JSON format** for diagram export
4. **Same canvas rendering** (coordinates, colors, styles)

You can seamlessly switch between:
- **Manual drawing** (your current UI)
- **AI generation** (Gemini API)
- **RL generation** (this system)

## 📄 Citation

If you use this code, please cite:

```
Feynman-GCPN: Physics-Informed Reinforcement Learning for Automated Feynman Diagram Generation
Wang, S. (2025)
```

## 📜 License

This project is part of the Feynman Forge toolset. See the main repository for license details.

## 🙏 Acknowledgments

- **Standard Model data** from Particle Data Group (PDG)
- **PyTorch Geometric** for graph neural network primitives
- **Stable-Baselines3** for PPO inspiration (reimplemented here)

---

**Happy Training! 🚀**

For questions or issues, please open an issue in the repository.
