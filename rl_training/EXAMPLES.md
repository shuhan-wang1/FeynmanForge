# Feynman-GCPN Usage Examples

## Quick Start

### 1. Install Dependencies

```powershell
cd rl_training
pip install -r requirements.txt
```

### 2. Train on Default Reaction (e⁻ + e⁺ → μ⁻ + μ⁺)

```powershell
python train.py
```

### 3. Monitor Training

Open `training_viz.html` in your browser to watch the agent learn in real-time.

### 4. View TensorBoard

```powershell
tensorboard --logdir=logs
```

Open `http://localhost:6006` to see training curves.

---

## Custom Reactions

### Example 1: Beta Decay (d → u + e⁻ + ν̄ₑ)

```powershell
python train.py --reaction "d->u+e+nu_e" --timesteps 50000
```

### Example 2: Compton Scattering (γ + e⁻ → γ + e⁻)

```powershell
python train.py --reaction "photon+e->photon+e" --timesteps 100000
```

### Example 3: Quark Scattering (u + d → u + d via gluon)

```powershell
python train.py --reaction "u+d->u+d" --timesteps 200000
```

---

## Advanced Configuration

### Custom Hyperparameters

```powershell
python train.py \
    --reaction "e+e->mu+mu" \
    --timesteps 500000 \
    --hidden-dim 256 \
    --lr 1e-4 \
    --lambda-penalty 10.0 \
    --device cuda
```

### Physics Gate Tuning

Higher `lambda-penalty` = stricter conservation enforcement:

```powershell
# Loose enforcement (more exploration)
python train.py --lambda-penalty 1.0

# Strict enforcement (faster convergence, less exploration)
python train.py --lambda-penalty 20.0
```

---

## Programmatic Usage

### Python Script Example

```python
from rl_training import (
    FeynmanDiagramEnv,
    FeynmanGCPN,
    PPOTrainer,
    PhysicsConstants
)

# Define reaction
initial_state = ['e', 'e']  # e⁻, e⁺
final_state = ['mu', 'mu']  # μ⁻, μ⁺

# Create environment
env = FeynmanDiagramEnv(
    initial_state=initial_state,
    final_state=final_state,
    max_vertices=10,
    max_steps=50
)

# Create model
num_particle_types = (
    len(PhysicsConstants.get_all_particles()) + 
    len(PhysicsConstants.BOSONS)
)

model = FeynmanGCPN(
    hidden_dim=128,
    num_particle_types=num_particle_types,
    lambda_penalty=5.0
)

# Create trainer
trainer = PPOTrainer(
    env=env,
    model=model,
    learning_rate=3e-4,
    device='cuda'
)

# Train
trainer.train(
    total_timesteps=100000,
    checkpoint_dir='my_checkpoints',
    log_dir='my_logs'
)
```

---

## Evaluation

### Evaluate Trained Model

```powershell
python evaluate.py \
    --checkpoint checkpoints/model_final.pt \
    --reaction "e+e->mu+mu" \
    --num-episodes 20 \
    --output-dir results
```

### Generate Best Diagram

```python
from rl_training import DiagramEvaluator, FeynmanGCPN, FeynmanDiagramEnv
import torch

# Load model
model = FeynmanGCPN()
checkpoint = torch.load('checkpoints/model_final.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Create environment
env = FeynmanDiagramEnv(['e', 'e'], ['mu', 'mu'])

# Generate diagram
evaluator = DiagramEvaluator(model, env)
shapes, reward = evaluator.generate_diagram(deterministic=True)

# Save
from rl_training import DiagramExporter
DiagramExporter.save_diagram(shapes, 'my_diagram.json')
```

---

## Visualization Integration

### Option 1: Live Training Monitor

1. Start training:
   ```powershell
   python train.py --reaction "e+e->mu+mu"
   ```

2. Open `training_viz.html` in browser

3. Watch diagrams update every 2 seconds

### Option 2: Import into Feynman Forge

1. After training, find `diagrams/current_best.json`

2. Open `feymann.html`

3. Click **Canvas Manager** → **Import**

4. Select `diagrams/current_best.json`

5. View and edit the diagram

### Option 3: Batch Export

```python
from rl_training import DiagramExporter

# Export training sequence
DiagramExporter.export_training_sequence(
    diagrams=[diagram1, diagram2, ...],
    rewards=[r1, r2, ...],
    filepath='animation.json'
)
```

---

## Available Particles

### Leptons
- `e` (electron e⁻)
- `mu` (muon μ⁻)
- `tau` (tau τ⁻)
- `nu_e` (electron neutrino νₑ)
- `nu_mu` (muon neutrino ν_μ)
- `nu_tau` (tau neutrino ν_τ)

### Quarks
- `u` (up)
- `d` (down)
- `c` (charm)
- `s` (strange)
- `t` (top)
- `b` (bottom)

### Bosons
- `photon` (γ)
- `gluon` (g)
- `w_plus` (W⁺)
- `w_minus` (W⁻)
- `z` (Z⁰)
- `higgs` (H)

---

## Common Reactions

### QED (Electromagnetic)

```powershell
# Electron-positron annihilation
python train.py --reaction "e+e->mu+mu"

# Compton scattering
python train.py --reaction "photon+e->photon+e"

# Pair production
python train.py --reaction "photon+photon->e+e"
```

### Weak Interactions

```powershell
# Beta decay (simplified)
python train.py --reaction "d->u+e+nu_e"

# Muon decay
python train.py --reaction "mu->e+nu_mu+nu_e"

# W decay
python train.py --reaction "w_plus->mu+nu_mu"
```

### QCD (Strong)

```powershell
# Quark-antiquark annihilation
python train.py --reaction "u+u->gluon+gluon"

# Gluon splitting
python train.py --reaction "gluon->u+u"
```

---

## Monitoring & Debugging

### Check Training Progress

```powershell
# Watch console output
python train.py --reaction "e+e->mu+mu"

# View TensorBoard
tensorboard --logdir=logs
```

### Common Issues

**1. No diagrams generated**
- Check `diagrams/` folder exists
- Ensure training ran for at least 10 episodes
- Verify reaction is valid

**2. Low rewards**
- Increase training time: `--timesteps 500000`
- Adjust physics gate: `--lambda-penalty 3.0`
- Check TensorBoard for policy/value loss trends

**3. Memory errors**
- Reduce batch size in `training.py`
- Lower `--hidden-dim 64`
- Use CPU: `--device cpu`

---

## Performance Tips

### GPU Acceleration

```powershell
# Verify CUDA available
python -c "import torch; print(torch.cuda.is_available())"

# Train with GPU
python train.py --device cuda
```

### Faster Training

1. **Increase rollout steps** (more data per update):
   ```python
   trainer.train(rollout_steps=4096)  # Default: 2048
   ```

2. **Parallel environments** (future work):
   ```python
   # Not yet implemented, but straightforward to add
   envs = [FeynmanDiagramEnv(...) for _ in range(4)]
   ```

3. **Mixed precision training**:
   ```python
   # Use torch.cuda.amp for faster training
   ```

---

## Output Files

### During Training

```
diagrams/
├── current_best.json       # Best diagram found so far
├── current_diagram.json    # Latest diagram
├── training_sequence.json  # History of diagrams
└── reaction_config.json    # Reaction metadata

checkpoints/
├── model_step_10000.pt
├── model_step_20000.pt
└── model_final.pt

logs/
└── feynman_gcpn_20251121_120000/
    └── events.out.tfevents.*
```

### After Evaluation

```
evaluation/
├── best_diagram.json
├── statistics.json
├── diagram_000.json
├── diagram_001.json
└── ...
```

---

## Next Steps

1. **Experiment with different reactions** to see what diagrams the agent learns

2. **Tune hyperparameters** to improve convergence speed

3. **Extend the action space** to include more complex topological operations

4. **Add curriculum learning** to train on progressively harder reactions

5. **Implement parallel environments** for faster data collection

6. **Export to LaTeX** for publication-ready diagrams

---

For more details, see the full README in `rl_training/README.md`.
