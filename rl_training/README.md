# Feynman-GCPN: Reinforcement Learning for Feynman Diagram Generation

A complete implementation of **Feynman-GCPN** (Graph Convolutional Policy Network), a novel Reinforcement Learning framework that learns to construct valid Feynman diagrams by navigating the topological space of particle interactions while respecting Standard Model conservation laws.

## 🚀 Overview

This project implements a **physics-informed RL agent** that:

1.  **Constructs Feynman diagrams step-by-step** as sequential graph operations
2.  **Enforces conservation laws** (charge, lepton number, baryon number, color charge) through a differentiable **Physics Gate**
3.  **Learns valid particle interactions** purely from environment rewards, without hard-coded rules
4.  **Discovers hidden conservation laws** (e.g., Baryon number) using a semi-supervised approach (V8 architecture)

## 📁 Project Structure

```
rl_training/
├── run_experiment.py          # Main entry point for training and evaluation
├── config.py                  # Configuration and hyperparameters
├── models_v8.py               # Neural networks (Split Embedding, Physics Gate)
├── env_v8.py                  # Environment wrapper with hybrid rewards
├── trainer_v8.py              # PPO training loop
├── evaluator_v8.py            # Evaluation and analysis tools
├── particle_utils.py          # Particle database & utilities
├── physics_engine.py          # Core physics logic & conservation laws
├── feynman_env.py             # Base Gymnasium environment
└── requirements.txt           # Python dependencies
```

## 🔬 Methodology: Conservation Law Discovery (V8)

This project implements a semi-supervised Reinforcement Learning framework designed to **rediscover** physics laws (specifically Baryon Number conservation) while learning to construct valid Feynman diagrams.

### 1. Problem Formulation (MDP)

The Feynman diagram construction is modeled as a Markov Decision Process:

-   **State ($S_t$)**: A heterogeneous Directed Acyclic Graph (DAG) where nodes are interaction vertices and edges are particle propagators.
    -   *Representation*: PyTorch Geometric `Data` object + 12-dimensional summary vector (vertex counts, connectivity status, open lines).
-   **Action Space ($A$)**: Hierarchical discrete actions:
    1.  **Action Type**: `CONNECT`, `BRANCH` (1$\to$2), `MERGE` (2$\to$1), `SET_TYPE`, `TERMINATE`.
    2.  **Vertex Selection**: Pointer Network selects which vertex to modify.
    3.  **Particle Selection**: Chooses particle identity (e.g., $e^-, \gamma, u, d$).
-   **Reward ($R$)**: Hybrid "Scientist Reward" (see below).

### 2. Neural Architecture (V8)

The core innovation is the **Split-Embedding Physics-Gated Network**:

#### A. Split Particle Embedding (PQNE)
We split the particle embedding $E(p)$ into two components:
$$ E(p) = [ E_{\text{fixed}}(p) \oplus E_{\text{learnable}}(p) ] $$
-   **Fixed Part**: Encodes **known** quantum numbers (Charge $Q$, Lepton Number $L$). These are frozen.
-   **Learnable Part**: Randomly initialized. The model must learn to encode **unknown** properties (like Baryon Number $B$) in these dimensions to satisfy conservation constraints.

#### B. Split Conservation Mask (CLDM)
We learn a conservation confidence mask $\alpha \in [0, 1]^D$:
$$ \alpha = [ \alpha_{\text{fixed}} \oplus \alpha_{\text{learnable}} ] $$
-   $\alpha_{\text{fixed}} \approx 1.0$: We tell the model "Charge and Lepton number MUST be conserved".
-   $\alpha_{\text{learnable}}$: The model learns which of its new dimensions *should* be conserved.

#### C. Meta-Physics Gate
A differentiable gate modulates the policy output $\pi(a)$ based on conservation laws:
$$ \Gamma(a) = \exp\left( -\lambda \sum_{k} \alpha_k \cdot (\Delta_k(a))^2 \right) $$
where $\Delta_k(a)$ is the conservation mismatch (e.g., $\sum Q_{in} - \sum Q_{out}$) for action $a$.
The final policy is:
$$ \pi'(a|s) \propto \pi_\theta(a|s) \cdot \Gamma(a) $$

### 3. Training Strategy: "The Scientist Reward"

We use a semi-supervised reward structure to simulate scientific discovery:

| Regime | Physics Laws | Feedback Type | Reward Function |
|--------|-------------|---------------|-----------------|
| **Known** | Charge ($Q$), Lepton ($L$) | **Immediate** | $+2.0$ per valid vertex, $-2.0$ per violation |
| **Unknown** | Baryon ($B$) | **Sparse** | $0.0$ immediate. Only $+50.0$ if *entire* diagram is valid at the end. |

**Discovery Mechanism**:
1.  The **Physics Gate** forces the model to respect $Q$ and $L$ immediately.
2.  To get the global $+50.0$ reward, the model *must* also satisfy Baryon conservation (which is checked at the end).
3.  Since it receives no vertex-level feedback for $B$, it must **self-organize** its $E_{\text{learnable}}$ space to align with $B$ conservation, effectively "discovering" the law.

## ⚙️ Experimental Setup

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Total Steps** | 500,000 | Total environment interactions |
| **Batch Size** | 32 | PPO batch size |
| **Learning Rate** | 3e-4 | Adam optimizer learning rate |
| **Fixed Dim** | 2 | Dimensions for Q, L (Known) |
| **Learnable Dim** | 6 | Dimensions for discovery (Unknown) |
| **Physics Penalty ($\lambda$)** | 5.0 | Strength of physics gate enforcement |
| **Sparsity Weight** | 0.001 | Regularization to keep $\alpha_{\text{learnable}}$ sparse |

### Datasets

The model is trained on **Decay** reactions and tested on **Scattering** reactions to evaluate generalization to new topologies.

**Training Set (Decays 1$\to$N)**:
-   **Leptonic**: $\mu \to e \nu \bar{\nu}$, $\tau \to \mu \nu \bar{\nu}$, $Z \to \ell^+\ell^-$
-   **Hadronic (Critical for B discovery)**: $Z \to u\bar{u}$, $Z \to d\bar{d}$, $Z \to c\bar{c}$

**Testing Set (Scattering 2$\to$N)**:
-   **Annihilation**: $e^+e^- \to \mu^+\mu^-$
-   **Scattering**: $e^-e^- \to e^-e^-$
-   **Compton**: $e^-\gamma \to e^-\gamma$

## 🚂 Usage

### Basic Training
Run the main experiment script:
```powershell
python run_experiment.py
```
This will:
- Train the agent on decay reactions (1→N).
- Periodically evaluate on scattering reactions (2→N).
- Save results to `results/`.

### Quick Test
Run a short training session to verify everything works:
```powershell
python run_experiment.py --quick
```

### Custom Parameters
```powershell
python run_experiment.py --steps 100000 --learnable-dim 8 --output results/my_experiment
```

## 📊 Output

Results are saved in the `results/` directory (or specified output dir):
- `best.pt`: Best model checkpoint.
- `final.pt`: Final model checkpoint.
- `results.json`: Training metrics and analysis of discovered conservation laws.

## 🤝 Integration with Feynman Forge

The generated diagrams and models are compatible with the Feynman Forge frontend. The `physics_engine.py` mirrors the JavaScript physics logic, ensuring consistency.

## 📄 Citation

If you use this code, please cite:
```
Feynman-GCPN: Physics-Informed Reinforcement Learning for Automated Feynman Diagram Generation
Wang, S. (2025)
```
