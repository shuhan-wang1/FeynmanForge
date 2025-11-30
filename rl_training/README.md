# Mathematical Formulation & Physics Discovery Experiments:

This README file provides a rigorous explanation of how the Feynman-GCPN encodes Feynmann diagram to mathematical structure and how to intergate Reinforcement learning framework to enable the Neural Network to "discover" physics laws.

## 1. Mathematical Problem Encoding
We formulate the construction (drawing) of a Feynmann diagram as a Markov Decision Process (MDP) defined by the tuple $M = \langle S, A, P, R\rangle$.

### 1.1 State Space $S$
The state $s_t$ is a Heterogenous Directed Graph (DAG), $G_t = (V_t, E_t)$, where:
- $V_t$: is set of vertices. Types: $\{ \text{Initial, Final, Interaction}\}$
- $E_t$: is set of edges (particles). Particle property: $\{ \text{Particle ID, Charge, Lepton Num, Baryon Num,...}\}$

The graph is encoded using a Gated-Message Passing Neural Network (G-MPNN). For a node $v in V$, its embedding $h_v$ is updated iteratively:

$$h_v^{(k)} = \text{GRU}\left( h_v^{(k-1)}, \sum_{u \in \mathcal{N}(v)} \text{MLP}(h_u^{(k-1)}, e_{uv}) \right)$$

### 1.2 Action Space $A$
The action space is hierarchical and discrete. An action $a_t$ is a tuple:

$$a_t = \langle \text{Type}, \text{Vertex}_{idx}, \text{Particle}_{ID}, \text{Target}_{idx} \rangle$$

where Type $\in \{Connect, Branch, SetType, Terminate\}$.

### 1.3 Reward $R$
In this experiment, the reward is split into two components:

$$R(s, a) = R_{known}(s, a) + R_{unknown}(s_{terminal})$$

1. Known Laws ($R_\text{known}$): Dense rewards given at every step.
  - Encourages Charge and Lepton Number conservation (suppose charge and lepton conservation are given to the model).

2. Unknown laws ($R_\text{unknown}$): Sparse reward given only if all ground truth physics laws are observed (represent this particle reaction is fully observed in the Large Hadron Colliders)
  - Specifically targets Baryon Number and Color Charge
  - The Challenge: The agent receives no feedback regarding $B$ during generation. It only gets a large +50 reward if the final diagram is fully valid (i.e, conserves all laws and topologically valid)

## 2. Model Architecture: The Physics Discovery Engine

### 2.1 Split Particle Embedding (PQNE)
We represent every possible unknow conservation laws of particle as a vector $E(p) \in \R^d$. This vector is splitted by

$$E(p) = [E_{fixed}(p) \oplus E_{learnable}(p)]$$

- $E_{fixed}$ (Frozen): Explicitly encodes known quantum numbers ($Q, L$). Gradients are detached.
- $E_{learnable}$ (Trainable): Randomly initialized. The network must adjust these values via backpropagation to minimize the violation of the hidden law ($B$).

### 2.2 Split Conservation Mask (CLDM)
We define a mask vector $\alpha \in [0, 1]^d$ that represents the system's "confidence" that a specific dimension must be conserved.

$$\alpha = [\alpha_{fixed} \oplus \alpha_{learnable}]$$

- $\alpha_{fixed} \approx 1.0$ (We enforce known laws).
- $\alpha_{learnable} = \sigma(\theta_{logits})$ (The model learns which latent dimensions to conserve).

### 2.3 The Meta Physics Gate or Meta Physics Prior ($\gamma
This is the mechanism that enforces physical rules on the policy. For a potential action $a$ (e.g., adding a particle), we compute the conservation violation vector $\Delta(a)$ in the embedding space:

$$\Delta_k(a) = \left| \sum_{in} E_k - \sum_{out} E_k \right|$$

The Gate Value $\Gamma(a)$ is computed as a soft constraint:$$\Gamma(a) = \exp\left( - \lambda \sum_{k} \alpha_k (\Delta_k(a))^2 \right)$$

The final policy distribution $\pi'(a|s)$ is the raw policy modulated by physics:

$$\pi'(a|s) \propto \pi_\theta(a|s) \cdot \Gamma(a)$$

## How does model learns the physics
The learning process is a form of inverse constraint satisfaction:
1. Signal: The agent hits a valid final state (conserving Baryon number) by chance and gets a large reward ($+50$).
2. Backpropagation: To maximize this reward, the policy $\pi$ wants to select actions that lead to valid diagrams.
3. Gate Optimization: The easiest way to maximize $\pi'(a|s)$ for valid actions is to maximize $\Gamma(a)$.
    - $\Gamma(a)$ is maximized when $\sum \alpha_k (\Delta_k)^2 \to 0$.
4. Discovery:
  - For the hidden law (Baryon number), $\Delta_{fixed}$ is zero (irrelevant).
  - Therefore, the model must adjust $E_{learnable}$ such that they act like Baryon numbers (additively conserved) and increase $\alpha_{learnable}$ to enforce this conservation.

Verification of learning: We know the model has "learned" physics if, after training:
1. High Correlation: The values in a specific dimension of $E_{learnable}$ correlate linearly with the true Baryon numbers of the particles.
2. Sparsity: The corresponding $\alpha_{learnable}$ for that dimension is close to $1.0$, while irrelevant dimensions are near $0$.
