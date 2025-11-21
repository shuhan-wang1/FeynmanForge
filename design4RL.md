# Methodology for RL Feynman diagram drawing
We can formulate the problem of generating a Feynman diagram for a specific particle reaction $I \to F$ (initial $\to$ final state) as a Markov Decision Process (MDP). The agent constructs the diagram step-by-step, starting from a set of disconnected external lines and learning the particle physics interaction purely through a physics-informed validation environment.

## 1. MDP formulation
The problem is defined by the set $M = (S,A,P,R,\ell), where:
1. $S$ is the state space representing all possible partial Feynman diagrams (represented as Directed Acyclic Graphs)
2. $A$, the discrete action space of topological operations
3. $P$: the deterministic state transition dynamics $s_{t+1} = f(s_t, a_t)$
4. $R$: the reward function encoding conservation laws and topological constraints (such as Higgs boson only coupling to massive particles, EM force carrier couples with charged particles, any force carrier particles are Bosons, ...)
5. $\ell$: the discount factor

## 2. State Representation ($S$)
At time step $t$, the state $G_t \in S$ is represented as a heterogenous graph consisting of vertices (interaction vertices) and edges (propagated particles):

$$G_t = (V_t, E_t, X_t)$$

- $V_t$: set of vertices at time $t$
- $E_t$: set of edges at time $t$. An edge $e_{ij} \in E_t$ connects vertex $v_i$ to vertex $v_j$
- $X_t$: feature matrix encoding particle types, quantum numbers, ...,. We associate a feature vector $x_{uv}$ with each edge $(u,v) \in E_t$:

$$x_{uv} = \left[ \text{Type, Spin, Charge, Q, Lepton number, Baryon Number, Colour Charge, IsAnti}\right]$$

Additionally, define a global nodes feature vector to check the globally conserved quantities $C_{global} = [Q_net, L_net, B_net]_{target}$.

## 3. Action Space ($A$)
The action space is discrete and hierarchical. At each step $t$, the policy $\pi_\theta(a_t|G_t)$ samples an action $a_t$ from the following set:

1. `Connect(u, v)`: Add a propagator edge between two existing open half-lines $u$ and $v$ in the diagram.
2. `Branch(u)`: Create a new interaction vertex at the end of open line $u$, spawning one or two new open lines (eg $e^- \to e^- + \gamma$).
3. `SetType(e, p)`: Assign particle identity $p$ to a newly created edge $e$.
4. `Terminate()`: End the diagram construction process.

## 4, Physics-informed Reward Function ($R$)
The reward is composed of a dense step-wise reward and a sparse terminal reward:

$$R(s_t, a_t) = r_\text{step}(s_t, a_t) + \mathcal{I}(a_t = \text{Terminate}) \cdot r_\text{final}(G_T)$$

### 4.1 Step Reward: Local Conservation Checks
This reward is computed immediately after an action creates or modifies a vertex $v$. We verify standard model local symmetries:

Kirchhoff's Law for Quantum Numbers and Colour charge:
$$\sum_{e \in \text{in}(v)} Q(e) = \sum_{e \in \text{out}(v)} Q(e) \quad (\text{Charge})$$

$$\sum_{e \in \text{in}(v)} L(e) = \sum_{e \in \text{out}(v)} L(e) \quad (\text{Lepton Number})$$

$$\sum_{e \in \text{in}(v)} B(e) = \sum_{e \in \text{out}(v)} B(e) \quad (\text{Baryon Number})$$

$$\sum_{e \in \text{in}(v)} C(e) - \sum_{e \in \text{out}(v)} C(e) = 0 \quad (\text{Color Neutrality})$$

Also the coupling Rules:

- Higgs boson only couples to massive particles
- Electromagnetic force carrier (photon) couples with charged particles 
- Any force carrier particles are Bosons
- Gluons couple only to particles with color charge (quarks and other gluons)
- Weak force conversion rule, destory CP violation, but follows the global conservation of lepton and baryon numbers
- A $Z$ boson couples to any standard model fermion (except the hypothetical right-handed neutrino).
- the gauge bosons interact with each other.Triple Gauge Couplings (3 lines):$W^+ W^- \gamma$ (W bosons are charged, so they talk to photons).$W^+ W^- Z$ (W bosons have weak isospin, so they talk to Z).Quartic Gauge Couplings (4 lines):$W^+ W^- \gamma \gamma$$W^+ W^- Z Z$$W^+ W^- Z \gamma$$W^+ W^- W^+ W^-$Forbidden:Any vertex with only neutral bosons (e.g., $Z Z Z$, $\gamma \gamma \gamma$, $Z \gamma \gamma$). Neutral bosons do not couple to themselves.
## 4.2. Final Reward: Global Topology & Target
Upon termination, the final graph $G_T$ is evaluated against the user's requested reaction $I \to F$:

$$r_{\text{final}} = r_{\text{target}} + r_{\text{topo}} + r_{\text{complexity}}$$

1. Target Match ($r_{\text{target}}$): Ensures the external lines match the user input.

$$r_{\text{target}} = \begin{cases} +10 & \text{if } E_{\text{ext}}(G_T) \equiv I \cup F \\ -10 & \text{otherwise} \end{cases}$$

2. Topological Validity ($r_{\text{topo}}$): Ensures the graph is connected and has no dangling internal lines.

$$r_{\text{topo}} = \begin{cases} +5 & \text{if } G_T \text{ is connected } \land \text{NoDanglingLines}(G_T) \\ -5 & \text{otherwise} \end{cases}$$

Parsimony ($r_{\text{complexity}}$): Penalizes higher-order diagrams to encourage finding the Leading Order (LO) diagram first.

$$r_{\text{complexity}} = - \alpha \cdot |V_T|$$

We utilize a Graph Convolutional Policy Network (GCPN) parameterized by $\theta$.

1. Encoder: A Graph Neural Network (GNN) embeds the state $G_t$ into node embeddings $h_v$.

2. Policy Head: Maps embeddings to action probabilities:
$$\pi_\theta(a_t | G_t) = \text{Softmax}(\text{MLP}(h_v))$$

3. Objective: We maximize the expected return using Proximal Policy Optimization (PPO):
$$L(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$

Where $\hat{A}_t$ is the Generalized Advantage Estimation derived from our physics-based reward function $\mathcal{R}$.