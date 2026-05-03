## III. The Determinism Trilemma

The integration of approximate nearest neighbor (ANN) search into blockchain architectures necessitates a fundamental shift from probabilistic hardware execution to strict deterministic state machines. In this section, we formalize the requirements for consensus-safe vector indexing and propose the *Determinism Trilemma*—a framework addressing the root causes of state divergence in distributed HNSW graph construction.

### A. Problem Formalization

Let $G = (V, E)$ denote an HNSW graph where $V$ is the set of vertices (vector embeddings) and $E$ represents the set of edges (navigable connections across hierarchical layers). We define the deterministic graph construction function as:

$$ \Phi: (X, R) \rightarrow G $$

where $X = \{x_1, x_2, ..., x_n\}$ is an ordered sequence of vector insertions, and $R$ is a source of entropy used for layer assignment. 

**Definition 1 (Consensus-Safe ANN Index):** *An ANN indexing implementation is considered consensus-safe if and only if, for any given state transition block, all honest validator nodes $v_1, v_2 \in \mathcal{V}$ produce a bit-identical graph serialization:*
$$ \Phi_{v_1}(X, R) \equiv \Phi_{v_2}(X, R) $$

**Theorem 1 (Non-Determinism in Standard HNSW):** *Standard HNSW implementations inherently violate consensus safety due to three independent sources of hardware and system-level divergence.*
*Proof Sketch:* (1) **Floating-point variance:** Distance calculations $d(x, y)$ rely on IEEE 754 floating-point operations. Processors utilizing 80-bit extended precision internally (e.g., x86\_64 AVX-512) will produce different least-significant bits compared to processors enforcing strict 64-bit precision (e.g., ARM64 NEON). (2) **Entropy dependencies:** Standard implementations draw entropy from local system states (e.g., `/dev/urandom`), causing divergent layer assignments. (3) **Concurrency variance:** Multi-threaded graph construction yields non-deterministic node insertion orders, fundamentally altering the graph's topological structure. $\blacksquare$

### B. The Trilemma Constraints

To achieve Definition 1, a system must satisfy three foundational constraints. D-HNSW implements these constraints to eradicate hardware-induced divergence while preserving the mathematical integrity of the proximity graph.

**Constraint 1: Cryptographic Seeding (Entropy Determinism)**
The system must derive its layer assignment entropy directly from consensus-agreed artifacts. D-HNSW defines the random seed for a given insertion as:
$$ seed = \mathcal{H}(TxHash \oplus BlockHash) $$
where $\mathcal{H}$ is the Keccak-256 hash function. The XOR operation ($\oplus$) mitigates transaction grinding attacks, as the $BlockHash$ remains unpredictable to the user at the time the transaction ($TxHash$) is broadcast.

**Constraint 2: Fixed-Point Arithmetic (Hardware Determinism)**
All algebraic operations, particularly distance metric evaluations (e.g., Euclidean distance, Cosine similarity), must bypass floating-point ALUs. D-HNSW employs $I64F32$ representation (32 signed integer bits, $f=32$ fractional bits).

Transitioning from continuous $\mathbb{R}$ to a discrete fixed-point space inherently introduces quantization errors. To mathematically guarantee that this transition does not trigger catastrophic "routing divergence" (where the search path deviates entirely from the optimal trajectory), we establish the following distortion bound:

**Theorem 2 (Fixed-Point Distortion Bound):** *Let $x, y \in \mathbb{R}^D$ be two real-valued embedding vectors, and $\hat{x}, \hat{y}$ be their $I64F32$ fixed-point representations. The error in the squared Euclidean distance computation $\mathcal{E} = |d^2(x, y) - \hat{d}^2(\hat{x}, \hat{y})|$ is strictly bounded by:*
$$ \mathcal{E} \le D \cdot 2^{-f+1} \cdot \max_{i} |x_i - y_i| + D \cdot 2^{-2f+2} $$

*Proof:* 
The quantization error for a single dimension is defined as $\epsilon_{xi} = x_i - \hat{x}_i$, where $|\epsilon_{xi}| \le 2^{-f}$ due to truncation/rounding in the $f$-bit fractional representation. The 1D distance in the real space is $\Delta_i = x_i - y_i$, while the fixed-point distance is $\hat{\Delta}_i = \hat{x}_i - \hat{y}_i = \Delta_i - (\epsilon_{xi} - \epsilon_{yi})$. 
Let $\delta_i = \epsilon_{xi} - \epsilon_{yi}$, which implies bounded variance $|\delta_i| \le 2 \cdot 2^{-f} = 2^{-f+1}$. The squared distance divergence for dimension $i$ is:
$$ |\Delta_i^2 - \hat{\Delta}_i^2| = |\Delta_i^2 - (\Delta_i - \delta_i)^2| = |2\Delta_i\delta_i - \delta_i^2| \le 2|\Delta_i||\delta_i| + \delta_i^2 $$
Aggregating over $D$ dimensions, the total error is:
$$ \mathcal{E} \le \sum_{i=1}^D (2|\Delta_i| \cdot 2^{-f+1} + 2^{-2f+2}) \le D \cdot 2^{-f+1} \cdot \max_{i} |\Delta_i| + D \cdot 2^{-2f+2} $$ $\blacksquare$

*Practical Implication:* For normalized $D=768$ language model embeddings (where dimensions typically fall within $[-1, 1]$, making $\max|\Delta_i| \le 2$), utilizing $f=32$ fractional bits restricts the maximum theoretical error $\mathcal{E}$ to approximately $7.15 \times 10^{-7}$. This infinitesimal divergence guarantees that the topological sorting of nearest neighbors remains invariant compared to standard $f32$ implementations, entirely preventing topological forks.

**Constraint 3: Canonical Ordering (Topological Determinism)**
To prevent race conditions during graph construction, vector insertions must be sequenced canonically. D-HNSW serializes all insertions strictly according to their transaction index within the finalized blockchain block. While this temporarily suspends intra-block parallelization, inter-block parallelism remains unaffected.

### C. Consensus Safety Guarantee

**Theorem 3 (D-HNSW Consensus Safety):** *An HNSW implementation satisfying Constraints 1, 2, and 3 is strictly consensus-safe.*
*Proof:* Given an identical global state $(X, seed)$, Constraint 1 enforces an identical deterministic sequence for layer assignment. Constraint 2 guarantees that every pairwise distance comparison evaluates identically across all ALUs, regardless of hardware architecture. Constraint 3 ensures that all topological updates are executed in a deterministic sequence. By mathematical induction over $|X|$ insertions, all intermediate layer graphs and the final serialized graph byte-array are proven to be bit-identical across all nodes. $\blacksquare$

---

<a href="https://www.orchestra-research.com/"><img src="https://img.shields.io/badge/Orchestra-Research-6C3FC5.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PC9zdmc+" alt="Orchestra Research"></a>
