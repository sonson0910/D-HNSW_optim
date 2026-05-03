# Literature Review: Deterministic and Verifiable Approximate Nearest Neighbor Search in Decentralized Systems

## 1. Background

### Context and Motivation
As Artificial Intelligence (AI) expands from isolated, centralized environments into decentralized, agentic, and safety-critical domains, the systems that support AI must evolve. Modern AI agents increasingly rely on Retrieval-Augmented Generation (RAG) and long-term memory substrates to ground their logic. However, when these vector databases and inference engines are deployed on decentralized infrastructures—such as distributed ledgers or third-party cloud providers—a critical trust gap emerges. Users and smart contracts must verify that an outsourced query or AI inference was executed faithfully. This has driven a pressing need for verifiable approximate nearest neighbor (ANN) search and deterministic AI execution, ensuring that identical models and inputs yield bit-exact, provable results regardless of the underlying hardware.

### Core Concepts and Definitions
To navigate this intersection of AI and distributed systems, several advanced concepts are foundational:
- **Hierarchical Navigable Small World (HNSW) Graphs**: The industry standard for ANN search, routing queries through a multi-layered proximity graph. While highly efficient, standard HNSW relies on floating-point distance metrics (e.g., Cosine, L2) which are inherently non-deterministic across different CPU/GPU architectures.
- **Authenticated Data Structures (ADS)**: Cryptographic structures, such as Merkle trees, overlaid on top of databases or indexes (like HNSW) to generate verifiable proofs of inclusion or correct execution without requiring the verifier to hold the entire dataset.
- **Deterministic Execution**: The property where a computation yields the exact same bit-level output for a given input across different platforms. In AI, this is disrupted by IEEE 754 floating-point optimizations (like Fused Multiply-Add) and parallel reduction variations in hardware.
- **Optimistic Verification**: A cryptoeconomic model where outsourced computational results are accepted by default but can be challenged during a dispute window via deterministic re-execution. 

### Research Challenges
Researchers operating at the intersection of vector search and blockchain face the "Determinism Trilemma": balancing high-dimensional query latency, algorithmic recall, and cryptographic verifiability. Specifically, floating-point non-determinism means that an honest node might produce a slightly different vector embedding or distance calculation than another honest node simply because one uses an x86 processor and the other uses ARM. Existing cryptographic proofs (like zkML) are too computationally burdensome for billion-scale HNSW graphs. Consequently, researchers must either enforce strict determinism through fixed-point arithmetic, homogenize the hardware, or develop novel verification protocols that tolerate floating-point variances without opening attack vectors for malicious nodes.

## 2. Research Landscape

The literature addressing verifiable and deterministic vector search reflects a transition from optimizing raw retrieval performance to establishing trust in decentralized environments. Initially, baseline algorithms like [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) established the dominance of in-memory floating-point graph structures. However, as the focus shifted to trustless outsourced execution, researchers began overlaying cryptographic verification onto these structures. 

Currently, the research branches into two distinct paradigms for handling the non-determinism inherent in modern AI. The first paradigm enforces strict bit-level determinism. This is achieved either at the software level by abandoning floating-point math in favor of fixed-point arithmetic for memory substrates, or at the hardware level by pinning GPU architectures and driver versions to guarantee reproducible execution traces. The second paradigm embraces hardware heterogeneity, introducing nondeterminism-aware verification. Instead of demanding bit-exact equality, these systems utilize localized error bounds and interactive dispute protocols to accept slight floating-point divergences while heavily penalizing malicious deviations. Together, these approaches form a nascent but rapidly accelerating subfield aimed at securing decentralized AI infrastructure.

| Research Direction | Coverage | Key Papers | Insights |
|-------------------|----------|-----------|----------|
| Baseline ANN & Graph Structuring | ★★★★★ | [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320), [AQR-HNSW](https://arxiv.org/abs/2602.21600) | Highly mature. HNSW is established as the gold standard for recall/latency, though inherently memory-intensive and non-deterministic across architectures. |
| Verifiable ANN on Blockchain | ★★★ | [ANNProof](https://pure.bit.edu.cn/en/publications/annproof-building-a-verifiable-and-efficient-outsourced-approxima), [OPRCP](https://jwcn-eurasipjournals.springeropen.com/article/10.1186/s13638-018-1221-3) | Moderate exploration. Successfully maps Merkle structures to HNSW, drastically reducing verification overhead, but relies on homogenous execution environments. |
| Deterministic Vector Memory | ★★ | [Valori: A Deterministic Memory Substrate for AI Systems](https://arxiv.org/pdf/2512.22280) | Emerging direction. Replaces IEEE 754 floats with fixed-point arithmetic (Q16.16) to achieve cross-platform bit-exact consensus for RAG pipelines. |
| Nondeterministic-Aware Verification | ★★ | [Nondeterminism-Aware Optimistic Verification for Floating-Point Neural Networks](https://arxiv.org/html/2510.16028v1), [EigenAI](https://arxiv.org/pdf/2602.00182) | Emerging direction. Pioneers the use of operator-level thresholds to solve the cross-hardware float variance problem without sacrificing GPU acceleration. |

### Critical Evaluation

Does this literature solve the research question regarding deterministic ANN search and verifiable AI on distributed ledgers? The answer is partially. If we view the challenge as a puzzle, the literature provides robust individual pieces, but a fully unified, scalable system is still being assembled. 

We now have highly efficient baseline vector search topologies (HNSW) and mathematically sound methods for proving their query results (Merkle-authenticated data structures). Systems like ANNProof successfully demonstrate that outsourced graph traversal can be verified on a blockchain with manageable overhead. However, when these systems hit the reality of decentralized hardware networks, they collide with floating-point non-determinism. The literature addresses this barrier through two diverging, yet incomplete, paths. On one hand, frameworks like Valori provide strict cross-hardware determinism by stripping away floating-point operations in favor of fixed-point (Q16.16) arithmetic. While this solves the replicability issue required for standard consensus, it inherently sacrifices the dynamic range and hardware-accelerated (e.g., NVIDIA Tensor Core) speed of native FP32 operations. 

On the other hand, approaches like Nondeterminism-Aware Optimistic Verification (NAO) and EigenAI allow the use of native floating-point math by introducing complex error-bounding and optimistic challenge periods. While this maintains high performance, it introduces significant cryptoeconomic complexity and latency in finality during dispute resolutions. We are seeing a tug-of-war between modifying the math to fit the consensus protocol, or modifying the consensus protocol to tolerate the math. 

In short, while baseline ANN structures and basic cryptographic verifications are well-understood, achieving scalable, cross-platform deterministic vector search remains an open frontier because the field is still navigating the fundamental trade-off between floating-point hardware optimization and bit-exact reproducibility.

## 3. Detailed Paper Analysis

### [ANNProof: Building a verifiable and efficient outsourced approximate nearest neighbor search system on blockchain](https://pure.bit.edu.cn/en/publications/annproof-building-a-verifiable-and-efficient-outsourced-approxima)
*Lu et al., 2024*

#### **Overview and Key Insights**: 
As data-as-a-service models expand, clients frequently outsource K-approximate nearest neighbor (K-ANN) searches to untrusted third-party cloud or decentralized node providers. This paper addresses the critical risk of a malicious provider returning incorrect, incomplete, or manipulated vector search results. The core takeaway is the successful marriage of HNSW architectures with Authenticated Data Structures (ADS) to create a framework that enables a lightweight client or smart contract to cryptographically verify the integrity of an outsourced high-dimensional vector search. 

#### **Method**:
ANNProof introduces two novel authenticated structures to secure the search. First, the *Merkle HNSW node tree* authenticates the routing and distance computations across the proximity graph layers. Second, the *Merkle vector identifier tree* verifies the final dataset points. When a query is executed, the untrusted server generates a Verification Object (VO)—a cryptographic proof containing the sibling hashes of the traversed path. Because standard Merkle tree updates can be overwhelmingly expensive when dealing with massive vector insertions, the authors introduce a "Merkle sharding tree" optimization. This partitions the ADS, localizing state updates and massively reducing the overhead of maintaining the index on a blockchain smart contract. The smart contract acts as the ultimate arbiter, maintaining state consistency checks.

#### **Evaluation**:
The framework was evaluated against state-of-the-art verifiable query baselines using standard vector datasets. ANNProof demonstrated profound efficiency gains: reducing VO generation time by 160×, verification time by 120×, and VO size by 28×. Crucially, the overhead for building this authenticated structure added at most 2% to the standard HNSW index construction time. The sharding optimization accelerated updates by 53×, proving that blockchain-verified ANN is feasible for dynamic, updating datasets.

---

### [Valori: A Deterministic Memory Substrate for AI Systems](https://arxiv.org/pdf/2512.22280)
*Varshith Gudur, 2025*

#### **Overview and Key Insights**: 
This paper directly tackles the 'root of all evil' in decentralized AI consensus: floating-point non-determinism. Because IEEE 754 floating-point arithmetic behaves differently across x86 and ARM architectures (due to variations in Fused Multiply-Add and SIMD parallel reductions), identical vectors can yield different binary representations. This silent data divergence breaks state replicability, preventing vector databases from being safely deployed in consensus-driven or audit-heavy blockchain environments. Valori solves this by enforcing a strict determinism boundary at the memory layer, proving that deterministic memory is an absolute prerequisite for trustworthy, multi-platform AI.

#### **Method**:
Instead of attempting to force deterministic neural network inference, Valori treats the AI memory (the vector database) as a strict state machine. It entirely replaces `f32/f64` operations with Q16.16 fixed-point arithmetic (32-bit signed integers where the lower 16 bits represent the fraction). By using standard integer ALU instructions that execute identically across all CPU architectures, Valori guarantees that vector addition, subtraction, and distance calculations (using accumulator narrowing) are bit-identical on x86, ARM, RISC-V, and WebAssembly. The memory substrate is built as a `no_std` Rust kernel, ensuring complete serialization and deterministic snapshotting.

#### **Evaluation**:
The author conducted empirical evaluations encoding text via `sentence-transformers` on both an x86 Windows PC and an ARM64 MacBook. They demonstrated that while standard cosine similarity remains high, the raw hexadecimal bit-representations diverge in every dimension when using standard floats, inherently altering K-NN distance rankings at the margins. By implementing Q16.16 fixed-point arithmetic, Valori completely eradicated bit-level divergence, successfully aligning cross-hardware state representations without catastrophic loss to semantic retrieval accuracy. 

---

### [EigenAI: Deterministic Inference, Verifiable Results](https://arxiv.org/pdf/2602.00182)
*Alves et al., 2026*

#### **Overview and Key Insights**: 
EigenAI bridges the gap between off-chain AI execution and on-chain security. While Valori addresses memory determinism via fixed-point math, EigenAI addresses *inference* determinism for Large Language Models (LLMs) on GPUs. The paper argues that current Zero-Knowledge (zkML) proofs are too slow and resource-intensive for frontier models. To solve this, EigenAI leverages an "optimistic re-execution" protocol secured by EigenLayer's restaked collateral. This allows sovereign AI agents (e.g., automated traders or on-chain judges) to run inference rapidly, with the economic guarantee that any malicious output can be challenged and proven false.

#### **Method**:
The system enforces determinism at the hardware/software boundary by utilizing fixed GPU architectures, version-pinned drivers, and custom kernels with canonical reduction orders. When an operator runs inference, they publish an encrypted receipt to a Data Availability layer (EigenDA). During a challenge window, any verifier can request a re-execution. If a dispute occurs, it is executed inside a Trusted Execution Environment (TEE) with a threshold-released decryption key. Because the GPU execution is forced to be bit-exact, verification collapses from a complex cryptographic proof down to a simple byte-equality check. If a mismatch is found, the dishonest operator's Ethereum stake is slashed.

#### **Evaluation**:
While extensive quantitative benchmarks were omitted in the provided text, the architectural evaluation demonstrates that this protocol transitions the computational burden from $O(n)$ continuous cryptographic proving (as seen in zkML) down to the cost of standard native inference. By reducing the verification to a byte-equality check during disputes, EigenAI achieves state-of-the-art steady-state latency while inheriting billions of dollars in economic security from the Ethereum validator base.

---

### [Nondeterminism-Aware Optimistic Verification for Floating-Point Neural Networks](https://arxiv.org/html/2510.16028v1)
*Yao et al., 2025*

#### **Overview and Key Insights**: 
Rather than forcing hardware to be deterministic (like EigenAI) or abandoning floating-point math (like Valori), this paper proposes a revolutionary alternative: accepting floating-point nondeterminism and building a verification protocol around it. The authors recognize that forcing strict determinism discards mature vendor libraries and cripples hardware heterogeneity. They introduce NAO, a tolerance-aware optimistic verification protocol that accepts AI outputs if they fall within mathematically sound, operator-level acceptance regions, rather than requiring bitwise equality.

#### **Method**:
NAO combines two error models: (1) IEEE-754 worst-case theoretical bounds computed per-operator, and (2) tight empirical percentile profiles calibrated across diverse hardware. During an optimistic dispute game, the computation graph is recursively partitioned via a Merkle-anchored interactive protocol until a single disputed operator (e.g., a specific matrix multiplication) is isolated. At this leaf node, adjudication relies on checking if the divergence violates the empirical thresholds or theoretical bounds. The system is implemented as a PyTorch-compatible runtime that computes these bounds in real-time while executing unmodified vendor FP32 kernels.

#### **Evaluation**:
Evaluated across CNNs, Transformers, and LLMs (e.g., Qwen3-8B) using diverse hardware (A100, H100, RTX4090), NAO introduced a negligible 0.3% latency overhead. Crucially, the authors found that their empirical thresholds were $10^2$ to $10^3$ times tighter than theoretical bounds. Bound-aware adversarial attacks, where malicious nodes tried to alter embeddings while hiding within the floating-point noise margin, achieved a 0% success rate under NAO's empirical checks, proving the system's robustness.

## 4. Open Questions and Future Directions

Based on the synthesis of these papers, several high-impact areas remain open for future research:

- **Bridging Fixed-Point Vectors with Hardware Acceleration**: While fixed-point representations (Q16.16) solve determinism natively, modern GPUs and NPUs (e.g., NVIDIA Tensor Cores) are heavily optimized for FP16, BF16, and FP8. Future research must explore how to hardware-accelerate fixed-point nearest neighbor searches without losing the deterministic guarantees required by blockchain environments.
- **Dynamic Thresholding for Graph-Based Searching**: Applying Nondeterminism-Aware Optimistic Verification (NAO) to the highly non-linear pathing of HNSW graphs remains unsolved. Because a minor floating-point divergence can cause an HNSW search to traverse an entirely different neighborhood branch, research is needed to create threshold bounds that account for graph routing divergence, rather than just tensor mathematical errors.
- **Scalability of Authenticated Data Structures (ADS)**: While Merkle-sharded HNSW trees work for millions of vectors, their update costs for billion-scale dynamic datasets on decentralized ledgers are still prohibitive. Investigating cryptographic accumulators or polynomial commitments as lightweight replacements for Merkle trees in vector indexes represents a highly promising frontier.
- **Privacy-Preserving Deterministic Search**: As Fully Homomorphic Encryption (FHE) becomes practical for vector search, maintaining determinism over encrypted manifolds is unexplored. Combining deterministic fixed-point substrates with FHE could allow smart contracts to execute private, verifiable RAG pipelines entirely on-chain.

## References

1. [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320). *Malkov, Yu. A., Yashunin, D. A.*
2. [AQR-HNSW: Accelerating Approximate Nearest Neighbor Search via Density-aware Quantization and Multi-stage Re-ranking](https://arxiv.org/abs/2602.21600). *Tewary, G. A., Gantayat, N. C., Zhang, J.*
3. [ANNProof: Building a verifiable and efficient outsourced approximate nearest neighbor search system on blockchain](https://pure.bit.edu.cn/en/publications/annproof-building-a-verifiable-and-efficient-outsourced-approxima). *Lu, L., Wen, Z., Yuan, Y., He, Q., Chen, J., Liu, Z.*
4. [Valori: A Deterministic Memory Substrate for AI Systems](https://arxiv.org/pdf/2512.22280). *Gudur, V.*
5. [EigenAI: Deterministic Inference, Verifiable Results](https://arxiv.org/pdf/2602.00182). *Alves, D. R., Patankar, V., Pereira, M., Stephens, J., Vaziri, N., Kannan, S.*
6. [Nondeterminism-Aware Optimistic Verification for Floating-Point Neural Networks](https://arxiv.org/html/2510.16028v1). *Yao, J., Su, H., Liao, T., Cheng, Z., Zhang, H., Wang, X., Viswanath, P.*
7. [OPRCP: approximate nearest neighbor binary search algorithm for hybrid data over WMSN blockchain](https://jwcn-eurasipjournals.springeropen.com/article/10.1186/s13638-018-1221-3). *Liu, H., Wei, X., Xiao, R., Chen, L., Du, X., Zhang, S.*

---

<a href="https://www.orchestra-research.com/"><img src="https://img.shields.io/badge/Orchestra-Research-6C3FC5.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PC9zdmc+" alt="Orchestra Research"></a>
