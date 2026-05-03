# ============================================================
#  D-HNSW Artifact Evaluation — Dockerfile
#  Deterministic HNSW for Blockchain & Verifiable AI
# ============================================================
#  This container reproduces ALL experiments from the paper:
#    - Fixed-point accuracy benchmarks (Table II–V)
#    - Determinism verification via SHA-256 (Table VI)
#    - Cross-architecture verification (x86-64 / ARM64 / RISC-V)
#    - SIFT-1M real-world validation (Table IV)
#    - Ablation study (Figure 6)
#    - Performance overhead analysis (Figure 8)
#
#  Build:  docker build -t dhnsw-artifact .
#  Run:    docker run --rm -v $(pwd)/output:/artifact/output dhnsw-artifact
#  Shell:  docker run --rm -it dhnsw-artifact bash
# ============================================================

FROM rust:1.78-bookworm AS builder

# Install cross-compilation toolchains for determinism verification
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-aarch64-linux-gnu \
    gcc-riscv64-linux-gnu \
    libc6-dev-arm64-cross \
    libc6-dev-riscv64-cross \
    qemu-user-static \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    sha256sum \
    && rm -rf /var/lib/apt/lists/*

# Add Rust cross-compilation targets
RUN rustup target add \
    x86_64-unknown-linux-gnu \
    aarch64-unknown-linux-gnu \
    riscv64gc-unknown-linux-gnu

# Configure cross-compilation linkers
RUN mkdir -p /.cargo && echo '\
[target.aarch64-unknown-linux-gnu]\n\
linker = "aarch64-linux-gnu-gcc"\n\
\n\
[target.riscv64gc-unknown-linux-gnu]\n\
linker = "riscv64-linux-gnu-gcc"\n\
' > /.cargo/config.toml

WORKDIR /artifact

# ---- Copy project files ----
COPY Cargo.toml Cargo.lock* ./
COPY src/ ./src/
COPY benches/ ./benches/
COPY tests/ ./tests/

# ---- Build for all architectures ----
RUN cargo build --release --target x86_64-unknown-linux-gnu
RUN cargo build --release --target aarch64-unknown-linux-gnu || echo "ARM64 cross-build: optional"
RUN cargo build --release --target riscv64gc-unknown-linux-gnu || echo "RISC-V cross-build: optional"

# ---- Runtime stage ----
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    qemu-user-static \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment for experiment scripts
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir \
    numpy>=1.24 \
    matplotlib>=3.7 \
    h5py>=3.8 \
    scipy>=1.10 \
    tqdm>=4.65

WORKDIR /artifact

# Copy built binaries
COPY --from=builder /artifact/target/x86_64-unknown-linux-gnu/release/dhnsw /usr/local/bin/dhnsw
COPY --from=builder /artifact/target/aarch64-unknown-linux-gnu/release/dhnsw /usr/local/bin/dhnsw-arm64 
COPY --from=builder /artifact/target/riscv64gc-unknown-linux-gnu/release/dhnsw /usr/local/bin/dhnsw-riscv64

# Copy experiment infrastructure
COPY experiments/ ./experiments/
COPY scripts/ ./scripts/
COPY datasets/ ./datasets/
COPY paper/ ./paper/

# Create output directory
RUN mkdir -p /artifact/output/figures \
             /artifact/output/tables \
             /artifact/output/logs \
             /artifact/output/hashes

# Make scripts executable
RUN chmod +x scripts/*.sh

# ---- Entry point ----
ENTRYPOINT ["/artifact/scripts/run_all.sh"]
CMD ["--full"]
