#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# D-HNSW Cross-Platform Determinism Test
# ──────────────────────────────────────────────────────────────────────────────
#
# This script verifies that D-HNSW produces bit-identical graphs on the current
# platform. For true cross-platform verification, run this on:
#   - x86_64 Linux/macOS
#   - ARM64 (Apple Silicon / Raspberry Pi / AWS Graviton)
#
# Then compare the SHA-256 hashes printed by each run.
#
# Usage:
#   ./scripts/cross_platform_test.sh
#
# Requirements:
#   - Rust toolchain (stable)
#   - cargo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "═══════════════════════════════════════════════════════════════"
echo " D-HNSW Cross-Platform Determinism Verification"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo " Platform:  $(uname -m)"
echo " OS:        $(uname -s)"
echo " Rust:      $(rustc --version 2>/dev/null || echo 'not found')"
echo " Date:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

# ── Step 1: Build ────────────────────────────────────────────────────────
echo "▶ Building D-HNSW (release mode)..."
cargo build --release --features benchmarks 2>&1

# ── Step 2: Run unit tests ───────────────────────────────────────────────
echo ""
echo "▶ Running unit tests..."
cargo test --release 2>&1

# ── Step 3: Run determinism benchmark ────────────────────────────────────
echo ""
echo "▶ Running determinism verification..."
cargo bench --bench cross_platform_determinism --features benchmarks 2>&1 | head -50

# ── Step 4: Run ablation benchmarks ──────────────────────────────────────
echo ""
echo "▶ Running ablation benchmarks..."
cargo bench --bench ablation_benchmark --features benchmarks 2>&1 | head -100

# ── Step 5: Summary ─────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " ✓ All tests passed on $(uname -m)"
echo ""
echo " To verify cross-platform determinism:"
echo "   1. Run this script on x86_64 and ARM64"
echo "   2. Compare the SHA-256 hash printed above"
echo "   3. Hashes MUST be identical for the paper's claim to hold"
echo "═══════════════════════════════════════════════════════════════"
