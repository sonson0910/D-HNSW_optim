#!/bin/bash
# ============================================================
#  D-HNSW Artifact Evaluation — Master Runner
#  Usage:
#    ./run_all.sh --full      # Run ALL experiments (~2-4 hours)
#    ./run_all.sh --quick     # Quick smoke test (~10 minutes)
#    ./run_all.sh --verify    # Determinism verification only (~5 min)
#    ./run_all.sh --bench     # Performance benchmarks only (~1 hour)
#    ./run_all.sh --cross     # Cross-architecture test only (~15 min)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARTIFACT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${ARTIFACT_DIR}/output"
LOG_FILE="${OUTPUT_DIR}/logs/experiment_$(date +%Y%m%d_%H%M%S).log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"; }
ok()  { echo -e "${GREEN}[✓]${NC} $1" | tee -a "$LOG_FILE"; }
err() { echo -e "${RED}[✗]${NC} $1" | tee -a "$LOG_FILE"; }
warn(){ echo -e "${YELLOW}[!]${NC} $1" | tee -a "$LOG_FILE"; }

mkdir -p "${OUTPUT_DIR}/figures" "${OUTPUT_DIR}/tables" "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/hashes"

MODE="${1:---full}"

# ============================================================
#  Banner
# ============================================================
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     D-HNSW: Deterministic HNSW for Blockchain & AI     ║"
echo "║            Artifact Evaluation Package v1.0             ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Mode: $(printf '%-47s' "$MODE") ║"
echo "║  Output: $(printf '%-45s' "$OUTPUT_DIR") ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ============================================================
#  Step 0: Environment Check
# ============================================================
log "Step 0: Checking environment..."

check_binary() {
    if command -v "$1" &>/dev/null; then
        ok "$1 found: $(command -v "$1")"
        return 0
    else
        warn "$1 not found"
        return 1
    fi
}

check_binary dhnsw
check_binary python3
HAVE_ARM64=false; check_binary dhnsw-arm64 && HAVE_ARM64=true
HAVE_RISCV=false; check_binary dhnsw-riscv64 && HAVE_RISCV=true
HAVE_QEMU=false; check_binary qemu-aarch64-static && HAVE_QEMU=true

# ============================================================
#  Step 1: Download Datasets (if not present)
# ============================================================
download_datasets() {
    log "Step 1: Checking datasets..."
    
    DATASET_DIR="${ARTIFACT_DIR}/datasets"
    mkdir -p "$DATASET_DIR"
    
    # SIFT-1M (128-d, 1M vectors, ~500MB)
    if [ ! -f "${DATASET_DIR}/sift-128-euclidean.hdf5" ]; then
        log "Downloading SIFT-1M from ann-benchmarks..."
        wget -q --show-progress -O "${DATASET_DIR}/sift-128-euclidean.hdf5" \
            "http://ann-benchmarks.com/sift-128-euclidean.hdf5" || {
            warn "SIFT-1M download failed. Using synthetic data."
        }
    else
        ok "SIFT-1M dataset found"
    fi
    
    # GloVe-100 (100-d, 1.2M vectors)
    if [ ! -f "${DATASET_DIR}/glove-100-angular.hdf5" ]; then
        log "Downloading GloVe-100..."
        wget -q --show-progress -O "${DATASET_DIR}/glove-100-angular.hdf5" \
            "http://ann-benchmarks.com/glove-100-angular.hdf5" || {
            warn "GloVe-100 download failed."
        }
    else
        ok "GloVe-100 dataset found"
    fi
    
    # Fashion-MNIST (784-d, 60K vectors)
    if [ ! -f "${DATASET_DIR}/fashion-mnist-784-euclidean.hdf5" ]; then
        log "Downloading Fashion-MNIST..."
        wget -q --show-progress -O "${DATASET_DIR}/fashion-mnist-784-euclidean.hdf5" \
            "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5" || {
            warn "Fashion-MNIST download failed."
        }
    else
        ok "Fashion-MNIST dataset found"
    fi
    
    # GIST-960 (960-d, 1M vectors, ~4GB) — only for --full mode
    if [ "$MODE" = "--full" ]; then
        if [ ! -f "${DATASET_DIR}/gist-960-euclidean.hdf5" ]; then
            log "Downloading GIST-960 (large, ~4GB)..."
            wget -q --show-progress -O "${DATASET_DIR}/gist-960-euclidean.hdf5" \
                "http://ann-benchmarks.com/gist-960-euclidean.hdf5" || {
                warn "GIST-960 download failed."
            }
        else
            ok "GIST-960 dataset found"
        fi
    fi
}

# ============================================================
#  Step 2: Determinism Verification (Core Claim)
# ============================================================
run_determinism_verification() {
    log "Step 2: Determinism Verification (SHA-256 hash matching)..."
    
    HASH_DIR="${OUTPUT_DIR}/hashes"
    
    # Run D-HNSW index construction 5 times with same seed
    for run in $(seq 1 5); do
        log "  Run ${run}/5: Building index..."
        dhnsw build \
            --input "${ARTIFACT_DIR}/datasets/sift-128-euclidean.hdf5" \
            --output "${HASH_DIR}/index_run${run}.bin" \
            --M 16 --ef-construction 200 \
            --seed 42 \
            --format q32_32 \
            2>&1 | tee -a "$LOG_FILE"
        
        # Compute SHA-256 hash
        sha256sum "${HASH_DIR}/index_run${run}.bin" | tee -a "${HASH_DIR}/hashes.txt"
    done
    
    # Verify all hashes are identical
    UNIQUE_HASHES=$(awk '{print $1}' "${HASH_DIR}/hashes.txt" | sort -u | wc -l)
    if [ "$UNIQUE_HASHES" -eq 1 ]; then
        ok "DETERMINISM VERIFIED: All 5 runs produce identical SHA-256 hash"
        echo "PASS" > "${OUTPUT_DIR}/determinism_result.txt"
    else
        err "DETERMINISM FAILED: Found ${UNIQUE_HASHES} distinct hashes"
        echo "FAIL" > "${OUTPUT_DIR}/determinism_result.txt"
        exit 1
    fi
}

# ============================================================
#  Step 3: Cross-Architecture Verification
# ============================================================
run_cross_arch_verification() {
    log "Step 3: Cross-Architecture Determinism Verification..."
    
    HASH_DIR="${OUTPUT_DIR}/hashes"
    
    # x86-64 (native)
    log "  Building index on x86-64 (native)..."
    dhnsw build \
        --input "${ARTIFACT_DIR}/datasets/sift-128-euclidean.hdf5" \
        --output "${HASH_DIR}/index_x86_64.bin" \
        --M 16 --ef-construction 200 --seed 42 --format q32_32 \
        --scale 10000 \
        2>&1 | tee -a "$LOG_FILE"
    X86_HASH=$(sha256sum "${HASH_DIR}/index_x86_64.bin" | awk '{print $1}')
    ok "  x86-64 hash: ${X86_HASH:0:16}..."
    
    # ARM64 (via QEMU)
    if $HAVE_ARM64 && $HAVE_QEMU; then
        log "  Building index on ARM64 (QEMU)..."
        qemu-aarch64-static /usr/local/bin/dhnsw-arm64 build \
            --input "${ARTIFACT_DIR}/datasets/sift-128-euclidean.hdf5" \
            --output "${HASH_DIR}/index_arm64.bin" \
            --M 16 --ef-construction 200 --seed 42 --format q32_32 \
            --scale 10000 \
            2>&1 | tee -a "$LOG_FILE"
        ARM64_HASH=$(sha256sum "${HASH_DIR}/index_arm64.bin" | awk '{print $1}')
        ok "  ARM64 hash:  ${ARM64_HASH:0:16}..."
        
        if [ "$X86_HASH" = "$ARM64_HASH" ]; then
            ok "  CROSS-ARCH MATCH: x86-64 ≡ ARM64 ✓"
        else
            err "  CROSS-ARCH MISMATCH: x86-64 ≠ ARM64"
        fi
    else
        warn "  ARM64 binary or QEMU not available, skipping"
    fi
    
    # RISC-V (via QEMU)
    if $HAVE_RISCV && $HAVE_QEMU; then
        log "  Building index on RISC-V 64 (QEMU)..."
        qemu-riscv64-static /usr/local/bin/dhnsw-riscv64 build \
            --input "${ARTIFACT_DIR}/datasets/sift-128-euclidean.hdf5" \
            --output "${HASH_DIR}/index_riscv64.bin" \
            --M 16 --ef-construction 200 --seed 42 --format q32_32 \
            --scale 10000 \
            2>&1 | tee -a "$LOG_FILE"
        RISCV_HASH=$(sha256sum "${HASH_DIR}/index_riscv64.bin" | awk '{print $1}')
        ok "  RISC-V hash: ${RISCV_HASH:0:16}..."
        
        if [ "$X86_HASH" = "$RISCV_HASH" ]; then
            ok "  CROSS-ARCH MATCH: x86-64 ≡ RISC-V ✓"
        else
            err "  CROSS-ARCH MISMATCH: x86-64 ≠ RISC-V"
        fi
    else
        warn "  RISC-V binary or QEMU not available, skipping"
    fi
    
    # Summary
    echo "=== Cross-Architecture Verification ===" > "${OUTPUT_DIR}/cross_arch_result.txt"
    echo "x86-64: ${X86_HASH}" >> "${OUTPUT_DIR}/cross_arch_result.txt"
    [ -n "${ARM64_HASH:-}" ] && echo "ARM64:  ${ARM64_HASH}" >> "${OUTPUT_DIR}/cross_arch_result.txt"
    [ -n "${RISCV_HASH:-}" ] && echo "RISCV:  ${RISCV_HASH}" >> "${OUTPUT_DIR}/cross_arch_result.txt"
}

# ============================================================
#  Step 4: Recall & Accuracy Benchmarks
# ============================================================
run_recall_benchmarks() {
    log "Step 4: Recall & Accuracy Benchmarks..."
    
    python3 "${ARTIFACT_DIR}/experiments/run_recall_benchmark.py" \
        --datasets-dir "${ARTIFACT_DIR}/datasets" \
        --output-dir "${OUTPUT_DIR}" \
        --scales 10000 50000 100000 200000 \
        --ef-values 16 32 64 128 256 \
        --M 16 \
        --format q32_32 \
        2>&1 | tee -a "$LOG_FILE"
    
    ok "Recall benchmarks complete. Results in ${OUTPUT_DIR}/tables/"
}

# ============================================================
#  Step 5: Performance Benchmarks
# ============================================================
run_performance_benchmarks() {
    log "Step 5: Performance Benchmarks..."
    
    python3 "${ARTIFACT_DIR}/experiments/run_performance_benchmark.py" \
        --datasets-dir "${ARTIFACT_DIR}/datasets" \
        --output-dir "${OUTPUT_DIR}" \
        --iterations 10 \
        --warmup 3 \
        2>&1 | tee -a "$LOG_FILE"
    
    ok "Performance benchmarks complete. Results in ${OUTPUT_DIR}/tables/"
}

# ============================================================
#  Step 6: Generate Figures
# ============================================================
generate_figures() {
    log "Step 6: Generating publication figures..."
    
    python3 "${ARTIFACT_DIR}/experiments/generate_plots.py" \
        --results-dir "${OUTPUT_DIR}" \
        --figures-dir "${OUTPUT_DIR}/figures" \
        2>&1 | tee -a "$LOG_FILE"
    
    ok "Figures generated in ${OUTPUT_DIR}/figures/"
}

# ============================================================
#  Step 7: Generate Summary Report
# ============================================================
generate_report() {
    log "Step 7: Generating summary report..."
    
    REPORT="${OUTPUT_DIR}/ARTIFACT_REPORT.md"
    cat > "$REPORT" << 'HEADER'
# D-HNSW Artifact Evaluation Report

Generated automatically by the D-HNSW artifact evaluation package.

## Environment
HEADER
    
    echo "- **Date**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$REPORT"
    echo "- **Hostname**: $(hostname)" >> "$REPORT"
    echo "- **Architecture**: $(uname -m)" >> "$REPORT"
    echo "- **OS**: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 || echo 'N/A')" >> "$REPORT"
    echo "- **Rust**: $(rustc --version 2>/dev/null || echo 'N/A')" >> "$REPORT"
    echo "" >> "$REPORT"
    
    echo "## Results Summary" >> "$REPORT"
    echo "" >> "$REPORT"
    
    # Determinism
    if [ -f "${OUTPUT_DIR}/determinism_result.txt" ]; then
        RESULT=$(cat "${OUTPUT_DIR}/determinism_result.txt")
        echo "### Determinism Verification: **${RESULT}**" >> "$REPORT"
        if [ -f "${OUTPUT_DIR}/hashes/hashes.txt" ]; then
            echo '```' >> "$REPORT"
            cat "${OUTPUT_DIR}/hashes/hashes.txt" >> "$REPORT"
            echo '```' >> "$REPORT"
        fi
    fi
    
    # Cross-arch
    if [ -f "${OUTPUT_DIR}/cross_arch_result.txt" ]; then
        echo "" >> "$REPORT"
        echo "### Cross-Architecture Verification" >> "$REPORT"
        echo '```' >> "$REPORT"
        cat "${OUTPUT_DIR}/cross_arch_result.txt" >> "$REPORT"
        echo '```' >> "$REPORT"
    fi
    
    echo "" >> "$REPORT"
    echo "## Generated Artifacts" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "| Artifact | Path |" >> "$REPORT"
    echo "|----------|------|" >> "$REPORT"
    find "${OUTPUT_DIR}" -type f -name "*.pdf" -o -name "*.json" -o -name "*.csv" -o -name "*.txt" 2>/dev/null | \
        sort | while read f; do
            echo "| $(basename "$f") | $(realpath --relative-to="${OUTPUT_DIR}" "$f") |" >> "$REPORT"
        done
    
    ok "Report generated: ${REPORT}"
}

# ============================================================
#  Main Execution
# ============================================================
case "$MODE" in
    --quick)
        log "Running QUICK smoke test..."
        download_datasets
        run_determinism_verification
        generate_report
        ;;
    --verify)
        log "Running DETERMINISM VERIFICATION only..."
        download_datasets
        run_determinism_verification
        run_cross_arch_verification
        generate_report
        ;;
    --bench)
        log "Running PERFORMANCE BENCHMARKS only..."
        download_datasets
        run_recall_benchmarks
        run_performance_benchmarks
        generate_figures
        generate_report
        ;;
    --cross)
        log "Running CROSS-ARCHITECTURE test only..."
        download_datasets
        run_cross_arch_verification
        generate_report
        ;;
    --full)
        log "Running FULL artifact evaluation..."
        download_datasets
        run_determinism_verification
        run_cross_arch_verification
        run_recall_benchmarks
        run_performance_benchmarks
        generate_figures
        generate_report
        ;;
    *)
        echo "Usage: $0 [--full|--quick|--verify|--bench|--cross]"
        exit 1
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              Artifact Evaluation Complete               ║"
echo "║  Results: ${OUTPUT_DIR}                                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
