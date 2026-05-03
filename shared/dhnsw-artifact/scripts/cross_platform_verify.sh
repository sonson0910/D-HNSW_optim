#!/bin/bash
# D-HNSW Cross-Platform Verification Script
# Builds and tests on x86-64 and ARM64, compares SHA-256 hashes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/cross_platform_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="$RESULTS_DIR/report_${TIMESTAMP}.json"

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "D-HNSW Cross-Platform Determinism Verification"
echo "=============================================="

# Check docker buildx
if ! docker buildx version &>/dev/null; then
    echo "ERROR: docker buildx not found. Install Docker Buildx."
    exit 1
fi

# Create/use multiarch builder
BUILDER_NAME="dhnsw-multiarch"
if ! docker buildx inspect "$BUILDER_NAME" &>/dev/null; then
    echo "Creating multiarch builder..."
    docker buildx create --name "$BUILDER_NAME" \
        --driver docker-container \
        --platform linux/amd64,linux/arm64 \
        --use
    docker buildx inspect --bootstrap "$BUILDER_NAME"
else
    docker buildx use "$BUILDER_NAME"
fi

echo ""
echo "=== Phase 1: Build for each platform ==="

PLATFORMS=("linux/amd64" "linux/arm64")
declare -A HASHES_GRAPH
declare -A HASHES_INDEX
declare -A HASHES_DIST
declare -A BUILD_STATUS

for PLATFORM in "${PLATFORMS[@]}"; do
    ARCH=$(echo "$PLATFORM" | cut -d/ -f2)
    echo ""
    echo "--- Building for $PLATFORM ---"
    
    # Build image for specific platform
    docker buildx build \
        --platform "$PLATFORM" \
        --tag "dhnsw-verify:${ARCH}" \
        --load \
        -f "$PROJECT_DIR/Dockerfile" \
        "$PROJECT_DIR" 2>&1 | tail -5
    
    if [ $? -eq 0 ]; then
        BUILD_STATUS[$ARCH]="success"
        echo "  Build OK: $PLATFORM"
    else
        BUILD_STATUS[$ARCH]="failed"
        echo "  Build FAILED: $PLATFORM"
        continue
    fi
    
    echo "  Running determinism test..."
    
    # Run test container and capture output
    OUTPUT=$(docker run --rm --platform "$PLATFORM" \
        "dhnsw-verify:${ARCH}" \
        /app/dhnsw-verify --test-determinism 2>&1) || true
    
    # Extract hashes from output
    HASHES_GRAPH[$ARCH]=$(echo "$OUTPUT" | grep "GRAPH_HASH:" | awk '{print $2}' || echo "N/A")
    HASHES_INDEX[$ARCH]=$(echo "$OUTPUT" | grep "INDEX_HASH:" | awk '{print $2}' || echo "N/A")
    HASHES_DIST[$ARCH]=$(echo "$OUTPUT" | grep "DIST_HASH:" | awk '{print $2}' || echo "N/A")
    
    echo "  Graph Hash: ${HASHES_GRAPH[$ARCH]:-N/A}"
    echo "  Index Hash: ${HASHES_INDEX[$ARCH]:-N/A}"
    echo "  Dist Hash:  ${HASHES_DIST[$ARCH]:-N/A}"
done

echo ""
echo "=== Phase 2: Cross-Platform Hash Comparison ==="

GRAPH_MATCH="false"
INDEX_MATCH="false"
DIST_MATCH="false"

if [ "${HASHES_GRAPH[amd64]:-}" = "${HASHES_GRAPH[arm64]:-}" ] && \
   [ "${HASHES_GRAPH[amd64]:-}" != "N/A" ] && \
   [ -n "${HASHES_GRAPH[amd64]:-}" ]; then
    GRAPH_MATCH="true"
    echo "Graph Hash:  MATCH (${HASHES_GRAPH[amd64]:0:16}...)"
else
    echo "Graph Hash:  MISMATCH"
    echo "  amd64: ${HASHES_GRAPH[amd64]:-N/A}"
    echo "  arm64: ${HASHES_GRAPH[arm64]:-N/A}"
fi

if [ "${HASHES_INDEX[amd64]:-}" = "${HASHES_INDEX[arm64]:-}" ] && \
   [ "${HASHES_INDEX[amd64]:-}" != "N/A" ] && \
   [ -n "${HASHES_INDEX[amd64]:-}" ]; then
    INDEX_MATCH="true"
    echo "Index Hash:  MATCH (${HASHES_INDEX[amd64]:0:16}...)"
else
    echo "Index Hash:  MISMATCH"
fi

if [ "${HASHES_DIST[amd64]:-}" = "${HASHES_DIST[arm64]:-}" ] && \
   [ "${HASHES_DIST[amd64]:-}" != "N/A" ] && \
   [ -n "${HASHES_DIST[amd64]:-}" ]; then
    DIST_MATCH="true"
    echo "Dist Hash:   MATCH (${HASHES_DIST[amd64]:0:16}...)"
else
    echo "Dist Hash:   MISMATCH"
fi

ALL_PASS="false"
if [ "$GRAPH_MATCH" = "true" ] && [ "$INDEX_MATCH" = "true" ] && [ "$DIST_MATCH" = "true" ]; then
    ALL_PASS="true"
fi

echo ""
echo "=== RESULT: $([ "$ALL_PASS" = "true" ] && echo 'ALL PASS' || echo 'SOME FAILED') ==="

# Generate JSON report
cat > "$REPORT" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "platforms": ["linux/amd64", "linux/arm64"],
  "results": {
    "amd64": {
      "build": "${BUILD_STATUS[amd64]:-failed}",
      "graph_hash": "${HASHES_GRAPH[amd64]:-N/A}",
      "index_hash": "${HASHES_INDEX[amd64]:-N/A}",
      "dist_hash": "${HASHES_DIST[amd64]:-N/A}"
    },
    "arm64": {
      "build": "${BUILD_STATUS[arm64]:-failed}",
      "graph_hash": "${HASHES_GRAPH[arm64]:-N/A}",
      "index_hash": "${HASHES_INDEX[arm64]:-N/A}",
      "dist_hash": "${HASHES_DIST[arm64]:-N/A}"
    }
  },
  "comparison": {
    "graph_match": $GRAPH_MATCH,
    "index_match": $INDEX_MATCH,
    "dist_match": $DIST_MATCH,
    "all_pass": $ALL_PASS
  }
}
EOF

echo ""
echo "Report saved: $REPORT"
