"""
D-HNSW LLM Embeddings Experiment (1536 dimensions)
====================================================
Downloads DBPedia-OpenAI3 embeddings from HuggingFace,
converts to HDF5, runs I64F32 search + determinism verification.
"""

import modal
import os
import json
import time
import hashlib
import numpy as np
from datetime import datetime

app = modal.App("dhnsw-llm-1536d")

sdk_path = os.environ.get('ORCHESTRA_SDK_PATH', '/root/vm_worker/src')

ml_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system numpy h5py hnswlib scipy scikit-learn "
        "requests tqdm datasets huggingface_hub"
    )
    .env({
        "AGENT_ID": os.getenv("AGENT_ID", ""),
        "PROJECT_ID": os.getenv("PROJECT_ID", ""),
        "USER_ID": os.getenv("USER_ID", ""),
    })
    .add_local_dir(sdk_path, remote_path="/root/src")
)

volume = modal.Volume.from_name("dhnsw-data", create_if_missing=True)


def compute_safe_frac_bits(max_abs_val, dims):
    """Compute safe fractional bits for I64F32 to avoid overflow."""
    import math
    if max_abs_val == 0:
        return 32
    log2_d = math.log2(max(dims, 1))
    log2_max = math.log2(max(max_abs_val, 1e-10))
    safe_frac = (62 - log2_d - 2 * log2_max) / 2
    return max(int(safe_frac), 8)


class I64F32DistanceL2:
    """I64F32 fixed-point L2 distance calculator."""
    def __init__(self, data, frac_bits):
        self.frac_bits = frac_bits
        self.scale = 2 ** frac_bits
        self.inv_scale_sq = 1.0 / (self.scale * self.scale)
        self.data_fixed = np.round(
            data.astype(np.float64) * self.scale
        ).astype(np.int64)

    def distance_batch(self, query_fixed, indices):
        diffs = query_fixed - self.data_fixed[indices]
        return np.sum(
            diffs.astype(np.float64) * diffs.astype(np.float64), axis=1
        ) * self.inv_scale_sq

    def convert_query(self, query):
        return np.round(
            query.astype(np.float64) * self.scale
        ).astype(np.int64)


class I64F32DistanceCosine:
    """I64F32 fixed-point Cosine distance calculator."""
    def __init__(self, data, frac_bits):
        self.frac_bits = frac_bits
        self.scale = 2 ** frac_bits
        self.inv_scale_sq = 1.0 / (self.scale * self.scale)
        self.data_fixed = np.round(
            data.astype(np.float64) * self.scale
        ).astype(np.int64)
        self.norms_sq = np.sum(
            self.data_fixed.astype(np.float64) ** 2, axis=1
        ) * self.inv_scale_sq

    def distance_batch(self, query_fixed, indices):
        dots = np.sum(
            query_fixed.astype(np.float64) *
            self.data_fixed[indices].astype(np.float64), axis=1
        ) * self.inv_scale_sq
        norm_q = float(np.sum(
            query_fixed.astype(np.float64) ** 2
        )) * self.inv_scale_sq
        denoms = np.sqrt(norm_q * self.norms_sq[indices])
        denoms = np.maximum(denoms, 1e-30)
        return 1.0 - dots / denoms

    def convert_query(self, query):
        return np.round(
            query.astype(np.float64) * self.scale
        ).astype(np.int64)


class I32F16DistanceL2:
    """I32F16 fast approximate distance for two-phase search."""
    def __init__(self, data, frac_bits):
        self.frac_bits = frac_bits
        self.scale = 2 ** frac_bits
        self.inv_scale_sq = 1.0 / (self.scale * self.scale)
        scaled = np.round(data.astype(np.float64) * self.scale)
        scaled = np.clip(scaled, -2**31 + 1, 2**31 - 1)
        self.data_fixed = scaled.astype(np.int32)

    def distance_batch(self, query_fixed, indices):
        diffs = (query_fixed.astype(np.int64) -
                 self.data_fixed[indices].astype(np.int64))
        return np.sum(diffs * diffs, axis=1) * self.inv_scale_sq

    def convert_query(self, query):
        scaled = np.round(query.astype(np.float64) * self.scale)
        scaled = np.clip(scaled, -2**31 + 1, 2**31 - 1)
        return scaled.astype(np.int32)


def compute_recall(labels, gt, k=10):
    recall = 0
    for i in range(len(labels)):
        gt_set = set(gt[i, :k].tolist())
        pred_set = set(labels[i, :k].tolist() if hasattr(labels[i], 'tolist')
                       else list(labels[i][:k]))
        recall += len(gt_set & pred_set) / k
    return recall / len(labels)


def download_dbpedia_openai3(dest_dir="/workspace/datasets"):
    """Download DBPedia-OpenAI3-1536d from HuggingFace and convert to HDF5."""
    import h5py
    from datasets import load_dataset
    from sklearn.neighbors import NearestNeighbors

    hdf5_path = os.path.join(dest_dir, "dbpedia-openai3-1536-cosine.hdf5")
    if os.path.exists(hdf5_path):
        print(f"  Dataset already exists: {hdf5_path}")
        return hdf5_path

    os.makedirs(dest_dir, exist_ok=True)
    print("  Downloading DBPedia-OpenAI3-1536d from HuggingFace...")

    ds = load_dataset(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K",
        split="train"
    )

    print(f"  Loaded {len(ds)} records")

    # Extract embeddings
    all_embeddings = np.array(ds["openai3-text-embedding-3-large-1536"],
                              dtype=np.float32)
    print(f"  Embeddings shape: {all_embeddings.shape}")

    n_total = len(all_embeddings)
    n_train = min(n_total - 1000, 99000)
    n_test = 1000

    # Shuffle with fixed seed for reproducibility
    rng = np.random.RandomState(2024)
    indices = rng.permutation(n_total)

    train_data = all_embeddings[indices[:n_train]]
    test_data = all_embeddings[indices[n_train:n_train + n_test]]

    # Compute ground truth neighbors (brute-force cosine)
    print("  Computing ground truth neighbors (cosine)...")
    nn = NearestNeighbors(n_neighbors=100, metric='cosine', algorithm='brute')
    nn.fit(train_data)
    distances, neighbors = nn.kneighbors(test_data)

    # Save as HDF5
    print(f"  Saving to {hdf5_path}...")
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('train', data=train_data)
        f.create_dataset('test', data=test_data)
        f.create_dataset('neighbors', data=neighbors.astype(np.int32))
        f.create_dataset('distances', data=distances.astype(np.float32))
        f.attrs['metric'] = 'cosine'
        f.attrs['dimension'] = 1536
        f.attrs['source'] = 'DBPedia-OpenAI3-text-embedding-3-large'
        f.attrs['n_train'] = n_train
        f.attrs['n_test'] = n_test

    print(f"  Done! Train: {n_train}, Test: {n_test}, Dim: 1536")
    return hdf5_path


# ============================================================
# EXPERIMENT: LLM Embeddings 1536d Full Pipeline
# ============================================================
@app.function(
    image=ml_image,
    volumes={"/workspace": volume},
    cpu=4,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("orchestra-supabase")]
)
def experiment_llm_1536d():
    """
    Full D-HNSW experiment on 1536-dimensional LLM embeddings.
    Tests: I64F32 search, Two-Phase search, SHA-256 determinism.
    """
    import sys
    sys.path.insert(0, "/root")
    from src.orchestra_sdk.experiment import Experiment
    import hnswlib
    import h5py

    os.makedirs("/workspace/results", exist_ok=True)

    exp = Experiment.init(
        name="D-HNSW LLM Embeddings 1536d",
        description="Full D-HNSW pipeline on OpenAI 1536d embeddings",
        config={"dataset": "dbpedia-openai3-1536", "dim": 1536},
        x_axis_label="Experiment"
    )

    # Step 1: Download dataset
    print("=" * 60)
    print("STEP 1: Download DBPedia-OpenAI3-1536d")
    print("=" * 60)
    hdf5_path = download_dbpedia_openai3()
    volume.commit()

    # Load data
    with h5py.File(hdf5_path, 'r') as f:
        base = np.array(f['train'])
        queries = np.array(f['test'][:500])
        gt = np.array(f['neighbors'][:500])

    n, d = base.shape
    nq = queries.shape[0]
    print(f"\nData: {n} vectors, {d} dims, {nq} queries")

    # Stats
    max_abs = max(np.max(np.abs(base)), np.max(np.abs(queries)))
    mean_abs = np.mean(np.abs(base))
    print(f"Value range: max_abs={max_abs:.4f}, mean_abs={mean_abs:.4f}")

    frac_bits = compute_safe_frac_bits(max_abs, d)
    print(f"Frac bits: {frac_bits}, Scale: 2^{frac_bits}")

    # Step 2: I64F32 Search (Experiment A)
    print("\n" + "=" * 60)
    print("STEP 2: I64F32 Search (Cosine)")
    print("=" * 60)

    dist_calc = I64F32DistanceCosine(base, frac_bits)

    # Build hnswlib index
    idx = hnswlib.Index(space='cosine', dim=d)
    idx.init_index(max_elements=n, M=16, ef_construction=200)
    idx.add_items(base)

    ef_values = [10, 20, 40, 64, 100, 128, 200]
    f32_results = []
    i64_results = []

    print("\n  --- Float32 Baseline ---")
    for ef in ef_values:
        idx.set_ef(ef)
        t0 = time.time()
        labels, _ = idx.knn_query(queries, k=10)
        qt = time.time() - t0
        recall = compute_recall(labels, gt)
        qps = nq / qt
        f32_results.append({
            "ef": ef, "recall_at_10": round(recall, 4),
            "qps": round(qps, 1),
            "latency_us": round(qt / nq * 1e6, 1)
        })
        print(f"    ef={ef:>4d}: R@10={recall:.4f}, QPS={qps:.0f}")

    print("\n  --- D-HNSW I64F32 Cosine ---")
    for ef in ef_values:
        idx.set_ef(ef)
        candidates, _ = idx.knn_query(queries, k=ef)

        t0 = time.time()
        all_labels = np.zeros((nq, 10), dtype=np.int64)
        for i in range(nq):
            q_fixed = dist_calc.convert_query(queries[i])
            dists = dist_calc.distance_batch(q_fixed, candidates[i])
            sorted_idx = np.argsort(dists)[:10]
            all_labels[i] = candidates[i][sorted_idx]
        qt = time.time() - t0

        recall = compute_recall(all_labels, gt)
        qps = nq / qt
        i64_results.append({
            "ef": ef, "recall_at_10": round(recall, 4),
            "qps": round(qps, 1),
            "latency_us": round(qt / nq * 1e6, 1)
        })
        print(f"    ef={ef:>4d}: R@10={recall:.4f}, QPS={qps:.0f}")

    # Overhead at ef=128
    f32_128 = next(r for r in f32_results if r["ef"] == 128)
    i64_128 = next(r for r in i64_results if r["ef"] == 128)
    overhead = f32_128["qps"] / max(i64_128["qps"], 1)
    print(f"\n  Overhead at ef=128: {overhead:.2f}x")
    print(f"  Recall diff: {i64_128['recall_at_10'] - f32_128['recall_at_10']:.4f}")

    # Step 3: Two-Phase Search (Experiment B)
    print("\n" + "=" * 60)
    print("STEP 3: Two-Phase Search")
    print("=" * 60)

    import math
    frac_bits_32 = max(int((30 - math.log2(max(d, 1)) -
                            2 * math.log2(max(max_abs, 1e-10))) / 2), 4)

    dist_i32 = I32F16DistanceL2(base, frac_bits_32)
    dist_i64_l2 = I64F32DistanceL2(base, frac_bits)

    tp_results = []
    for ef in [64, 128, 200]:
        idx.set_ef(ef)
        candidates, _ = idx.knn_query(queries, k=ef)
        phase1_k = min(30, ef)

        t0 = time.time()
        all_labels = np.zeros((nq, 10), dtype=np.int64)
        for i in range(nq):
            q32 = dist_i32.convert_query(queries[i])
            d32 = dist_i32.distance_batch(q32, candidates[i])
            top_idx = np.argsort(d32)[:phase1_k]
            top_cands = candidates[i][top_idx]

            q64 = dist_i64_l2.convert_query(queries[i])
            d64 = dist_i64_l2.distance_batch(q64, top_cands)
            final = np.argsort(d64)[:10]
            all_labels[i] = top_cands[final]
        qt = time.time() - t0

        recall = compute_recall(all_labels, gt)
        qps = nq / qt
        tp_results.append({
            "ef": ef, "recall_at_10": round(recall, 4),
            "qps": round(qps, 1)
        })
        print(f"    ef={ef}: R@10={recall:.4f}, QPS={qps:.0f}")

    # Step 4: SHA-256 Determinism (Experiment C)
    print("\n" + "=" * 60)
    print("STEP 4: SHA-256 Determinism Verification")
    print("=" * 60)

    N_RUNS = 5
    scale = 2 ** frac_bits
    base_fixed = np.round(
        base[:10000].astype(np.float64) * scale
    ).astype(np.int64)
    test_queries = queries[:100]

    # Distance determinism
    rng = np.random.RandomState(42)
    idx_a = rng.randint(0, 10000, 500)
    idx_b = rng.randint(0, 10000, 500)

    dist_hashes = []
    for run in range(N_RUNS):
        dists = np.zeros(500, dtype=np.int64)
        for i in range(500):
            diff = base_fixed[idx_a[i]] - base_fixed[idx_b[i]]
            dists[i] = np.sum(diff.astype(np.int64) * diff.astype(np.int64))
        h = hashlib.sha256(dists.tobytes()).hexdigest()
        dist_hashes.append(h)
        print(f"    Distance run {run+1}: {h[:16]}...")

    dist_ok = len(set(dist_hashes)) == 1
    print(f"  Distance determinism: {'PASS' if dist_ok else 'FAIL'}")

    # Search determinism
    search_hashes = []
    for run in range(N_RUNS):
        all_labels = np.zeros((len(test_queries), 10), dtype=np.int64)
        all_dists = np.zeros((len(test_queries), 10), dtype=np.float64)
        for i in range(len(test_queries)):
            q = np.round(
                test_queries[i].astype(np.float64) * scale
            ).astype(np.int64)
            inv_sq = 1.0 / (scale * scale)
            batch_dists = np.sum(
                (q - base_fixed).astype(np.float64) ** 2, axis=1
            ) * inv_sq
            top_k = np.argsort(batch_dists)[:10]
            all_labels[i] = top_k
            all_dists[i] = batch_dists[top_k]

        h_obj = hashlib.sha256()
        h_obj.update(all_labels.tobytes())
        dr = np.round(all_dists * 1e15).astype(np.int64)
        h_obj.update(dr.tobytes())
        h = h_obj.hexdigest()
        search_hashes.append(h)
        print(f"    Search run {run+1}: {h[:16]}...")

    search_ok = len(set(search_hashes)) == 1
    print(f"  Search determinism: {'PASS' if search_ok else 'FAIL'}")

    # Error analysis
    print("\n  --- Error Analysis (I64F32 vs f64) ---")
    n_pairs = 1000
    pairs_a = rng.randint(0, min(n, 10000), n_pairs)
    pairs_b = rng.randint(0, min(n, 10000), n_pairs)
    base_sub = base[:10000]

    errors_i64_vs_f64 = []
    errors_f32_vs_f64 = []
    for i in range(n_pairs):
        a, b = int(pairs_a[i]), int(pairs_b[i])
        diff_f64 = base_sub[a].astype(np.float64) - base_sub[b].astype(np.float64)
        d_f64 = float(np.sum(diff_f64 ** 2))

        diff_fixed = base_fixed[a] - base_fixed[b]
        d_i64 = float(np.sum(
            diff_fixed.astype(np.float64) ** 2
        )) / (scale * scale)

        diff_f32 = base_sub[a].astype(np.float32) - base_sub[b].astype(np.float32)
        d_f32 = float(np.sum(diff_f32.astype(np.float64) ** 2))

        if d_f64 > 1e-20:
            errors_i64_vs_f64.append(abs(d_i64 - d_f64) / d_f64)
            errors_f32_vs_f64.append(abs(d_f32 - d_f64) / d_f64)

    if errors_i64_vs_f64:
        mean_i64 = np.mean(errors_i64_vs_f64)
        max_i64 = np.max(errors_i64_vs_f64)
        mean_f32 = np.mean(errors_f32_vs_f64)
        max_f32 = np.max(errors_f32_vs_f64)
        improvement = max_f32 / max(max_i64, 1e-30)
        print(f"  I64F32 vs f64: mean={mean_i64:.2e}, max={max_i64:.2e}")
        print(f"  f32 vs f64:    mean={mean_f32:.2e}, max={max_f32:.2e}")
        print(f"  I64F32 improvement over f32: {improvement:.0f}x")
    else:
        mean_i64 = max_i64 = mean_f32 = max_f32 = improvement = 0

    # Compile results
    results = {
        "dataset": "dbpedia-openai3-1536",
        "source": "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K",
        "n_vectors": n,
        "dims": d,
        "n_queries": nq,
        "metric": "cosine",
        "frac_bits": frac_bits,
        "value_range": {"max_abs": float(max_abs), "mean_abs": float(mean_abs)},
        "experiment_a_i64f32": {
            "float32_results": f32_results,
            "i64f32_results": i64_results,
            "overhead_ef128": round(overhead, 3),
            "recall_diff_ef128": round(
                i64_128["recall_at_10"] - f32_128["recall_at_10"], 4
            ),
        },
        "experiment_b_two_phase": tp_results,
        "experiment_c_determinism": {
            "n_runs": N_RUNS,
            "distance_hashes": dist_hashes,
            "distance_deterministic": dist_ok,
            "search_hashes": search_hashes,
            "search_deterministic": search_ok,
            "overall_pass": dist_ok and search_ok,
        },
        "error_analysis": {
            "n_pairs": n_pairs,
            "i64f32_vs_f64_mean": float(mean_i64),
            "i64f32_vs_f64_max": float(max_i64),
            "f32_vs_f64_mean": float(mean_f32),
            "f32_vs_f64_max": float(max_f32),
            "i64f32_improvement_over_f32": float(improvement),
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open("/workspace/results/llm_1536d_results.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()

    exp.log({
        "overhead": overhead,
        "recall_f32": f32_128["recall_at_10"],
        "recall_i64": i64_128["recall_at_10"],
        "determinism_pass": 1 if (dist_ok and search_ok) else 0,
        "error_i64_max": float(max_i64),
    }, step=0)

    exp.finish("completed")
    print("\n" + "=" * 60)
    print("ALL LLM EXPERIMENTS COMPLETE!")
    return results


@app.local_entrypoint()
def main():
    print("D-HNSW LLM Embeddings 1536d Experiment")
    print("=" * 60)
    results = experiment_llm_1536d.remote()

    print("\n=== SUMMARY ===")
    ea = results["experiment_a_i64f32"]
    print(f"Overhead: {ea['overhead_ef128']:.2f}x")
    print(f"Recall diff: {ea['recall_diff_ef128']:.4f}")

    ec = results["experiment_c_determinism"]
    status = "PASS" if ec["overall_pass"] else "FAIL"
    print(f"Determinism: {status}")

    err = results["error_analysis"]
    print(f"I64F32 max error: {err['i64f32_vs_f64_max']:.2e}")
    print(f"f32 max error: {err['f32_vs_f64_max']:.2e}")
    print(f"Improvement: {err['i64f32_improvement_over_f32']:.0f}x")
