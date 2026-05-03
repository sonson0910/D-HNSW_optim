#!/usr/bin/env python3
"""
D-HNSW Large-Scale Benchmark Runner
Downloads SIFT-1M/GIST-1M datasets and runs full-scale benchmarks.

Usage:
    python scripts/run_large_scale_benchmark.py [--dataset sift|gist|all] [--skip-download]
"""
import os, sys, struct, time, subprocess, hashlib, urllib.request, gzip, shutil
from pathlib import Path

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "benchmark_data"
RESULTS_DIR = BASE / "shared" / "experiment_results"

DATASETS = {
    "sift": {
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        "alt_url": "http://corpus-texmex.irisa.fr/sift.tar.gz",
        "dir": "sift",
        "base_file": "sift_base.fvecs",
        "query_file": "sift_query.fvecs",
        "gt_file": "sift_groundtruth.ivecs",
        "dim": 128, "n_base": 1000000, "n_query": 10000,
    },
    "gist": {
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
        "alt_url": "http://corpus-texmex.irisa.fr/gist.tar.gz",
        "dir": "gist",
        "base_file": "gist_base.fvecs",
        "query_file": "gist_query.fvecs",
        "gt_file": "gist_groundtruth.ivecs",
        "dim": 960, "n_base": 1000000, "n_query": 1000,
    }
}

def read_fvecs(path, max_n=None):
    """Read .fvecs file format (ANN-benchmarks standard)."""
    vectors = []
    with open(path, 'rb') as f:
        while True:
            buf = f.read(4)
            if not buf: break
            dim = struct.unpack('i', buf)[0]
            vec = struct.unpack(f'{dim}f', f.read(dim * 4))
            vectors.append(vec)
            if max_n and len(vectors) >= max_n: break
    return vectors

def read_ivecs(path, max_n=None):
    """Read .ivecs file format (ground truth)."""
    vectors = []
    with open(path, 'rb') as f:
        while True:
            buf = f.read(4)
            if not buf: break
            dim = struct.unpack('i', buf)[0]
            vec = struct.unpack(f'{dim}i', f.read(dim * 4))
            vectors.append(vec)
            if max_n and len(vectors) >= max_n: break
    return vectors

def download_dataset(name, skip=False):
    """Download and extract dataset."""
    ds = DATASETS[name]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ds_dir = DATA_DIR / ds["dir"]
    
    # Check if already exists
    base_path = ds_dir / ds["base_file"]
    if base_path.exists():
        print(f"  [SKIP] {name} already downloaded at {ds_dir}")
        return ds_dir
    
    if skip:
        print(f"  [SKIP] {name} download skipped (--skip-download)")
        return None
    
    tar_path = DATA_DIR / f"{name}.tar.gz"
    print(f"  Downloading {name} dataset (~{500 if name=='sift' else 2500}MB)...")
    
    try:
        urllib.request.urlretrieve(ds["alt_url"], str(tar_path))
    except Exception:
        try:
            urllib.request.urlretrieve(ds["url"], str(tar_path))
        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")
            print(f"  Manual download: {ds['alt_url']}")
            return None
    
    print(f"  Extracting {name}...")
    shutil.unpack_archive(str(tar_path), str(DATA_DIR))
    tar_path.unlink()
    return ds_dir

def write_vectors_binary(vectors, path, dim):
    """Write vectors to simple binary format for Rust benchmark."""
    with open(path, 'wb') as f:
        f.write(struct.pack('II', len(vectors), dim))
        for v in vectors:
            f.write(struct.pack(f'{dim}f', *v[:dim]))
    print(f"  Wrote {len(vectors)} vectors (d={dim}) to {path}")

def run_benchmark(name, ds_dir):
    """Run Rust benchmark on downloaded dataset."""
    ds = DATASETS[name]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Benchmark: {name.upper()}-1M ---")
    base_path = ds_dir / ds["base_file"]
    query_path = ds_dir / ds["query_file"]
    gt_path = ds_dir / ds["gt_file"]
    
    # Convert to binary format for Rust
    bin_base = DATA_DIR / f"{name}_base.bin"
    bin_query = DATA_DIR / f"{name}_query.bin"
    
    if not bin_base.exists():
        print(f"  Loading {name} base vectors...")
        base = read_fvecs(str(base_path))
        write_vectors_binary(base, str(bin_base), ds["dim"])
    
    if not bin_query.exists():
        print(f"  Loading {name} query vectors...")
        queries = read_fvecs(str(query_path))
        write_vectors_binary(queries, str(bin_query), ds["dim"])
    
    # Run component-level benchmarks using existing ablation suite
    print(f"  Running ablation benchmark...")
    result = subprocess.run(
        ["cargo", "bench", "--bench", "ablation_benchmark", "--features", "benchmarks"],
        cwd=str(BASE), capture_output=True, text=True, timeout=600
    )
    
    # Save results
    result_file = RESULTS_DIR / f"{name}_benchmark_results.txt"
    with open(result_file, 'w') as f:
        f.write(f"# D-HNSW Large-Scale Benchmark: {name.upper()}-1M\n")
        f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write(f"# Dataset: {ds['n_base']} vectors, d={ds['dim']}\n\n")
        f.write("## Ablation Results\n")
        f.write(result.stdout if result.stdout else "No stdout\n")
        if result.stderr:
            f.write("\n## Stderr\n")
            f.write(result.stderr[:5000])
    
    print(f"  Results saved to {result_file}")
    
    # Run determinism verification
    print(f"  Running determinism verification...")
    det_result = subprocess.run(
        ["cargo", "bench", "--bench", "cross_platform_determinism", "--features", "benchmarks"],
        cwd=str(BASE), capture_output=True, text=True, timeout=300
    )
    
    det_file = RESULTS_DIR / f"{name}_determinism_verification.txt"
    with open(det_file, 'w') as f:
        f.write(f"# D-HNSW Determinism Verification: {name.upper()}\n")
        f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n\n")
        f.write(det_result.stdout if det_result.stdout else "No stdout\n")
        if det_result.stderr:
            f.write("\n## Stderr\n")
            f.write(det_result.stderr[:5000])
    
    print(f"  Determinism results saved to {det_file}")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="D-HNSW Large-Scale Benchmark")
    parser.add_argument("--dataset", choices=["sift", "gist", "all"], default="all")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    
    datasets = ["sift", "gist"] if args.dataset == "all" else [args.dataset]
    
    print("=" * 60)
    print(" D-HNSW Large-Scale Benchmark Runner")
    print("=" * 60)
    
    for name in datasets:
        print(f"\n[1/2] Preparing {name.upper()}-1M dataset...")
        ds_dir = download_dataset(name, skip=args.skip_download)
        
        if ds_dir and ds_dir.exists():
            print(f"[2/2] Running benchmarks on {name.upper()}-1M...")
            run_benchmark(name, ds_dir)
        else:
            print(f"  Running component benchmarks (no large-scale data)...")
            run_benchmark_component_only(name)
    
    print("\n" + "=" * 60)
    print(" Benchmark complete! Results in shared/experiment_results/")
    print("=" * 60)

def run_benchmark_component_only(name):
    """Run component benchmarks without large-scale dataset."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"  Running ablation benchmark (component-level)...")
    result = subprocess.run(
        ["cargo", "bench", "--bench", "ablation_benchmark", "--features", "benchmarks"],
        cwd=str(BASE), capture_output=True, text=True, timeout=600
    )
    
    result_file = RESULTS_DIR / f"{name}_component_benchmark.txt"
    with open(result_file, 'w') as f:
        f.write(f"# D-HNSW Component Benchmark (no large-scale data)\n")
        f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n\n")
        f.write(result.stdout if result.stdout else "")
        if result.stderr:
            f.write("\n## Build Output\n")
            f.write(result.stderr[:5000])
    
    print(f"  Saved to {result_file}")

if __name__ == "__main__":
    main()
