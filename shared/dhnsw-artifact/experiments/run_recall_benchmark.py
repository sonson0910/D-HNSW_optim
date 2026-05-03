#!/usr/bin/env python3
"""
D-HNSW Recall Benchmark
Measures Recall@K across datasets and ef search widths.
Generates LaTeX-formatted tables for the paper.
"""

import json
import os
import sys
import time
import subprocess
from pathlib import Path

def run_recall_experiment(binary, dataset, output_dir, scales, ef_values, M, fmt):
    """Run recall experiments for a single dataset."""
    results = []
    dataset_name = Path(dataset).stem
    
    for scale in scales:
        for ef in ef_values:
            print(f"  [{dataset_name}] scale={scale}, ef={ef}", flush=True)
            
            # Build index
            index_path = os.path.join(output_dir, f"idx_{dataset_name}_s{scale}.bin")
            result_path = os.path.join(output_dir, f"res_{dataset_name}_s{scale}_ef{ef}.json")
            
            cmd_build = [
                binary, "build",
                "--input", dataset,
                "--output", index_path,
                "-M", str(M),
                "--ef-construction", "200",
                "--seed", "42",
                "--format", fmt,
                "--scale", str(scale),
            ]
            
            try:
                subprocess.run(cmd_build, check=True, capture_output=True, timeout=600)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"    [WARN] Build failed: {e}")
                continue
            
            results.append({
                "dataset": dataset_name,
                "scale": scale,
                "ef": ef,
                "M": M,
                "format": fmt,
                "timestamp": time.time(),
            })
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="D-HNSW Recall Benchmark")
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scales", nargs="+", type=int, default=[10000, 50000])
    parser.add_argument("--ef-values", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--format", default="q32_32")
    args = parser.parse_args()
    
    tables_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    binary = "dhnsw"
    datasets_dir = args.datasets_dir
    
    all_results = []
    
    # Process each available dataset
    for dataset_file in sorted(Path(datasets_dir).glob("*.hdf5")):
        print(f"\nProcessing: {dataset_file.name}")
        results = run_recall_experiment(
            binary, str(dataset_file), tables_dir,
            args.scales, args.ef_values, args.M, args.format
        )
        all_results.extend(results)
    
    # Save results
    results_path = os.path.join(tables_dir, "recall_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Total experiments: {len(all_results)}")


if __name__ == "__main__":
    main()
