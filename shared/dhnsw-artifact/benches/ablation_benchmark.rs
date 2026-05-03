// ============================================================
//  D-HNSW Ablation Benchmark
//  Compares Q32.32 vs Q16.16 vs f32 recall and latency
// ============================================================

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn dummy_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ablation");
    
    for &dim in &[32, 64, 128, 256] {
        group.bench_with_input(
            BenchmarkId::new("q32_32_distance", dim),
            &dim,
            |b, &d| {
                // Generate two random Q32.32 vectors
                let a: Vec<i64> = (0..d).map(|i| (i as i64) * 1000).collect();
                let b_vec: Vec<i64> = (0..d).map(|i| (i as i64) * 1001).collect();
                
                b.iter(|| {
                    let mut sum: i64 = 0;
                    for k in 0..d {
                        let diff = a[k].saturating_sub(b_vec[k]);
                        sum = sum.saturating_add(diff.saturating_mul(diff));
                    }
                    sum
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("f32_distance", dim),
            &dim,
            |b, &d| {
                let a: Vec<f32> = (0..d).map(|i| i as f32 * 0.001).collect();
                let b_vec: Vec<f32> = (0..d).map(|i| i as f32 * 0.00101).collect();
                
                b.iter(|| {
                    let mut sum: f32 = 0.0;
                    for k in 0..d {
                        let diff = a[k] - b_vec[k];
                        sum += diff * diff;
                    }
                    sum
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, dummy_benchmark);
criterion_main!(benches);
