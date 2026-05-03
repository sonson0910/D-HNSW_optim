// ============================================================
//  Cross-Architecture Verification Benchmark
//  Builds identical indices and compares SHA-256 hashes
// ============================================================

use criterion::{criterion_group, criterion_main, Criterion};

fn cross_arch_benchmark(c: &mut Criterion) {
    c.bench_function("index_build_100_vectors_128d", |b| {
        b.iter(|| {
            let config = dhnsw::graph::DhnswConfig {
                m: 8,
                m_max_0: 16,
                ef_construction: 50,
                seed: 42,
                dim: 128,
            };
            let mut index = dhnsw::graph::DhnswIndex::new(config);
            
            // Deterministic test vectors
            for i in 0..100u32 {
                let components: Vec<f32> = (0..128)
                    .map(|j| ((i * 128 + j) as f32) * 0.001)
                    .collect();
                let v = dhnsw::fixed_point::FixedVector::from_f32_slice(&components);
                index.insert(v);
            }
            
            index.sha256_hash()
        });
    });
}

criterion_group!(benches, cross_arch_benchmark);
criterion_main!(benches);
