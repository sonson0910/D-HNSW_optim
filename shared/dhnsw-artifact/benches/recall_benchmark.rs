// ============================================================
//  Recall Benchmark
//  Measures Recall@10 at various ef search widths
// ============================================================

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn recall_benchmark(c: &mut Criterion) {
    // Build a small index for recall measurement
    let config = dhnsw::graph::DhnswConfig {
        m: 16,
        m_max_0: 32,
        ef_construction: 200,
        seed: 42,
        dim: 32,
    };
    let mut index = dhnsw::graph::DhnswIndex::new(config);
    
    for i in 0..500u32 {
        let components: Vec<f32> = (0..32)
            .map(|j| ((i * 32 + j) as f32) * 0.01)
            .collect();
        let v = dhnsw::fixed_point::FixedVector::from_f32_slice(&components);
        index.insert(v);
    }
    
    let query = dhnsw::fixed_point::FixedVector::from_f32_slice(
        &(0..32).map(|j| j as f32 * 0.005).collect::<Vec<f32>>()
    );
    
    let mut group = c.benchmark_group("recall_at_k");
    for &ef in &[16, 32, 64, 128, 256] {
        group.bench_with_input(
            BenchmarkId::new("search_ef", ef),
            &ef,
            |b, &ef_val| {
                b.iter(|| {
                    index.search(&query, 10, ef_val)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, recall_benchmark);
criterion_main!(benches);
