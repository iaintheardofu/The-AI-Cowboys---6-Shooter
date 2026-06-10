use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn ntt_benchmark(c: &mut Criterion) {
    use yield_daemon::zk_prover::ntt::NttDomain;
    use yield_daemon::zk_prover::montgomery::MontgomeryU256;

    let mut group = c.benchmark_group("NTT");

    for log_n in [8, 10, 12, 14, 16] {
        let domain = NttDomain::new(log_n);
        let n = 1 << log_n;

        group.bench_with_input(
            BenchmarkId::new("forward", format!("2^{}", log_n)),
            &n,
            |b, &n| {
                let mut data: Vec<MontgomeryU256> = (0..n)
                    .map(|i| MontgomeryU256::from_u64(i as u64 + 1))
                    .collect();
                b.iter(|| {
                    domain.forward(&mut data);
                });
            },
        );
    }

    group.finish();
}

fn montgomery_benchmark(c: &mut Criterion) {
    use yield_daemon::zk_prover::montgomery::MontgomeryU256;

    let mut group = c.benchmark_group("Montgomery");

    let a = MontgomeryU256::from_u64(0xDEADBEEF);
    let b = MontgomeryU256::from_u64(0xCAFEBABE);

    group.bench_function("mul", |bench| {
        bench.iter(|| a.mont_mul(&b));
    });

    group.bench_function("sqr", |bench| {
        bench.iter(|| a.mont_sqr());
    });

    group.bench_function("add", |bench| {
        bench.iter(|| a.mont_add(&b));
    });

    group.finish();
}

fn arena_benchmark(c: &mut Criterion) {
    use yield_daemon::memory::arena::Arena;

    let mut group = c.benchmark_group("Arena");

    group.bench_function("alloc_64B_1000x", |bench| {
        let arena = Arena::new(1024 * 1024);
        bench.iter(|| {
            for _ in 0..1000 {
                arena.alloc(64, 8).unwrap();
            }
            arena.reset();
        });
    });

    group.bench_function("reset", |bench| {
        let arena = Arena::new(1024 * 1024);
        bench.iter(|| {
            arena.reset();
        });
    });

    group.finish();
}

criterion_group!(benches, ntt_benchmark, montgomery_benchmark, arena_benchmark);
criterion_main!(benches);
