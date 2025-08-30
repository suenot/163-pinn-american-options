use criterion::{criterion_group, criterion_main, Criterion};
use pinn_american_options::{
    pricer::AmericanPINNPricer,
    lsm::lsm_american_option,
    OptionParams, OptionType,
};

fn bench_pinn_pricing(c: &mut Criterion) {
    let params = OptionParams {
        strike: 100.0,
        risk_free_rate: 0.05,
        volatility: 0.2,
        maturity: 1.0,
        option_type: OptionType::Put,
        s_max: 300.0,
    };

    let pricer = AmericanPINNPricer::new(params.clone(), &[16, 16], 1000.0);

    c.bench_function("pinn_single_price", |b| {
        b.iter(|| pricer.price(100.0, 0.5))
    });

    c.bench_function("lsm_10k_paths", |b| {
        b.iter(|| lsm_american_option(100.0, &params, 50, 10000))
    });
}

criterion_group!(benches, bench_pinn_pricing);
criterion_main!(benches);
