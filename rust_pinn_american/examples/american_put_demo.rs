//! Demo: Price an American put option with PINN and compare to LSM.
//!
//! Run with: cargo run --example american_put_demo

use pinn_american_options::{
    pricer::AmericanPINNPricer,
    lsm::lsm_american_option,
    greeks::GreeksComputer,
    OptionParams, OptionType,
};

fn main() {
    println!("American Put Option Pricing Demo");
    println!("================================\n");

    let params = OptionParams {
        strike: 100.0,
        risk_free_rate: 0.05,
        volatility: 0.2,
        maturity: 1.0,
        option_type: OptionType::Put,
        s_max: 300.0,
    };

    // Train PINN
    println!("Training PINN (this may take a moment)...\n");
    let mut pricer = AmericanPINNPricer::new(params.clone(), &[16, 16], 1000.0);
    let history = pricer.train(1000, 0.01, 100, 30, 30, true);
    println!("\nFinal loss: {:.6}\n", history.last().unwrap_or(&0.0));

    // Price comparison
    println!("{:>8} {:>10} {:>10} {:>10} {:>8} {:>8}", "S", "PINN", "LSM", "Payoff", "Delta", "Gamma");
    println!("{}", "-".repeat(60));

    for s in [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0] {
        let pinn_v = pricer.price(s, 0.0);
        let lsm = lsm_american_option(s, &params, 50, 20000);
        let payoff = params.payoff(s);
        let greeks = GreeksComputer::compute(&pricer, s, 0.0);

        println!(
            "{:>8.1} {:>10.4} {:>10.4} {:>10.4} {:>8.4} {:>8.4}",
            s, pinn_v, lsm.price, payoff, greeks.delta, greeks.gamma
        );
    }

    println!("\nNote: PINN accuracy improves with more training epochs");
    println!("and larger network architectures.");
}
