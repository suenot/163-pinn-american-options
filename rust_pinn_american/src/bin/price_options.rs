//! Price American options and compute Greeks.
//!
//! Usage:
//!   cargo run --bin price_options -- --spot 100 --strike 100

use clap::Parser;
use colored::Colorize;
use pinn_american_options::{
    pricer::AmericanPINNPricer,
    greeks::GreeksComputer,
    lsm::lsm_american_option,
    OptionParams, OptionType,
};

#[derive(Parser, Debug)]
#[command(name = "price_options", about = "Price American options with PINN")]
struct Args {
    /// Spot price
    #[arg(long, default_value_t = 100.0)]
    spot: f64,

    /// Strike price
    #[arg(long, default_value_t = 100.0)]
    strike: f64,

    /// Risk-free rate
    #[arg(long, default_value_t = 0.05)]
    rate: f64,

    /// Volatility
    #[arg(long, default_value_t = 0.2)]
    vol: f64,

    /// Maturity (years)
    #[arg(long, default_value_t = 1.0)]
    maturity: f64,

    /// Option type: put or call
    #[arg(long, default_value = "put")]
    option_type: String,

    /// Training epochs
    #[arg(long, default_value_t = 2000)]
    epochs: usize,
}

fn main() {
    let args = Args::parse();

    let option_type = match args.option_type.as_str() {
        "put" => OptionType::Put,
        "call" => OptionType::Call,
        _ => {
            eprintln!("Invalid option type.");
            std::process::exit(1);
        }
    };

    let params = OptionParams {
        strike: args.strike,
        risk_free_rate: args.rate,
        volatility: args.vol,
        maturity: args.maturity,
        option_type,
        s_max: 3.0 * args.strike,
    };

    println!("{}", "═".repeat(60).bright_cyan());
    println!("{}", " American Option Pricing ".bright_cyan().bold());
    println!("{}", "═".repeat(60).bright_cyan());

    // Train PINN
    println!("\n{}", "Training PINN model...".bright_yellow());
    let mut pricer = AmericanPINNPricer::new(params.clone(), &[16, 16], 1000.0);
    pricer.train(args.epochs, 0.01, 200, 50, 50, false);
    println!("{}", "Training complete.".bright_green());

    // PINN price
    let pinn_price = pricer.price(args.spot, 0.0);

    // LSM benchmark
    let lsm_result = lsm_american_option(args.spot, &params, 100, 100000);

    // Greeks
    let greeks = GreeksComputer::compute(&pricer, args.spot, 0.0);

    println!("\n{}", "═".repeat(60).bright_cyan());
    println!("{}", " Results ".bright_cyan().bold());
    println!("{}", "═".repeat(60).bright_cyan());
    println!("  Spot:       {:>12.4}", args.spot);
    println!("  Strike:     {:>12.4}", args.strike);
    println!("  Type:       {:>12?}", option_type);
    println!("{}", "─".repeat(40));
    println!(
        "  {}:  {:>12.4}",
        "PINN Price".bright_green(),
        pinn_price
    );
    println!(
        "  {}:   {:>12.4} (+- {:.4})",
        "LSM Price".bright_blue(),
        lsm_result.price,
        lsm_result.std_error
    );
    println!(
        "  Payoff:     {:>12.4}",
        params.payoff(args.spot)
    );
    println!("{}", "─".repeat(40));
    println!("{}", " Greeks ".bright_yellow().bold());
    println!("  Delta:      {:>12.6}", greeks.delta);
    println!("  Gamma:      {:>12.6}", greeks.gamma);
    println!("  Theta:      {:>12.6}", greeks.theta);
    println!("  Vega:       {:>12.6}", greeks.vega);
    println!("  Rho:        {:>12.6}", greeks.rho);
    println!("{}", "═".repeat(60).bright_cyan());
}
