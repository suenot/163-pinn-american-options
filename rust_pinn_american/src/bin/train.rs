//! Train PINN for American option pricing.
//!
//! Usage:
//!   cargo run --bin train -- --strike 100 --vol 0.2 --maturity 1.0 --epochs 2000

use clap::Parser;
use colored::Colorize;
use pinn_american_options::{
    pricer::AmericanPINNPricer, lsm::lsm_american_option,
    OptionParams, OptionType,
};

#[derive(Parser, Debug)]
#[command(name = "train", about = "Train PINN for American option pricing")]
struct Args {
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

    /// Number of training epochs
    #[arg(long, default_value_t = 2000)]
    epochs: usize,

    /// Learning rate
    #[arg(long, default_value_t = 0.01)]
    lr: f64,

    /// Number of interior collocation points
    #[arg(long, default_value_t = 200)]
    n_interior: usize,

    /// Hidden layer sizes (comma separated)
    #[arg(long, default_value = "16,16")]
    hidden: String,
}

fn main() {
    let args = Args::parse();

    let option_type = match args.option_type.as_str() {
        "put" => OptionType::Put,
        "call" => OptionType::Call,
        _ => {
            eprintln!("Invalid option type. Use 'put' or 'call'.");
            std::process::exit(1);
        }
    };

    let hidden_layers: Vec<usize> = args
        .hidden
        .split(',')
        .map(|s| s.trim().parse().unwrap_or(16))
        .collect();

    let params = OptionParams {
        strike: args.strike,
        risk_free_rate: args.rate,
        volatility: args.vol,
        maturity: args.maturity,
        option_type,
        s_max: 3.0 * args.strike,
    };

    println!("{}", "═".repeat(60).bright_cyan());
    println!("{}", " PINN American Option Training ".bright_cyan().bold());
    println!("{}", "═".repeat(60).bright_cyan());
    println!("  Strike:     {:.2}", params.strike);
    println!("  Rate:       {:.4}", params.risk_free_rate);
    println!("  Volatility: {:.4}", params.volatility);
    println!("  Maturity:   {:.2}y", params.maturity);
    println!("  Type:       {:?}", params.option_type);
    println!("  Hidden:     {:?}", hidden_layers);
    println!("  Epochs:     {}", args.epochs);
    println!("{}", "─".repeat(60));

    let mut pricer = AmericanPINNPricer::new(params.clone(), &hidden_layers, 1000.0);

    println!("\n{}", "Training PINN...".bright_yellow());
    let history = pricer.train(
        args.epochs,
        args.lr,
        args.n_interior,
        50,  // n_boundary
        50,  // n_terminal
        true,
    );

    // Print final loss
    if let Some(&final_loss) = history.last() {
        println!(
            "\n{}: {:.6}",
            "Final loss".bright_green(),
            final_loss
        );
    }

    // Price on a grid
    println!("\n{}", "═".repeat(60).bright_cyan());
    println!("{}", " Option Prices at t=0 ".bright_cyan().bold());
    println!("{}", "═".repeat(60).bright_cyan());
    println!("{:>10} {:>12} {:>12} {:>12}", "S", "PINN", "LSM", "Payoff");
    println!("{}", "─".repeat(50));

    let s_values: Vec<f64> = (0..11).map(|i| 50.0 + i as f64 * 10.0).collect();

    for &s in &s_values {
        let pinn_price = pricer.price(s, 0.0);
        let lsm_result = lsm_american_option(s, &params, 50, 20000);
        let payoff = params.payoff(s);

        println!(
            "{:>10.2} {:>12.4} {:>12.4} {:>12.4}",
            s, pinn_price, lsm_result.price, payoff
        );
    }

    // Exercise boundary
    println!("\n{}", "═".repeat(60).bright_cyan());
    println!("{}", " Exercise Boundary ".bright_cyan().bold());
    println!("{}", "═".repeat(60).bright_cyan());

    let t_values: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
    let boundary = pricer.find_exercise_boundary(&t_values, 500);

    println!("{:>10} {:>12}", "t", "S*(t)");
    println!("{}", "─".repeat(25));
    for (t, s_star) in t_values.iter().zip(boundary.iter()) {
        println!("{:>10.2} {:>12.2}", t, s_star);
    }

    println!("\n{}", "Training complete!".bright_green().bold());
}
