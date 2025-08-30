//! Fetch crypto market data from Bybit for option pricing.
//!
//! Usage:
//!   cargo run --bin fetch_data -- --symbol BTCUSDT --interval D --limit 365

use clap::Parser;
use colored::Colorize;
use pinn_american_options::data::{BybitClient, generate_synthetic_data};

#[derive(Parser, Debug)]
#[command(name = "fetch_data", about = "Fetch Bybit crypto data for option pricing")]
struct Args {
    /// Trading pair
    #[arg(long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval (1, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
    #[arg(long, default_value = "D")]
    interval: String,

    /// Number of candles
    #[arg(long, default_value_t = 200)]
    limit: usize,

    /// Use synthetic data instead of API
    #[arg(long, default_value_t = false)]
    synthetic: bool,

    /// Output CSV file
    #[arg(long, default_value = "market_data.csv")]
    output: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("{}", "═".repeat(60).bright_cyan());
    println!("{}", " Bybit Market Data Fetcher ".bright_cyan().bold());
    println!("{}", "═".repeat(60).bright_cyan());

    if args.synthetic {
        println!(
            "\n{}",
            "Using synthetic data (GBM simulation)...".bright_yellow()
        );

        let prices = generate_synthetic_data(50000.0, 0.0, 0.6, args.limit);

        // Compute volatility
        let log_returns: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        let mean_ret: f64 =
            log_returns.iter().sum::<f64>() / log_returns.len() as f64;
        let var: f64 = log_returns
            .iter()
            .map(|r| (r - mean_ret).powi(2))
            .sum::<f64>()
            / (log_returns.len() - 1) as f64;
        let vol = var.sqrt() * (365.0_f64).sqrt();

        println!("  Symbol:     {}", args.symbol);
        println!("  Data points: {}", prices.len());
        println!("  Start:      {:.2}", prices[0]);
        println!("  End:        {:.2}", prices.last().unwrap());
        println!("  Volatility: {:.4}", vol);

        // Save to CSV
        let mut wtr = csv::Writer::from_path(&args.output)?;
        wtr.write_record(&["index", "price"])?;
        for (i, p) in prices.iter().enumerate() {
            wtr.write_record(&[i.to_string(), format!("{:.2}", p)])?;
        }
        wtr.flush()?;
        println!("\n  Saved to: {}", args.output.bright_green());
    } else {
        println!(
            "\n  Fetching {} {} candles for {}...",
            args.limit,
            args.interval,
            args.symbol.bright_yellow()
        );

        let client = BybitClient::new();

        match client
            .fetch_market_data(&args.symbol, &args.interval, args.limit)
            .await
        {
            Ok(data) => {
                println!("\n{}", "  Market Data Summary".bright_green().bold());
                println!("  Symbol:       {}", data.symbol);
                println!("  Candles:      {}", data.candles.len());
                println!("  Current:      {:.2}", data.current_price);
                println!("  Volatility:   {:.4} ({:.1}%)", data.volatility, data.volatility * 100.0);

                if !data.candles.is_empty() {
                    let high = data
                        .candles
                        .iter()
                        .map(|c| c.high)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let low = data
                        .candles
                        .iter()
                        .map(|c| c.low)
                        .fold(f64::INFINITY, f64::min);
                    println!("  Period high:  {:.2}", high);
                    println!("  Period low:   {:.2}", low);
                }

                // Save to CSV
                let mut wtr = csv::Writer::from_path(&args.output)?;
                wtr.write_record(&[
                    "timestamp", "open", "high", "low", "close", "volume",
                ])?;
                for candle in &data.candles {
                    wtr.write_record(&[
                        candle.timestamp.to_string(),
                        format!("{:.2}", candle.open),
                        format!("{:.2}", candle.high),
                        format!("{:.2}", candle.low),
                        format!("{:.2}", candle.close),
                        format!("{:.2}", candle.volume),
                    ])?;
                }
                wtr.flush()?;
                println!("\n  Saved to: {}", args.output.bright_green());

                // Suggested option parameters
                println!("\n{}", "  Suggested Option Parameters".bright_yellow().bold());
                println!("  Strike (ATM): {:.2}", data.current_price);
                println!("  Volatility:   {:.4}", data.volatility);
                println!(
                    "  Example: cargo run --bin train -- --strike {:.0} --vol {:.2}",
                    data.current_price, data.volatility
                );
            }
            Err(e) => {
                println!(
                    "\n  {} {}",
                    "Error:".bright_red(),
                    e
                );
                println!("  Falling back to synthetic data...");

                let prices = generate_synthetic_data(50000.0, 0.0, 0.6, args.limit);
                println!("  Generated {} synthetic prices", prices.len());

                let mut wtr = csv::Writer::from_path(&args.output)?;
                wtr.write_record(&["index", "price"])?;
                for (i, p) in prices.iter().enumerate() {
                    wtr.write_record(&[i.to_string(), format!("{:.2}", p)])?;
                }
                wtr.flush()?;
                println!("  Saved to: {}", args.output.bright_green());
            }
        }
    }

    println!("\n{}", "═".repeat(60).bright_cyan());
    Ok(())
}
