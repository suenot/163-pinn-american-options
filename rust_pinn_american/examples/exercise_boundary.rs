//! Demo: Compute and display the optimal exercise boundary for an American put.
//!
//! Run with: cargo run --example exercise_boundary

use pinn_american_options::{
    pricer::AmericanPINNPricer,
    lsm::lsm_american_option,
    OptionParams, OptionType,
};

fn main() {
    println!("American Put Exercise Boundary Demo");
    println!("===================================\n");

    let params = OptionParams {
        strike: 100.0,
        risk_free_rate: 0.05,
        volatility: 0.2,
        maturity: 1.0,
        option_type: OptionType::Put,
        s_max: 300.0,
    };

    // Train PINN
    println!("Training PINN...\n");
    let mut pricer = AmericanPINNPricer::new(params.clone(), &[16, 16], 1000.0);
    pricer.train(1500, 0.01, 150, 40, 40, false);

    // Compute exercise boundary
    let n_t = 20;
    let t_values: Vec<f64> = (0..n_t).map(|i| i as f64 / n_t as f64).collect();
    let pinn_boundary = pricer.find_exercise_boundary(&t_values, 500);

    // LSM boundary for comparison
    let lsm_result = lsm_american_option(100.0, &params, 100, 50000);
    let lsm_boundary = &lsm_result.exercise_boundary;

    println!("Optimal Exercise Boundary S*(t)");
    println!("For American Put: exercise when S < S*(t)\n");
    println!("{:>8} {:>12} {:>12}", "t", "PINN S*(t)", "LSM S*(t)");
    println!("{}", "-".repeat(35));

    for (i, &t) in t_values.iter().enumerate() {
        let lsm_step = (t * 100.0) as usize;
        let lsm_val = if lsm_step < lsm_boundary.len() && !lsm_boundary[lsm_step].is_nan() {
            format!("{:.2}", lsm_boundary[lsm_step])
        } else {
            "N/A".to_string()
        };

        println!("{:>8.3} {:>12.2} {:>12}", t, pinn_boundary[i], lsm_val);
    }

    println!("\nInterpretation:");
    println!("  - The exercise boundary separates the (S, t) plane into two regions:");
    println!("  - EXERCISE region: below the boundary (for puts)");
    println!("  - CONTINUATION region: above the boundary (for puts)");
    println!("  - At maturity (t -> T), S*(T) -> K for a put option");
    println!("  - Strike K = {:.2}", params.strike);

    // ASCII visualization
    println!("\nExercise Boundary (ASCII plot):");
    println!();
    let max_s = params.strike * 1.1;
    let rows = 15;
    let cols = 40;

    for row in (0..rows).rev() {
        let s_val = (row as f64 / rows as f64) * max_s;
        print!("{:>6.1} |", s_val);
        for col in 0..cols {
            let t_frac = col as f64 / cols as f64;
            // Find closest boundary value
            let idx = ((t_frac * n_t as f64) as usize).min(n_t - 1);
            let boundary_s = pinn_boundary[idx];

            if (s_val - boundary_s).abs() < max_s / rows as f64 {
                print!("*");
            } else if s_val < boundary_s {
                print!(".");
            } else {
                print!(" ");
            }
        }
        println!();
    }
    print!("       +");
    for _ in 0..cols {
        print!("-");
    }
    println!();
    println!("        t=0{:>37}", "t=T");
    println!("\n  *: Exercise boundary  .: Exercise region  (space): Continue");
}
